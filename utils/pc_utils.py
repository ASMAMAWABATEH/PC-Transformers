import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — helper that returns dE/dmu depending on energy function.
#
#   MSE energy:  E = ½ (target − μ)²   →  ∂E/∂μ = (target − μ) = bu_err
#   MAE energy:  E = |target − μ|      →  ∂E/∂μ = sign(target − μ) = sign(bu_err)
#
# Every other formula in the file is identical for both energy functions.
# Only swap this one value; the chain continues unchanged downstream.
# ─────────────────────────────────────────────────────────────────────────────
def compute_dE_dmu(bu_err: torch.Tensor, energy_fn_name: str) -> torch.Tensor:
    """
    Returns dE/dmu for the given energy function.

    MSE:  dE/dmu = bu_err          (gradient proportional to error size)
    MAE:  dE/dmu = sign(bu_err)    (gradient is always ±1, never scales)
    """
    if energy_fn_name == "mae":
        return torch.sign(bu_err)
    return bu_err   # default: mse / pc_e


def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device=device)


def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)

    cos = torch.zeros(end, dim)
    sin = torch.zeros(end, dim)
    cos[:, 0::2] = freqs.cos()
    cos[:, 1::2] = freqs.cos()
    sin[:, 0::2] = freqs.sin()
    sin[:, 1::2] = freqs.sin()

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq, xk, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


def rotate_half_transpose(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    The embedding update now follows the same energy rule as the other
    layers: MSE uses the raw residual, MAE uses sign(residual).
    """
    word_layer: nn.Embedding = layer["word"]
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)

    mu_word = word_layer(input_ids)
    mu = mu_word
    mu_norm = layer_norm(mu) if layer_norm is not None else mu
    error = target - mu_norm
    dE_dmu = compute_dE_dmu(error, energy_fn_name)

    if requires_update:
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = dE_dmu.reshape(-1, dE_dmu.size(-1))
            delta = torch.clamp(local_lr * flat_update, -0.01, 0.01)
            word_layer.weight.data.index_add_(0, flat_input_ids, delta)

    return mu, mu_word, error


def step_linear(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Module,
    lateral_conn: Optional[Any],
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
):
    """
    Predictive coding step for linear-like layers.

    CHANGE 2 (non-output layers):
        Before:  dE_dmu = bu_err
        After:   dE_dmu = compute_dE_dmu(bu_err, energy_fn_name)
        ── MSE keeps bu_err unchanged.
        ── MAE replaces it with sign(bu_err).

    CHANGE 3 (output / softmax layer):
        Before:  dE_dp = bu_err
        After:   dE_dp = compute_dE_dmu(bu_err, energy_fn_name)
        ── The softmax VJP formula below is the SAME for both energies;
           only the value fed into it changes.
    """
    if layer_norm is not None and layer_type == "fc1":
        x_input = layer_norm(x)
    elif layer_type == "fc2":
        x_input = F.gelu(x)
    else:
        x_input = x

    mu = layer(x_input)

    if layer_type == "fc1":
        mu = F.gelu(mu)
    elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
        mu = layer_norm(mu)

    if layer_type == "linear_output":
        probs = F.softmax(mu, dim=-1)
        bu_err = target - probs

        # ── CHANGE 3 ──────────────────────────────────────────────────────────
        # MSE: dE_dp = bu_err       (error magnitude drives the signal)
        # MAE: dE_dp = sign(bu_err) (only direction matters, not magnitude)
        # The softmax VJP that follows is unchanged for both.
        dE_dp = compute_dE_dmu(bu_err, energy_fn_name)
        # ─────────────────────────────────────────────────────────────────────

        norm_term = (dE_dp * probs).sum(dim=-1, keepdim=True)
        dE_dmu = probs * (dE_dp - norm_term)   # softmax VJP — same formula always
        error_proj = dE_dmu @ layer.weight

    else:
        bu_err = target - mu

        # ── CHANGE 2 ──────────────────────────────────────────────────────────
        # MSE: dE_dmu = bu_err       (gradient ∝ how wrong we are)
        # MAE: dE_dmu = sign(bu_err) (gradient is always ±1)
        dE_dmu = compute_dE_dmu(bu_err, energy_fn_name)
        # ─────────────────────────────────────────────────────────────────────

        error_proj = dE_dmu @ layer.weight

    error = error_proj - td_err if td_err is not None else error_proj

    if lateral_conn is not None:
        x = x + local_lr * lateral_conn.forward(x, error)
        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", dE_dmu, x_input.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * dE_dmu.mean(dim=(0, 1))
            layer.bias.data.add_(torch.clamp(delta_b, -0.01, 0.01))

    return x, mu, bu_err


def step_attn(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    rope_cache: Tuple[torch.Tensor, torch.Tensor],
    flash: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
):
    """
    Predictive coding step for attention using Sine-Cosine RoPE.

    CHANGE 4 (attention output):
        Before:  dE_dmu = bu_err
        After:   dE_dmu = compute_dE_dmu(bu_err, energy_fn_name)
        ── Everything downstream (dE/dV, softmax VJP, dE/dQ, dE/dK,
           RoPE unrotation, weight updates) is IDENTICAL for both
           energies — only the seed value changes here.
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"

    device = x.device
    x_norm = layer_norm(x) if layer_norm is not None else x

    q_proj = proj_layers["q_proj"]
    k_proj = proj_layers["k_proj"]
    v_proj = proj_layers["v_proj"]
    assert q_proj is not None and k_proj is not None and v_proj is not None

    B, S, E = target.shape
    head_dim = n_embed // num_heads

    Q_raw = q_proj(x_norm).view(B, num_heads, S, head_dim)
    K_raw = k_proj(x_norm).view(B, num_heads, S, head_dim)
    V_raw = v_proj(x_norm).view(B, num_heads, S, head_dim)

    Q = Q_raw.clone()
    K_new = K_raw.clone()
    V_new = V_raw

    cos, sin = rope_cache
    cos = cos.to(device)
    sin = sin.to(device)

    Q, K_new = apply_rotary_emb(Q, K_new, cos[:S], sin[:S])

    if use_cache and kv_cache is not None:
        K_cached, V_cached = kv_cache
        K = torch.cat([K_cached, K_new], dim=2)
        V = torch.cat([V_cached, V_new], dim=2)
    else:
        K, V = K_new, V_new

    causal_mask = torch.tril(
        torch.ones(S, K.size(2), device=device)
    ).unsqueeze(0).unsqueeze(0)

    if flash:
        mu_heads = apply_flash_attention(Q, K, V, mask=causal_mask)
    else:
        mu_heads, attn_weights, attn_scores = apply_standard_attention(Q, K, V, mask=causal_mask)

    mu = mu_heads.transpose(1, 2).contiguous().view(B, S, E)
    bu_err = target - mu

    error = bu_err - td_err if td_err is not None else bu_err

    scale = 1.0 / (head_dim ** 0.5)

    # ── CHANGE 4 ──────────────────────────────────────────────────────────────
    # This is the only line that changes between MSE and MAE.
    # MSE: dE_dmu = bu_err          → gradient scales with error size
    # MAE: dE_dmu = sign(bu_err)    → gradient is always ±1 per element
    #
    # Every formula below receives dE_dmu and propagates it unchanged —
    # the chain rule structure is identical for both energy functions.
    dE_dmu = compute_dE_dmu(bu_err, energy_fn_name)
    # ──────────────────────────────────────────────────────────────────────────

    dE_dmu_heads = dE_dmu.view(B, num_heads, S, head_dim)

    # dE/dV = A^T @ dE/dmu  (unchanged)
    dE_dV = torch.matmul(attn_weights.transpose(-2, -1), dE_dmu_heads)

    # dE/dA = dE/dmu @ V^T  (unchanged)
    dE_dA = torch.matmul(dE_dmu_heads, V.transpose(-2, -1))

    # Softmax VJP  (unchanged — same formula for MSE and MAE)
    norm_term = (dE_dA * attn_weights).sum(dim=-1, keepdim=True)
    dE_dS = attn_weights * (dE_dA - norm_term)

    if causal_mask is not None:
        dE_dS = dE_dS.masked_fill(causal_mask == 0, 0.0)

    # dE/dQ and dE/dK  (unchanged)
    dE_dQ = torch.matmul(dE_dS, K) * scale
    dE_dK = torch.matmul(dE_dS.transpose(-2, -1), Q) * scale

    # RoPE unrotation  (unchanged)
    cos_q = cos[:S].unsqueeze(0).unsqueeze(0)
    sin_q = sin[:S].unsqueeze(0).unsqueeze(0)
    K_len = K.size(2)
    cos_k = cos[:K_len].unsqueeze(0).unsqueeze(0)
    sin_k = sin[:K_len].unsqueeze(0).unsqueeze(0)

    dE_dQ_raw = (dE_dQ * cos_q) + (rotate_half_transpose(dE_dQ) * sin_q)
    dE_dK_raw = (dE_dK * cos_k) + (rotate_half_transpose(dE_dK) * sin_k)

    delta_x = torch.zeros_like(x_norm)

    for h in range(num_heads):
        dq = dE_dQ_raw[:, h]
        dk = dE_dK_raw[:, h]
        dv = dE_dV[:, h]

        if dk.size(1) > S:
            dk = dk[:, -S:, :]
            dv = dv[:, -S:, :]

        wq = q_proj.weight[:, h * head_dim:(h + 1) * head_dim]
        wk = k_proj.weight[:, h * head_dim:(h + 1) * head_dim]
        wv = v_proj.weight[:, h * head_dim:(h + 1) * head_dim]

        delta_q = torch.einsum('bsh,eh->bse', dq, wq)
        delta_k = torch.einsum('bsh,eh->bse', dk, wk)
        delta_v = torch.einsum('bsh,eh->bse', dv, wv)

        delta_x += delta_q + delta_k + delta_v

    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, delta_x)
        x = x + local_lr * delta_x
        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x = x + local_lr * delta_x

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        with torch.no_grad():
            for h in range(num_heads):
                dW_q = torch.einsum("bte,btd->ed", x_norm, dE_dQ_raw[:, h])
                dW_k = torch.einsum("bte,btd->ed", x_norm, dE_dK_raw[:, h])
                dW_v = torch.einsum("bte,btd->ed", x_norm, dE_dV[:, h])

                dW_q = torch.clamp(dW_q, -0.01, 0.01)
                dW_k = torch.clamp(dW_k, -0.01, 0.01)
                dW_v = torch.clamp(dW_v, -0.01, 0.01)

                q_proj.weight.data[:, h * head_dim:(h + 1) * head_dim] += local_lr * dW_q
                k_proj.weight.data[:, h * head_dim:(h + 1) * head_dim] += local_lr * dW_k
                v_proj.weight.data[:, h * head_dim:(h + 1) * head_dim] += local_lr * dW_v

                if update_bias:
                    if q_proj.bias is not None:
                        db_q = torch.clamp(dE_dQ_raw[:, h].mean(dim=(0, 1)), -0.01, 0.01)
                        q_proj.bias.data[h * head_dim:(h + 1) * head_dim] += local_lr * db_q
                    if k_proj.bias is not None:
                        db_k = torch.clamp(dE_dK_raw[:, h].mean(dim=(0, 1)), -0.01, 0.01)
                        k_proj.bias.data[h * head_dim:(h + 1) * head_dim] += local_lr * db_k
                    if v_proj.bias is not None:
                        db_v = torch.clamp(dE_dV[:, h].mean(dim=(0, 1)), -0.01, 0.01)
                        v_proj.bias.data[h * head_dim:(h + 1) * head_dim] += local_lr * db_v

    new_kv_cache = (K.detach(), V.detach()) if use_cache else None
    return x, mu, bu_err, new_kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 5 — add "mae" to the energy function registry.
#
# Note: ENERGY_FUNCTIONS is used only by finalize_step to compute a scalar
# energy value for logging. It does NOT drive the gradients — that is handled
# by compute_dE_dmu above. So adding the entry here just lets you log the
# correct MAE energy value during training.
# ─────────────────────────────────────────────────────────────────────────────
ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,
    "mae":  lambda mu, x: torch.abs(mu - x),          # <── new
    "kld":  lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}


def energy_fn(mu: torch.Tensor, x: torch.Tensor, energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(
            f"Unknown energy function: {energy_fn_name}. "
            f"Choose from {list(ENERGY_FUNCTIONS.keys())}"
        )
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)


def finalize_step(mu, target, error, t, layer_type, energy_fn_name):
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = float(energy_fn(mu, target, energy_fn_name).mean().item())
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors


def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    device = input_ids.device
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
