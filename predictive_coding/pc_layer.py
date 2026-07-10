import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from utils.optim.optim_utils import PCOptimizer
from predictive_coding.lateral_connc import LateralConnections
from utils.config_utils import load_best_config

class PCLayer(nn.Module):
    """
    Predictive Coding Layer wrapper that manages iterative inference state and
    delegates computation to helper functions (step_embed, step_attn, step_linear).
    """
    def __init__(
        self,
        T: int,
        lr: float,
        inference_lr: float,
        energy_fn_name: str,
        use_precision_weighting: bool = True,
        precision_lr: float = 1e-7,
        # Added output_energy_fn_name
        output_energy_fn_name: str = "ce",
        num_heads: Optional[int] = None,
        n_embed: Optional[int] = None,
        optimizer_name: str = "adam",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        optimizer_momentum: float = 0.9,
        optimizer_weight_decay: float = 0.01,
    ):
        super().__init__()
        best_config = load_best_config()
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.T = T
        self.local_lr = lr
        self.inference_lr = inference_lr
        self.clamp_value = best_config["clamp_value"]
        self.clip_value = best_config["clip_value"]
        self.energy_fn_name = energy_fn_name
        # Store output energy fn name separately
        self.output_energy_fn_name = output_energy_fn_name    # "ce"   — output layer
        self.num_heads = num_heads
        self.n_embed = n_embed

        # Precision weighting parameters
        self.use_precision_weighting = use_precision_weighting
        self.precision_lr = precision_lr
        # Cholesky factor L per layer — P = LLᵀ always positive definite.
        # sigma_inv kept as alias to P for compatibility with step functions.
        self.sigma_inv: Dict[str, torch.Tensor] = {}   # P = LLT
        self._chol_L:   Dict[str, torch.Tensor] = {}   # lower triangular L

        self.optimizer = PCOptimizer(
            opt_name=optimizer_name,
            beta1=optimizer_beta1,
            beta2=optimizer_beta2,
            eps=optimizer_eps,
            momentum=optimizer_momentum,
            weight_decay=optimizer_weight_decay,
        )
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy_scalar = 0.0
        self._errors = []
    
    def register_lateral(self, layer_type: str, size: int):
        """Create and register lateral connections for layer_type."""
        if layer_type not in self.lateral_connections:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr, self.inference_lr)
            self.add_module(f"lateral_{layer_type}", self.lateral_connections[layer_type])


    def get_or_init_sigma_inv(self, layer_type: str, size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Return precision matrix P = LLᵀ for this layer type.
        L is initialised to identity so P = I at the start.
        """
        if not self.use_precision_weighting or layer_type == "linear_output":
            return None

        if layer_type not in self._chol_L:
            # Start from L = I  →  P = I·Iᵀ = I
            self._chol_L[layer_type]   = torch.eye(size, device=device)
            self.sigma_inv[layer_type] = torch.eye(size, device=device)

        return self.sigma_inv[layer_type]

    def update_sigma_inv(self, layer_type: str, mismatch: torch.Tensor) -> None:
        """
        Update precision P = LLᵀ via gradient descent on the Cholesky factor L.
        Sigma = P⁻¹ computed via two triangular solves — O(D²), always stable.
        Positive definiteness is guaranteed by construction: P = LLᵀ ≥ 0 always.
        """
        if not self.use_precision_weighting or layer_type == "linear_output":
            return
        if layer_type not in self._chol_L:
            return

        # Sample covariance S = eT e / N  shape (D, D)
        e = mismatch.detach().reshape(-1, mismatch.size(-1))
        S = (e.T @ e) / max(e.shape[0], 1)

        # Current Cholesky factor L and precision P = LLT
        L = self._chol_L[layer_type]
        P = self.sigma_inv[layer_type]

        # Compute Sigma = P^-1 via triangular solve — no linalg.inv
        # Solve L X = I  =>  X = L^-1  =>  Sigma = XT X = L^-T L^-1
        eye = torch.eye(L.size(-1), device=L.device, dtype=L.dtype)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
        Sigma = L_inv.T @ L_inv

        # Gradient of free energy w.r.t. P  (Bogacz 2017, Friston 2008)
        # dF/dP = 0.5 * S - 0.5 * Sigma
        grad_P = 0.5 * S - 0.5 * Sigma

        # Chain rule through P = LLT  =>  dF/dL = 2 * grad_P @ L
        grad_L = 2.0 * grad_P @ L
        grad_L = torch.clip(grad_L, -1.0, 1.0)

        # Update L and enforce lower triangular structure
        new_L = L - self.precision_lr * grad_L
        new_L = torch.tril(new_L)

        # Recompute P = LLT — always positive semi-definite by construction
        new_P = new_L @ new_L.T

        self._chol_L[layer_type]   = new_L
        self.sigma_inv[layer_type] = new_P
        

    def _reset_step_state(self) -> None:
        """Reset step-local accumulators, kept for future extension."""
        return
    
    def _get_cached_state(self, layer_type: str):
        return self._x_cache.get(layer_type, None)
    
    def forward(
        self,
        target_activity: torch.Tensor,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool,
        td_err:  Optional[torch.Tensor] = None,
        layer: Optional[nn.Module] = None,
        layer_norm: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        flash: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ADD THIS
        use_cache: bool = False, 
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        x = self._get_cached_state(layer_type)

        if rope_cache is not None:
            self.rope_cache = rope_cache


        if layer_type == "embed":
            sigma_inv = self.get_or_init_sigma_inv(layer_type, target_activity.size(-1), target_activity.device)

            mu, mu_word, bu_err = step_embed(
                t,
                T,
                target_activity,
                layer,
                layer_type,
                input_ids,
                self.local_lr,
                self.clamp_value,
                self.clip_value,
                self.energy_fn_name,
                requires_update,
                layer_norm=layer_norm,
                optimizer=self.optimizer,
                sigma_inv=sigma_inv,
            )            
            # store for later retrieval
            self._x_cache["embed"] = (mu_word)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            # compute energy
            error = target_activity - mu
                
            energy, step_errors = finalize_step(
                mu, target_activity, error, t, layer_type,
                self.energy_fn_name, self.output_energy_fn_name,
                sigma_inv=sigma_inv,
            )
            self._energy_scalar = energy  # overwrite each step; only the last T-step survives
            self._errors.extend(step_errors)
            
            if requires_update and sigma_inv is not None:
                self.update_sigma_inv(layer_type, error)
            return mu_word
        
        elif layer_type == "attn":
            sigma_inv = self.get_or_init_sigma_inv(layer_type, target_activity.size(-1), target_activity.device)
            
            lateral_conn = None
            x, mu, bu_err, new_kv_cache = step_attn(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.inference_lr,
                self.clamp_value,
                self.clip_value,
                self.energy_fn_name,
                requires_update,
                self.num_heads,
                self.n_embed,
                td_err=td_err, 
                layer_norm=layer_norm,
                rope_cache=self.rope_cache, 
                flash=flash, 
                kv_cache=kv_cache,  
                use_cache=use_cache,
                optimizer=self.optimizer,
                sigma_inv=sigma_inv,
            )
            error = target_activity - mu
            if requires_update and sigma_inv is not None:
                self.update_sigma_inv(layer_type, error)
            # Store cache for retrieval
            if use_cache:
                self._last_kv_cache = new_kv_cache
        
        else:
            sigma_inv = self.get_or_init_sigma_inv(layer_type, target_activity.size(-1), target_activity.device) if target_activity is not None else None

            lateral_conn = None
            x, mu, bu_err = step_linear(
                t,
                T,
                target_activity,
                x,
                layer, 
                lateral_conn,  
                layer_type,
                self.local_lr, 
                self.inference_lr,
                self.clamp_value, 
                self.clip_value,
                self.energy_fn_name, 
                requires_update,
                td_err=td_err, 
                layer_norm=layer_norm,
                optimizer=self.optimizer,
                sigma_inv=sigma_inv,
            )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None: 
         self._error_cache[layer_type] = bu_err.detach().clone()   
        

        # target_activity is None when the output layer is unclamped during
        # generation; there is no prediction error / energy to record then.
        if target_activity is not None:
            error = target_activity - mu
    
            energy, step_errors = finalize_step(
                mu, target_activity, error, t, layer_type,
                self.energy_fn_name, self.output_energy_fn_name,
                sigma_inv=sigma_inv,
            )
            self._energy_scalar = energy  # overwrite each step; only the last T-step survives
            self._errors.extend(step_errors)
            
            if requires_update and sigma_inv is not None:
                self.update_sigma_inv(layer_type, error)
                
        # update x cache
        self._x_cache[layer_type] = x
        return x, mu

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word) from embedding weights
        - attn: creates random initialization shaped (B, S, H_out)
        - linear/others: random init sized to layer input dimension
        """
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            
            x_word = layer["word"].weight[input_ids] 
            self._x_cache["embed"] = (x_word)
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            self._x_cache["attn"] = x_init(batch_size, seq_len, H_out, device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            
            self.register_lateral(layer_type, input_dim)  
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached activity tensor for a given layer type."""
        return self._x_cache.get(layer_type, None)
    
    def get_sigma_inv(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached inverse precision tensor for a given layer type."""
        return self.sigma_inv.get(layer_type, None)
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached mu (prediction) tensor for a given layer type."""
        return self._mu_cache.get(layer_type, None)
    
    def get_td_err(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached top-down error tensor for a given layer type."""
        return self._error_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        """Get the energy from the final inference step (T) for the layer."""
        return float(self._energy_scalar)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy_scalar = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        
    def get_errors(self) -> list:
        """Get the errors from the most recent converged iteration."""
        # Each step can contribute multiple error entries, so we find the last unique 'step'
        if not self._errors:
            return []
        last_step_val = self._errors[-1]["step"]
        return [err for err in self._errors if err["step"] == last_step_val]

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)
    def set_inference_learning_rate(self, inference_lr: float):
        """Set the inference learning rate for the layer."""
        self.inference_lr = float(inference_lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)
    
    def get_inference_learning_rate(self) -> float:
        """Get the current inference learning rate for the layer."""
        return float(self.inference_lr)
