from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    """
    Configuration dataclass for the predictive coding transformer model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_embed (int): Embedding dimension size.
        dropout (float): Dropout probability.
        lr (float): [DEPRECATED] Local learning rate for predictive coding layers.
            Kept for backward compatibility. Used as the fallback value for
            hidden_lr/output_lr when those are not explicitly provided.
        inference_lr (float): [DEPRECATED] Inference learning rate. Kept for
            backward compatibility. Used as the fallback value for
            hidden_inference_lr/output_inference_lr when not explicitly provided.
        hidden_lr (Optional[float]): Local learning rate for hidden predictive
            coding layers (embed, attn, linear_attn, fc1, fc2). Falls back to
            `lr` if not set.
        output_lr (Optional[float]): Local learning rate for the output
            (linear_output) predictive coding layer. Falls back to `lr` if not set.
        hidden_inference_lr (Optional[float]): Inference learning rate for
            hidden predictive coding layers. Falls back to `inference_lr` if not set.
        output_inference_lr (Optional[float]): Inference learning rate for the
            output predictive coding layer. Falls back to `inference_lr` if not set.
        weight_init_type (str): Weight initialization strategy applied to
            trainable weight matrices ("default", "xavier", "kaiming", "normal").
        peak_learning_rate (float): Peak learning rate for learning rate scheduling.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        T (int): Number of inference steps for predictive coding.
        num_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        batch_size (int): Batch size for training/evaluation.
        num_epochs (int): Number of training epochs.
        energy_fn_name (str): Name of the energy function to use for error computation.
        use_flash_attention (bool): Whether to use FlashAttention.
        optimizer_name (str): Optimizer to use for PC weight updates (adam or sgd).
        optimizer_beta1 (float): Adam beta1 coefficient.
        optimizer_beta2 (float): Adam beta2 coefficient.
        optimizer_eps (float): Adam epsilon.
    """
    vocab_size: int
    block_size: int
    lr: float
    inference_lr: float
    peak_learning_rate: Optional[float]
    warmup_steps: Optional[int] 
    n_embed: int 
    dropout: float 
    T: int 
    num_heads: int 
    n_blocks: int 
    batch_size: int
    num_epochs: int
    internal_energy_fn_name:str
    output_energy_fn_name: str
    combined_internal_weight: float 
    combined_output_weight: float
    use_flash_attention: bool
    alpha: float
    clamp_value: float
    clip_value: float
    optimizer_name: str = "adam"
    output_optimizer_name: str = "adam"
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    optimizer_momentum: float = 0.9
    optimizer_weight_decay: float = 0.01

    # --- Hidden/output learning-rate split (new) ---
    hidden_lr: Optional[float] = None
    output_lr: Optional[float] = None
    hidden_inference_lr: Optional[float] = None
    output_inference_lr: Optional[float] = None

    # --- Configurable weight initialization (new) ---
    weight_init_type: str = "default"

    def __post_init__(self):
        # Fall back to the deprecated flat lr/inference_lr if hidden/output
        # variants were not explicitly provided. Keeps old configs/tuning
        # files working unchanged.
        if self.hidden_lr is None:
            self.hidden_lr = self.lr
        if self.output_lr is None:
            self.output_lr = self.lr
        if self.hidden_inference_lr is None:
            self.hidden_inference_lr = self.inference_lr
        if self.output_inference_lr is None:
            self.output_inference_lr = self.inference_lr

        valid_init_types = {"default", "xavier", "kaiming", "normal"}
        if self.weight_init_type not in valid_init_types:
            raise ValueError(
                f"Invalid weight_init_type: {self.weight_init_type!r}. "
                f"Choose from {sorted(valid_init_types)}."
            )