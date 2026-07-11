import torch.nn as nn
from predictive_coding.pc_layer import PCLayer
from utils.model_utils import init_weights

class Attention(nn.Module):
    """
    Multi-head self-attention module with predictive coding layers for use in transformer architectures.
    Computes attention scores, applies masking, and outputs context vectors.
    Includes KV caching for efficient generation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.num_heads
        self.dropout = nn.Dropout(config.dropout)

        self.q = nn.Linear(config.n_embed, config.n_embed)
        self.k = nn.Linear(config.n_embed, config.n_embed)
        self.v = nn.Linear(config.n_embed, config.n_embed)
        self.output = nn.Linear(config.n_embed, config.n_embed)

        # Hidden predictive coding layer -> use hidden_lr / hidden_inference_lr
        for layer in (self.q, self.k, self.v, self.output):
            init_weights(layer, config.weight_init_type)

        self.pc_qkv = PCLayer(
            T=config.T,
            lr=config.hidden_lr,
            inference_lr=config.hidden_inference_lr,
            energy_fn_name=config.internal_energy_fn_name,
            num_heads=config.num_heads,
            n_embed=config.n_embed,
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_momentum=config.optimizer_momentum,
            optimizer_weight_decay=config.optimizer_weight_decay,
        )

        self.pc_output = PCLayer(
            T=config.T,
            lr=config.hidden_lr,
            inference_lr=config.hidden_inference_lr,
            energy_fn_name=config.internal_energy_fn_name,
            optimizer_name=config.optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_momentum=config.optimizer_momentum,
            optimizer_weight_decay=config.optimizer_weight_decay,
        )
        
        # KV cache for generation: stores (K, V) tensors
        self.kv_cache = None
        
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None