import torch.nn as nn
from predictive_coding.pc_layer import PCLayer
from utils.model_utils import init_weights

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)

        init_weights(self.output, config.weight_init_type)

        # Output predictive coding layer -> use output_lr / output_inference_lr
        self.pc_layer = PCLayer(
            T=config.T,
            lr=config.output_lr,
            inference_lr=config.output_inference_lr,
            energy_fn_name=config.output_energy_fn_name,
            optimizer_name=config.output_optimizer_name,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_eps=config.optimizer_eps,
            optimizer_momentum=config.optimizer_momentum,
            optimizer_weight_decay=config.optimizer_weight_decay,
        )