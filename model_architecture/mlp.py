import torch.nn as nn
from predictive_coding.pc_layer import PCLayer
from utils.model_utils import init_weights

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used within the transformer architecture.
    Includes two linear layers and two predictive coding layers for local learning.
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        for layer in (self.fc1, self.fc2):
            init_weights(layer, config.weight_init_type)

        # Hidden predictive coding layers -> use hidden_lr / hidden_inference_lr
        self.pc_layer2 = PCLayer(
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

        self.pc_layer1 = PCLayer(
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