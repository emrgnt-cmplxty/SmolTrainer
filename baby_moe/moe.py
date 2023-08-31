"""Additional classes to form a Mixture of Experts model on top of GPT."""
from nano_gpt.model import GPT, Block
import torch
import torch.nn as nn
from torch.nn import functional as F


class ExpertTransformerLayer(nn.Module):
    """A single transformer block used as an expert in the Mixture of Experts mechanism."""

    def __init__(self, config):
        super(ExpertTransformerLayer, self).__init__()
        self.block = Block(config)

    def forward(self, x, targets=None):
        """Forward pass through the transformer block."""
        return self.block(x)


class GatingMechanism(nn.Module):
    """Mechanism to compute the weights for each expert based on the input tensor."""

    def __init__(self, num_experts, input_dim, k=None):
        super(GatingMechanism, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.k = k  # Number of experts to consider during gating

    def forward(self, x):
        logits = self.fc(x)
        if self.k and self.k < self.num_experts:
            _, topk_indices = torch.topk(logits, self.k, dim=-1)
            mask = logits.new_zeros(*logits.size())
            mask.scatter_(-1, topk_indices, 1)
            logits = logits * mask
        return F.softmax(logits, dim=-1)


class MixtureOfExpertsBlock(nn.Module):
    """A block combining multiple experts and a gating mechanism to produce a weighted output."""

    def __init__(self, config, num_experts, k=None):
        super(MixtureOfExpertsBlock, self).__init__()
        self.experts = nn.ModuleList(
            [ExpertTransformerLayer(config) for _ in range(num_experts)]
        )
        self.gate = GatingMechanism(num_experts, config.n_embd, k)

    def forward(self, x, targets=None):
        """Compute the weighted output of the mixture of experts."""

        weights = self.gate(x[:, -1, :])  # Consider the last token for gating
        outputs = [expert(x, targets) for expert in self.experts]

        # Reshape weights to be of shape [12, num_experts, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Expand weights to match the shape of the outputs for broadcasting
        weights = weights.expand(
            -1, -1, outputs[0].shape[1], outputs[0].shape[2]
        )

        return sum(weights[:, i] * o for i, o in enumerate(outputs))


class MoEGPT(GPT):
    """GPT model with a Mixture of Experts mechanism on top."""

    def __init__(self, config, num_experts, k=None):
        super(MoEGPT, self).__init__(config)
        self.transformer.h = nn.ModuleList(
            [
                MixtureOfExpertsBlock(config, num_experts, k)
                for _ in range(config.n_layer)
            ]
        )
