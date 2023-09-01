"""Additional classes to form a Mixture of Experts model on top of GPT."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nano_gpt.model import GPT, Block


class ExpertTransformerLayer(nn.Module):
    """A transformer block that acts as an individual expert in the MoE mechanism."""

    def __init__(self, config) -> None:
        super(ExpertTransformerLayer, self).__init__()
        # Initialize a transformer block using the provided configuration
        self.block = Block(config)

    def forward(self, x) -> torch.Tensor:
        """Compute the forward pass through the transformer block."""
        return self.block(x)


class GatingMechanism(nn.Module):
    """Determines the gating weights for each expert based on the input tensor."""

    def __init__(
        self, num_experts: int, input_dim: int, top_k: Optional[int] = None
    ) -> None:
        super(GatingMechanism, self).__init__()
        # Fully connected layer to determine logits for each expert
        self.fc = nn.Linear(input_dim, num_experts)
        # Number of experts to be considered
        self.num_experts = num_experts
        # If provided, only the top-k experts will be considered for gating
        self.top_k = top_k

    def forward(self, x) -> torch.Tensor:
        """Compute the gating weights for each expert."""

        # Get logits for each expert
        logits = self.fc(x)

        # Mask all but top-k logits if top_k is set
        if self.top_k and self.top_k < self.num_experts:
            _, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, topk_indices, 1)
            logits *= mask

        # Convert logits to softmax probabilities to get gating weights
        return F.softmax(logits, dim=-1)


class MixtureOfExpertsBlock(nn.Module):
    """Combines individual experts and a gating mechanism to produce a combined output."""

    def __init__(
        self, config, num_experts: int, top_k: Optional[int] = None
    ) -> None:
        super(MixtureOfExpertsBlock, self).__init__()
        # Initialize a list of experts
        self.experts = nn.ModuleList(
            [ExpertTransformerLayer(config) for _ in range(num_experts)]
        )
        # Gating mechanism to determine the contribution of each expert
        self.gate = GatingMechanism(num_experts, config.n_embd, top_k)

    def forward(self, x) -> torch.Tensor:
        """Calculate a weighted output based on expert responses."""

        # Determine weights for each expert
        weights = self.gate(x[:, -1, :])

        # If top_k is set, only compute the output from top-k experts
        if self.gate.top_k:
            _, topk_indices = torch.topk(weights, self.gate.top_k, dim=-1)
            topk_outputs = [
                self.experts[idx](x).unsqueeze(1) for idx in topk_indices[0]
            ]
            outputs = torch.cat(topk_outputs, dim=1)
            weights = torch.gather(weights, 1, topk_indices)
        else:
            # For all experts, compute outputs and concatenate
            outputs = torch.cat(
                [expert(x).unsqueeze(1) for expert in self.experts], dim=1
            )

        # Compute a weighted sum of expert outputs
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        N, C, H, W = outputs.shape
        expanded_weights = weights.expand(N, C, H, W)
        return (expanded_weights * outputs).sum(dim=1)


class MoEGPT(GPT):
    """A GPT model enhanced with a Mixture of Experts mechanism."""

    def __init__(
        self, config, num_experts: int, top_k: Optional[int] = None
    ) -> None:
        super(MoEGPT, self).__init__(config)

        # Variables for estimating MFU
        self.num_experts = num_experts
        self.top_k = top_k

        # Modify the transformer blocks in GPT to be MoE blocks
        self.transformer.h = nn.ModuleList(
            [
                MixtureOfExpertsBlock(config, num_experts, top_k)
                for _ in range(config.n_layer)
            ]
        )

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: int) -> float:
        """
        Override the MFU (Model FLOPs Utilization) estimation for MoEGPT to account for the MoE mechanism.
        This function estimates the model's flops utilization in units of A100 bfloat16 peak FLOPS.

        Args:
        - fwdbwd_per_iter (int): Number of forward and backward iterations.
        - dt (int): Time duration (typically in seconds) for which the operations are performed.

        Returns:
        - float: Estimated Model FLOPs Utilization.
        """

        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 312e12

        def estimate_flops_per_expert():
            """Estimate the number of FLOPS required for a single GPT expert."""
            # Calculate the number of parameters for a single expert
            N = self.get_num_params() // self.num_experts
            cfg = self.config
            L, H, Q, T = (
                cfg.n_layer,
                cfg.n_head,
                cfg.n_embd // cfg.n_head,
                cfg.block_size,
            )

            # Compute the FLOPS required per token
            flops_per_token = 6 * N + 12 * L * H * Q * T
            return flops_per_token * T * fwdbwd_per_iter

        # Determine the average number of experts used per layer
        M = self.top_k or self.num_experts

        # Calculate the FLOPS required for the gating mechanism
        L, T = self.config.n_layer, self.config.block_size
        G = 2 * M * self.config.n_embd
        gate_flops_per_token = 2 * L * T * G
        gate_flops_per_iter = gate_flops_per_token * T * fwdbwd_per_iter

        # Sum up the FLOPS required by the experts and the gating mechanism
        total_flops_achieved = (
            M * estimate_flops_per_expert() + gate_flops_per_iter
        ) / dt  # per second

        # Normalize the total FLOPS achieved by the peak FLOPS of the GPU
        return total_flops_achieved / flops_promised
