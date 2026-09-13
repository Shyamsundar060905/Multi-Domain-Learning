"""Change-guided dynamic adapters for bi-temporal change detection.

Given the two temporal feature maps x1, x2 [B, C, H, W] at the same point in
the Siamese encoder, the adapter builds a temporal-difference representation
and uses it to predict:

    1) sample-wise mixture weights over lightweight residual adapter experts;
    2) a spatial gate indicating where adaptation should be applied;
    3) FiLM-style channel scale and bias.

Both streams are adapted with the same routing, gate and FiLM parameters, so
the adaptation applied to each timestep is conditioned on how the two differ.

Differences from the reference implementation this was adapted from:
- FiLM starts as the identity (gamma = 1, beta = 0).  With PyTorch's default
  Linear init, ``beta`` is non-zero at step 0, so the adapter would shift the
  frozen backbone features before any training -- and every later frozen block
  would then see inputs it was never trained on.  With identity FiLM and
  zero-initialised experts the adapter is an exact identity at initialisation,
  the same guarantee ResidualAdapter gives.
- GroupNorm group counts always divide the channel count.
- The demonstration network is omitted; only the adapter and its balance loss
  are needed here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest group count <= max_groups that divides ``channels``."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualAdapterExpert(nn.Module):
    """A parameter-efficient residual convolutional expert (zero-initialised output)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class ChangeGuidedDynamicAdapter(nn.Module):
    """Difference-conditioned mixture of residual adapter experts.

    Args:
        channels: Number of channels in x1 and x2.
        num_experts: Number of adapter experts.
        reduction: Bottleneck reduction ratio for the change encoder and experts.
        temperature: Softmax temperature for expert routing.
        top_k: If given, keep only the k highest routing weights per sample
            (renormalised).  ``None`` uses every expert.
        use_signed_difference: Include x1 - x2 alongside |x1 - x2| and x1 * x2.

    Returns:
        adapted_1, adapted_2: adapted feature maps, same shape as the inputs.
        info: routing weights / logits and the spatial change gate.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        reduction: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_signed_difference: bool = True,
    ) -> None:
        super().__init__()
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts].")

        self.channels = channels
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k
        self.use_signed_difference = use_signed_difference

        diff_channels = channels * (3 if use_signed_difference else 2)
        hidden = max(channels // reduction, 16)

        self.change_encoder = nn.Sequential(
            ConvNormAct(diff_channels, hidden, kernel_size=1),
            ConvNormAct(hidden, hidden, kernel_size=3),
        )

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden, 2 * channels),
        )
        # Identity FiLM at init: gamma = 1, beta = 0 (see module docstring).
        film_out = self.film[-1]
        nn.init.zeros_(film_out.weight)
        with torch.no_grad():
            film_out.bias.zero_()
            film_out.bias[:channels].fill_(1.0)

        self.experts = nn.ModuleList(
            [ResidualAdapterExpert(channels, reduction=reduction) for _ in range(num_experts)]
        )

        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def _difference_features(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(x1 - x2)
        product = x1 * x2
        if self.use_signed_difference:
            return torch.cat([abs_diff, x1 - x2, product], dim=1)
        return torch.cat([abs_diff, product], dim=1)

    def _sparsify_router(self, weights: torch.Tensor) -> torch.Tensor:
        if self.top_k is None or self.top_k == self.num_experts:
            return weights
        values, indices = torch.topk(weights, k=self.top_k, dim=1)
        sparse = torch.zeros_like(weights).scatter_(1, indices, values)
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes; got {x1.shape} and {x2.shape}."
            )

        change_context = self.change_encoder(self._difference_features(x1, x2))

        router_logits = self.router(change_context)
        routing_weights = F.softmax(router_logits / self.temperature, dim=1)
        routing_weights = self._sparsify_router(routing_weights)

        gate = self.spatial_gate(change_context)
        gamma, beta = self.film(change_context).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        expert_residual_1 = torch.stack([expert(x1) for expert in self.experts], dim=1)
        expert_residual_2 = torch.stack([expert(x2) for expert in self.experts], dim=1)
        weights = routing_weights[:, :, None, None, None]

        residual_1 = (expert_residual_1 * weights).sum(dim=1)
        residual_2 = (expert_residual_2 * weights).sum(dim=1)

        residual_1 = gate * (gamma * residual_1 + beta)
        residual_2 = gate * (gamma * residual_2 + beta)

        adapted_1 = x1 + self.residual_scale * residual_1
        adapted_2 = x2 + self.residual_scale * residual_2

        info = {
            "routing_weights": routing_weights,
            "routing_logits": router_logits,
            "change_gate": gate,
        }
        return adapted_1, adapted_2, info


def routing_balance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    """Encourages all experts to receive traffic across a mini-batch."""
    mean_prob = routing_weights.mean(dim=0)
    uniform = torch.full_like(mean_prob, 1.0 / mean_prob.numel())
    return F.kl_div(mean_prob.clamp_min(1e-8).log(), uniform, reduction="batchmean")
