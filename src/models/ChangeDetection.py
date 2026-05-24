"""Binary change-detection head on top of a domain-adapter ResNet backbone.

Decoder design (sequential-mode-safe):
    Conv (shared)  ->  per-domain BN  ->  ReLU  ->  per-domain adapter
    ...
    per-domain classifier  ->  upsample

Only the convolution kernels themselves are shared across domains.  Every
distribution-sensitive component (BatchNorm statistics, residual adapter,
final classifier and its bias) is per-domain, which closes the leak points
that cause catastrophic forgetting when training a single domain for many
consecutive steps.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import ResidualAdapter


# ---------------------------------------------------------------------------
# Per-domain BatchNorm
# ---------------------------------------------------------------------------

class DomainBN(nn.Module):
    """One BatchNorm2d per domain, selected by string key at forward time.

    Running statistics, affine weight, and affine bias are all isolated per
    domain.  The shared convolution kernel sees gradients from both domains
    (mixed via training schedule), but normalisation is never contaminated
    across domains.
    """

    def __init__(self, num_features: int, domains: Iterable[str]):
        super().__init__()
        self.bns = nn.ModuleDict({d: nn.BatchNorm2d(num_features) for d in domains})

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.bns[domain](x)


class DomainConvBNReLU(nn.Module):
    """Shared Conv2d -> per-domain BN -> ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int,
        domains: Iterable[str],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = DomainBN(out_ch, domains)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.act(self.bn(self.conv(x), domain))


# ---------------------------------------------------------------------------
# Change-detection head
# ---------------------------------------------------------------------------

class ChangeDetectionModel(nn.Module):
    """Backbone -> |f1 - f2| -> domain-aware decoder."""

    def __init__(
        self,
        backbone,
        domain_list: Iterable[str] | None = None,
        in_channels: int = 2048,
        prior: float = 0.02,
    ):
        super().__init__()
        self.backbone = backbone

        if domain_list is None:
            domain_list = getattr(backbone, "domain_list", None)
        if domain_list is None:
            raise ValueError(
                "ChangeDetectionModel needs a domain_list (or a backbone with one)."
            )
        self.domain_list: List[str] = list(domain_list)

        # ------------------------------------------------------------------
        # Decoder: shared conv kernels + per-domain BN
        # ------------------------------------------------------------------
        self.reduce = DomainConvBNReLU(in_channels, 256, kernel_size=1, padding=0, domains=self.domain_list)
        self.conv1  = DomainConvBNReLU(256, 128, kernel_size=3, padding=1, domains=self.domain_list)
        self.conv2  = DomainConvBNReLU(128, 64,  kernel_size=3, padding=1, domains=self.domain_list)

        # Per-domain residual adapters between decoder stages (identity-init).
        self.decoder_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                "reduce": ResidualAdapter(256, reduction=8),
                "conv1":  ResidualAdapter(128, reduction=8),
                "conv2":  ResidualAdapter(64,  reduction=8),
            })
            for d in self.domain_list
        })

        # Per-domain final classifier with prior-bias init.  The bias sets
        # initial sigmoid output to ~``prior`` so the model does not collapse
        # to "predict background" under severe class imbalance.
        self.classifier = nn.ModuleDict({
            d: nn.Conv2d(64, 1, kernel_size=1) for d in self.domain_list
        })
        prior_bias = math.log(prior / (1.0 - prior))
        for clf in self.classifier.values():
            nn.init.normal_(clf.weight, std=0.01)
            nn.init.constant_(clf.bias, prior_bias)

    # ----------------------------------------------------------------------
    def _decode(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        ad = self.decoder_adapters[domain]
        x = ad["reduce"](self.reduce(x, domain))
        x = ad["conv1"](self.conv1(x, domain))
        x = ad["conv2"](self.conv2(x, domain))
        return self.classifier[domain](x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in self.classifier:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.classifier)}"
            )

        f1 = self.backbone(img1, domain)
        f2 = self.backbone(img2, domain)
        diff = torch.abs(f1 - f2)

        logits = self._decode(diff, domain)
        logits = F.interpolate(
            logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits
