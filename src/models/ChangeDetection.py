"""Binary change-detection head on top of a domain-adapter ResNet backbone.

The decoder mirrors the encoder design: shared convolutional weights with a
per-domain ``ResidualAdapter`` inserted between every decoder stage.  This
keeps the trainable surface small while giving each domain its own tunable
modulation of the segmentation pathway.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import ResidualAdapter


class ChangeDetectionModel(nn.Module):
    """Backbone -> |f1 - f2| -> shared decoder + per-domain decoder adapters."""

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
        # Shared decoder weights
        # ------------------------------------------------------------------
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(64, 1, kernel_size=1)

        # ------------------------------------------------------------------
        # Per-domain decoder adapters (zero-initialised -> identity start)
        # ------------------------------------------------------------------
        self.decoder_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                "reduce": ResidualAdapter(256, reduction=8),
                "conv1":  ResidualAdapter(128, reduction=8),
                "conv2":  ResidualAdapter(64,  reduction=8),
            })
            for d in self.domain_list
        })

        # Prior-bias initialisation on the final conv -- prevents the
        # background-collapse failure mode on imbalanced CD data.
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(prior / (1.0 - prior)))

    # ----------------------------------------------------------------------
    def _decode(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        ad = self.decoder_adapters[domain]
        x = ad["reduce"](self.reduce(x))
        x = ad["conv1"](self.conv1(x))
        x = ad["conv2"](self.conv2(x))
        x = self.classifier(x)
        return x

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in self.decoder_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.decoder_adapters)}"
            )

        f1 = self.backbone(img1, domain)
        f2 = self.backbone(img2, domain)
        diff = torch.abs(f1 - f2)

        logits = self._decode(diff, domain)
        logits = F.interpolate(
            logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits
