"""Per-domain change-detection model with concat fusion + deep supervision."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_bitemporal_fusion(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Concatenate both time steps and their absolute difference."""
    return torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)


class CDDecoder(nn.Module):
    """Main decoder on layer4 fusion (3 * in_channels)."""

    def __init__(self, in_channels: int, prior: float = 0.02):
        super().__init__()
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
        self._init_classifier(prior)

    def _init_classifier(self, prior: float) -> None:
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)


class CDAuxDecoder(nn.Module):
    """Lightweight auxiliary head on layer3 fusion for small-object supervision."""

    def __init__(self, in_channels: int, prior: float = 0.02):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(128, 1, kernel_size=1)
        self._init_classifier(prior)

    def _init_classifier(self, prior: float) -> None:
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.reduce(x))


class ChangeDetectionModel(nn.Module):
    """Backbone -> concat(f1,f2,|f1-f2|) -> per-domain decoder (+ aux on layer3)."""

    LAYER3_CHANNELS = 1024
    LAYER4_CHANNELS = 2048

    def __init__(
        self,
        backbone,
        domain_list: Iterable[str] | None = None,
        prior: float = 0.02,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_deep_supervision = use_deep_supervision

        if domain_list is None:
            domain_list = getattr(backbone, "domain_list", None)
        if domain_list is None:
            raise ValueError(
                "ChangeDetectionModel needs a domain_list (or a backbone with one)."
            )
        self.domain_list: List[str] = list(domain_list)

        fusion_l4 = 3 * self.LAYER4_CHANNELS
        fusion_l3 = 3 * self.LAYER3_CHANNELS

        self.decoders = nn.ModuleDict({
            d: CDDecoder(in_channels=fusion_l4, prior=prior)
            for d in self.domain_list
        })
        self.aux_decoders = nn.ModuleDict({
            d: CDAuxDecoder(in_channels=fusion_l3, prior=prior)
            for d in self.domain_list
        })

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor, domain: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if domain not in self.decoders:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.decoders)}"
            )

        f1_l3, f1_l4 = self.backbone.extract_features(img1, domain)
        f2_l3, f2_l4 = self.backbone.extract_features(img2, domain)

        fused_l4 = build_bitemporal_fusion(f1_l4, f2_l4)
        fused_l3 = build_bitemporal_fusion(f1_l3, f2_l3)

        logits = self.decoders[domain](fused_l4)
        logits = F.interpolate(
            logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
        )

        aux_logits = None
        if self.use_deep_supervision and self.training:
            aux_logits = self.aux_decoders[domain](fused_l3)
            aux_logits = F.interpolate(
                aux_logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )

        return logits, aux_logits

    def domain_parameters(self, domain: str):
        params = list(self.decoders[domain].parameters())
        params += list(self.aux_decoders[domain].parameters())
        backbone_adapters = getattr(self.backbone, "domain_adapters", None)
        if backbone_adapters is not None and domain in backbone_adapters:
            params += list(backbone_adapters[domain].parameters())
        return params
