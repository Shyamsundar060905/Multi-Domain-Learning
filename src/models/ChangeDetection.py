"""Per-domain change-detection model.

Notebook-style design (mirrors the classification setup in
``notebooks/MultiDomain.ipynb``):

- Backbone is shared and **frozen**; the only backbone-side trainable
  parameters are the per-domain residual adapters in layer3/layer4 (added by
  ``ResNetWithAdapters``).
- The entire decoder is **per-domain**: each domain owns its own
  ``reduce -> conv1 -> conv2 -> classifier`` stack.  Nothing in the decoder
  is shared across domains, so joint training cannot induce inter-domain
  interference at the head.
- Each per-domain classifier is prior-bias-initialised so the model starts
  near ``sigmoid(b) = prior`` and avoids the background-collapse failure
  mode on imbalanced change-detection data.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDDecoder(nn.Module):
    """A single domain's binary-CD decoder."""

    def __init__(self, in_channels: int = 2048, prior: float = 0.02):
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

        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)


class ChangeDetectionModel(nn.Module):
    """Backbone -> |f1 - f2| -> per-domain decoder -> upsample."""

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

        self.decoders = nn.ModuleDict({
            d: CDDecoder(in_channels=in_channels, prior=prior)
            for d in self.domain_list
        })

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in self.decoders:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.decoders)}"
            )

        f1 = self.backbone(img1, domain)
        f2 = self.backbone(img2, domain)
        diff = torch.abs(f1 - f2)

        logits = self.decoders[domain](diff)
        logits = F.interpolate(
            logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits

    # ----------------------------------------------------------------------
    def domain_parameters(self, domain: str):
        """All trainable parameters owned by ``domain``: backbone adapters +
        full per-domain decoder + classifier.  Pass this list directly to an
        ``Adam`` / ``AdamW`` optimiser for a notebook-style per-domain training
        loop.
        """
        params = list(self.decoders[domain].parameters())
        backbone_adapters = getattr(self.backbone, "domain_adapters", None)
        if backbone_adapters is not None and domain in backbone_adapters:
            params += list(backbone_adapters[domain].parameters())
        return params
