"""Binary change-detection head on top of a domain-adapter ResNet backbone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeDetectionModel(nn.Module):
    """Compute |f(img1) - f(img2)| -> 1x1 reduce -> 3x3 conv head -> upsample."""

    def __init__(self, backbone, in_channels: int = 2048, prior: float = 0.02):
        super().__init__()
        self.backbone = backbone

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # Prior-bias initialisation (focal-loss style).  Setting b = log(p/(1-p))
        # makes the model's initial output sigmoid(b) == prior, which prevents
        # the gradient from collapsing the head straight to "predict background"
        # during the first few iterations of severely-imbalanced training.
        final_conv = self.head[-1]
        nn.init.normal_(final_conv.weight, std=0.01)
        nn.init.constant_(final_conv.bias, math.log(prior / (1.0 - prior)))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, domain: str) -> torch.Tensor:
        f1 = self.backbone(img1, domain)
        f2 = self.backbone(img2, domain)

        diff = torch.abs(f1 - f2)
        x = self.reduce(diff)
        x = self.head(x)
        x = F.interpolate(x, size=img1.shape[-2:], mode="bilinear", align_corners=False)
        return x
