"""U-Net change detection: adapter encoder pyramid + shared trainable decoder."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import STAGE_CHANNELS


def build_bitemporal_fusion(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Concatenate both time steps and their absolute difference (3× channels)."""
    return torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)


# Fused skip channels = 3 × native stage width (keys match ``extract_multiscale``).
FUSED_CHANNELS = {
    "l1": 3 * STAGE_CHANNELS["layer1"],
    "l2": 3 * STAGE_CHANNELS["layer2"],
    "l3": 3 * STAGE_CHANNELS["layer3"],
    "l4": 3 * STAGE_CHANNELS["layer4"],
}


class ConvBlock(nn.Module):
    """Trainable Conv-BN-ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpStage(nn.Module):
    """One U-Net decoder step: ConvTranspose2d upsample → concat skip → merge conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, upsample_stride: int = 2):
        super().__init__()
        self.has_skip = skip_ch > 0
        self.upconv = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=upsample_stride, stride=upsample_stride
        )
        merge_in = in_ch + skip_ch if self.has_skip else in_ch
        self.merge_conv = ConvBlock(merge_in, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upconv(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Upsampled shape {x.shape[-2:]} != skip shape {skip.shape[-2:]}. "
                    "Use an input size divisible by 32."
                )
            x = torch.cat([x, skip], dim=1)
        return self.merge_conv(x)


class UNetDecoder(nn.Module):
    """Shared U-Net decoder (trainable) used by all domains."""

    def __init__(self, prior: float = 0.02):
        super().__init__()
        self.bottleneck = ConvBlock(FUSED_CHANNELS["l4"], 512, kernel_size=1, padding=0)

        # l4→l3, l3→l2, l2→l1 are 2×; l1→full resolution is 4× (H/4 → H).
        self.up_stages = nn.ModuleList([
            UNetUpStage(512, FUSED_CHANNELS["l3"], 256, upsample_stride=2),
            UNetUpStage(256, FUSED_CHANNELS["l2"], 128, upsample_stride=2),
            UNetUpStage(128, FUSED_CHANNELS["l1"], 64, upsample_stride=2),
            UNetUpStage(64, 0, 32, upsample_stride=4),
        ])

        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        self.aux_classifier = nn.Conv2d(256, 1, kernel_size=1)

        prior_bias = math.log(prior / (1.0 - prior))
        for head in (self.classifier, self.aux_classifier):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, prior_bias)

    def forward(
        self,
        fused_skips: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bottleneck(fused_skips["l4"])

        x = self.up_stages[0](x, fused_skips["l3"])
        aux = self.aux_classifier(x)

        x = self.up_stages[1](x, fused_skips["l2"])
        x = self.up_stages[2](x, fused_skips["l1"])
        x = self.up_stages[3](x)

        logits = self.classifier(x)
        return logits, aux


class ChangeDetectionModel(nn.Module):
    """Bi-temporal U-Net CD: adapter encoder pyramid + shared U-Net decoder."""

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

        self.decoder = UNetDecoder(prior=prior)

    def _fuse_pyramid(
        self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {k: build_bitemporal_fusion(p1[k], p2[k]) for k in p1}

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor, domain: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pyramid1 = self.backbone.extract_multiscale(img1, domain)
        pyramid2 = self.backbone.extract_multiscale(img2, domain)
        fused = self._fuse_pyramid(pyramid1, pyramid2)

        logits, aux = self.decoder(fused)

        # Aux head is at H/16; upsample mask path for deep-sup loss only.
        if self.use_deep_supervision and self.training and aux is not None:
            aux = F.interpolate(
                aux, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            aux = None

        return logits, aux

    def domain_parameters(self, domain: str) -> list:
        """Per-domain encoder adapters only."""
        backbone_adapters = getattr(self.backbone, "domain_adapters", None)
        if backbone_adapters is not None and domain in backbone_adapters:
            return list(backbone_adapters[domain].parameters())
        return []

    def shared_parameters(self) -> list:
        """Shared trainable decoder weights (common across domains)."""
        return list(self.decoder.parameters())
