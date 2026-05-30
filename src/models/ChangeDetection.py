"""U-Net change-detection head: frozen shared decoder blocks + per-domain adapters."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import ResidualAdapter, STAGE_CHANNELS


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


class FrozenConvBlock(nn.Module):
    """Conv-BN-ReLU block with frozen weights (shared decoder trunk)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


class UNetUpStage(nn.Module):
    """One U-Net decoder step: upsample → concat skip → frozen conv → domain adapter."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        domain_list: Iterable[str],
        adapter_reduction: int = 8,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.skip_ch = skip_ch
        merge_in = in_ch + skip_ch if skip_ch > 0 else in_ch
        self.merge_conv = FrozenConvBlock(merge_in, out_ch, kernel_size=3, padding=1)

        self.domain_adapters = nn.ModuleDict({
            d: ResidualAdapter(out_ch, reduction=adapter_reduction, dropout=adapter_dropout)
            for d in domain_list
        })

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor],
        domain: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if target_size is None:
            if skip is not None:
                target_size = skip.shape[-2:]
            else:
                raise ValueError("target_size required when skip is None")

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.merge_conv(x)
        return self.domain_adapters[domain](x)


class UNetDecoderWithAdapters(nn.Module):
    """Symmetric U-Net decoder: frozen merge convs + per-domain residual adapters."""

    def __init__(
        self,
        domain_list: Iterable[str],
        prior: float = 0.02,
        adapter_reduction: int = 8,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.domain_list = list(domain_list)

        # Bottleneck on fused l4 (6144 → 1024).
        self.bottleneck = FrozenConvBlock(FUSED_CHANNELS["l4"], 512, kernel_size=1, padding=0)

        # Upsampling stages: (in_ch, skip_ch, out_ch).
        self.up_stages = nn.ModuleList([
            UNetUpStage(512, FUSED_CHANNELS["l3"], 256, domain_list, adapter_reduction, adapter_dropout),
            UNetUpStage(256, FUSED_CHANNELS["l2"], 128, domain_list, adapter_reduction, adapter_dropout),
            UNetUpStage(128, FUSED_CHANNELS["l1"], 64, domain_list, adapter_reduction, adapter_dropout),
            UNetUpStage(64, 0, 32, domain_list, adapter_reduction, adapter_dropout),
        ])

        self.classifiers = nn.ModuleDict({
            d: nn.Conv2d(32, 1, kernel_size=1) for d in self.domain_list
        })

        # Deep-supervision head on first decoder scale (≈ layer3 resolution).
        self.aux_classifiers = nn.ModuleDict({
            d: nn.Conv2d(256, 1, kernel_size=1) for d in self.domain_list
        })

        prior_bias = math.log(prior / (1.0 - prior))
        for head in list(self.classifiers.values()) + list(self.aux_classifiers.values()):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, prior_bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bottleneck.train(False)
        for stage in self.up_stages:
            stage.merge_conv.train(False)
        return self

    def forward(
        self,
        fused_skips: Dict[str, torch.Tensor],
        domain: str,
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bottleneck(fused_skips["l4"])

        x = self.up_stages[0](x, fused_skips["l3"], domain)
        aux = self.aux_classifiers[domain](x)

        x = self.up_stages[1](x, fused_skips["l2"], domain)
        x = self.up_stages[2](x, fused_skips["l1"], domain)
        x = self.up_stages[3](x, skip=None, domain=domain, target_size=out_size)

        logits = self.classifiers[domain](x)
        return logits, aux

    def domain_parameters(self, domain: str) -> list:
        params: list = []
        for stage in self.up_stages:
            params += list(stage.domain_adapters[domain].parameters())
        params += list(self.classifiers[domain].parameters())
        params += list(self.aux_classifiers[domain].parameters())
        return params


class ChangeDetectionModel(nn.Module):
    """Bi-temporal U-Net CD: adapter encoder pyramid + adapter U-Net decoder."""

    def __init__(
        self,
        backbone,
        domain_list: Iterable[str] | None = None,
        prior: float = 0.02,
        use_deep_supervision: bool = True,
        adapter_reduction: int = 8,
        adapter_dropout: float = 0.1,
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

        self.decoder = UNetDecoderWithAdapters(
            self.domain_list,
            prior=prior,
            adapter_reduction=adapter_reduction,
            adapter_dropout=adapter_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.train(mode)
        return self

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

        logits, aux = self.decoder(fused, domain, out_size=img1.shape[-2:])

        if not (self.use_deep_supervision and self.training):
            aux = None

        return logits, aux

    def domain_parameters(self, domain: str) -> list:
        params = self.decoder.domain_parameters(domain)
        backbone_adapters = getattr(self.backbone, "domain_adapters", None)
        if backbone_adapters is not None and domain in backbone_adapters:
            params += list(backbone_adapters[domain].parameters())
        return params
