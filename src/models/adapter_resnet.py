"""U-Net encoder: frozen down-path + per-domain residual adapters at each scale."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """1x1 bottleneck adapter with identity init and optional dropout."""

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(channels // reduction, 8)

        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.bn(self.down(x)))))


# Pyramid channel widths (must match decoder ``FUSED_CHANNELS``).
STAGE_CHANNELS = {
    "l1": 64,
    "l2": 128,
    "l3": 256,
    "l4": 512,
}


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block (optionally frozen via ``freeze()``)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class UNetDownStage(nn.Module):
    """Pool -> frozen double conv -> domain adapter (one pyramid level)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        domain_list: Iterable[str],
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )
        self.domain_adapters = nn.ModuleDict({
            d: ResidualAdapter(out_ch, reduction=adapter_reduction, dropout=adapter_dropout)
            for d in domain_list
        })

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return self.domain_adapters[domain](x)


class UNetEncoderWithAdapters(nn.Module):
    """Frozen U-Net encoder pyramid + trainable per-domain adapters (symmetric to decoder)."""

    def __init__(
        self,
        domain_list: Iterable[str],
        adapter_dropout: float = 0.1,
        adapter_reduction: int = 16,
    ):
        super().__init__()
        self.domain_list = list(domain_list)
        self.feature_channels = STAGE_CHANNELS["l4"]

        # 512 -> 128 (H/4), 64 channels — first pyramid level (l1).
        self.stem = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        self.stem_pool = nn.MaxPool2d(2)
        self.stem_adapters = nn.ModuleDict({
            d: ResidualAdapter(64, reduction=adapter_reduction, dropout=adapter_dropout)
            for d in self.domain_list
        })

        self.down_stages = nn.ModuleList([
            UNetDownStage(64, STAGE_CHANNELS["l2"], domain_list,
                          adapter_reduction, adapter_dropout),
            UNetDownStage(STAGE_CHANNELS["l2"], STAGE_CHANNELS["l3"], domain_list,
                          adapter_reduction, adapter_dropout),
            UNetDownStage(STAGE_CHANNELS["l3"], STAGE_CHANNELS["l4"], domain_list,
                          adapter_reduction, adapter_dropout),
        ])

        self._freeze_trunk()

    def _freeze_trunk(self) -> None:
        for block in self.stem:
            block.freeze()
        for stage in self.down_stages:
            stage.pool.eval()
            for block in stage.conv:
                block.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        for block in self.stem:
            block.eval()
        for stage in self.down_stages:
            stage.pool.eval()
            for block in stage.conv:
                block.eval()
        return self

    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        params = list(self.stem_adapters[domain].parameters())
        for stage in self.down_stages:
            params += list(stage.domain_adapters[domain].parameters())
        return params

    def adapter_parameters(self, domain: str | None = None):
        if domain is None:
            for d in self.domain_list:
                yield from self.domain_parameters(d)
        else:
            yield from self.domain_parameters(domain)

    def extract_multiscale(self, x: torch.Tensor, domain: str) -> Dict[str, torch.Tensor]:
        """Return encoder pyramid ``{l1, l2, l3, l4}`` with domain adapters applied."""
        if domain not in self.stem_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.stem_adapters)}"
            )

        x = self.stem(x)
        x = self.stem_pool(x)
        x = self.stem_pool(x)
        l1 = self.stem_adapters[domain](x)

        l2 = self.down_stages[0](l1, domain)
        l3 = self.down_stages[1](l2, domain)
        l4 = self.down_stages[2](l3, domain)
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def extract_features(self, x: torch.Tensor, domain: str):
        feats = self.extract_multiscale(x, domain)
        return feats["l3"], feats["l4"]

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.extract_multiscale(x, domain)["l4"]


# Backward-compatible alias (same interface as the old ResNet encoder wrapper).
ResNetWithAdapters = UNetEncoderWithAdapters
