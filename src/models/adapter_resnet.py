"""U-Net style encoder: frozen ResNet50 + per-domain residual adapters at every stage."""

from __future__ import annotations

from typing import Dict

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


# Native channel widths at each ResNet stage (ResNet50).
STAGE_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


class ResNetWithAdapters(nn.Module):
    """Frozen ResNet50 encoder with per-domain adapters after every bottleneck.

    Produces a four-level feature pyramid (l1..l4) for U-Net skip connections.
    """

    def __init__(self, base, domain_list, adapter_dropout: float = 0.1, adapter_reduction: int = 16):
        super().__init__()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.feature_channels = STAGE_CHANNELS["layer4"]
        self.domain_list = list(domain_list)

        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                stage: nn.ModuleList([
                    ResidualAdapter(ch, reduction=adapter_reduction, dropout=adapter_dropout)
                    for _ in getattr(self, stage)
                ])
                for stage, ch in STAGE_CHANNELS.items()
            })
            for d in self.domain_list
        })

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for p in module.parameters():
                p.requires_grad = False
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _run_stage(
        self, stage: nn.Sequential, adapters: nn.ModuleList, x: torch.Tensor
    ) -> torch.Tensor:
        for block, adapter in zip(stage, adapters):
            x = block(x)
            x = adapter(x)
        return x

    def extract_multiscale(self, x: torch.Tensor, domain: str) -> Dict[str, torch.Tensor]:
        """Return encoder pyramid ``{l1, l2, l3, l4}`` with domain adapters applied."""
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        ad = self.domain_adapters[domain]
        x = self.stem(x)
        l1 = self._run_stage(self.layer1, ad["layer1"], x)
        l2 = self._run_stage(self.layer2, ad["layer2"], l1)
        l3 = self._run_stage(self.layer3, ad["layer3"], l2)
        l4 = self._run_stage(self.layer4, ad["layer4"], l3)
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def extract_features(self, x: torch.Tensor, domain: str):
        """Backward-compatible (layer3, layer4) tuple."""
        feats = self.extract_multiscale(x, domain)
        return feats["l3"], feats["l4"]

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.extract_multiscale(x, domain)["l4"]

    def adapter_parameters(self, domain: str | None = None):
        if domain is None:
            for p in self.domain_adapters.parameters():
                yield p
        else:
            for p in self.domain_adapters[domain].parameters():
                yield p
