"""Multi-domain ResNet backbone with frozen weights + lightweight residual adapters."""

from __future__ import annotations

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


class ResNetWithAdapters(nn.Module):
    """ResNet50 backbone (frozen) with per-domain residual adapters in layer3/4."""

    def __init__(self, base, domain_list, adapter_dropout: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.feature_channels = 2048
        self.mid_channels = 1024
        self.domain_list = list(domain_list)

        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                "layer3": nn.ModuleList(
                    [ResidualAdapter(1024, dropout=adapter_dropout) for _ in self.layer3]
                ),
                "layer4": nn.ModuleList(
                    [ResidualAdapter(2048, dropout=adapter_dropout) for _ in self.layer4]
                ),
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

    def extract_features(self, x: torch.Tensor, domain: str):
        """Return (layer3, layer4) feature maps for deep supervision."""
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self._run_stage(self.layer3, self.domain_adapters[domain]["layer3"], x)
        f4 = self._run_stage(self.layer4, self.domain_adapters[domain]["layer4"], f3)
        return f3, f4

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        _, f4 = self.extract_features(x, domain)
        return f4

    def adapter_parameters(self, domain: str | None = None):
        if domain is None:
            for p in self.domain_adapters.parameters():
                yield p
        else:
            for p in self.domain_adapters[domain].parameters():
                yield p
