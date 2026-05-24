"""Multi-domain ResNet backbone with frozen weights + lightweight residual adapters.

Design:
- The original ResNet (stem, layer1..layer4) is shared across all domains and frozen.
- For each domain we add a small ``ResidualAdapter`` after every bottleneck in
  layer3 and layer4. The adapter is the ONLY domain-specific, trainable piece
  inside the backbone.
- Forward signature: ``backbone(x, domain)`` -- routes through the requested
  domain's adapters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """1x1 bottleneck adapter with a residual (identity) skip connection.

    Output = x + W2(ReLU(BN(W1(x)))).  The second conv is zero-initialised so
    the adapter starts as the identity function, which means inserting it into
    a pretrained network does not perturb features at step 0.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 8)

        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)  # start as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.bn(self.down(x))))


class ResNetWithAdapters(nn.Module):
    """ResNet50 backbone (frozen) with per-domain residual adapters in layer3/4."""

    def __init__(self, base, domain_list):
        super().__init__()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   #  256 ch
        self.layer2 = base.layer2   #  512 ch
        self.layer3 = base.layer3   # 1024 ch
        self.layer4 = base.layer4   # 2048 ch

        self.feature_channels = 2048
        self.domain_list = list(domain_list)

        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                "layer3": nn.ModuleList(
                    [ResidualAdapter(1024) for _ in self.layer3]
                ),
                "layer4": nn.ModuleList(
                    [ResidualAdapter(2048) for _ in self.layer4]
                ),
            })
            for d in self.domain_list
        })

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze stem + every layer; only ``domain_adapters`` stays trainable."""
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

    def _run_stage(self, stage: nn.Sequential, adapters: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block, adapter in zip(stage, adapters):
            x = block(x)
            x = adapter(x)
        return x

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self._run_stage(self.layer3, self.domain_adapters[domain]["layer3"], x)
        x = self._run_stage(self.layer4, self.domain_adapters[domain]["layer4"], x)
        return x

    def adapter_parameters(self, domain: str | None = None):
        """Yield trainable adapter parameters, optionally restricted to one domain."""
        if domain is None:
            for p in self.domain_adapters.parameters():
                yield p
        else:
            for p in self.domain_adapters[domain].parameters():
                yield p
