"""U-Net-style encoder: frozen ImageNet ResNet50 pyramid + per-domain adapters."""

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


# Pyramid channel widths at each scale (ResNet50 stage outputs).
STAGE_CHANNELS = {
    "l1": 256,
    "l2": 512,
    "l3": 1024,
    "l4": 2048,
}

_RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")


class ResNetWithAdapters(nn.Module):
    """Frozen ResNet50 encoder pyramid + trainable per-domain adapters.

    Adapters sit after every ResNet bottleneck block.  Outputs ``{l1..l4}`` at
    H/4, H/8, H/16, H/32 for U-Net skip connections.
    """

    def __init__(
        self,
        base: nn.Module,
        domain_list: Iterable[str],
        adapter_dropout: float = 0.1,
        adapter_reduction: int = 16,
        domain_bn_in_adapter: bool = False,
        unfreeze_layer4: bool = False,
        adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
    ):
        super().__init__()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.feature_channels = STAGE_CHANNELS["l4"]
        self.domain_list = list(domain_list)
        self.domain_bn_in_adapter = domain_bn_in_adapter
        self.unfreeze_layer4 = unfreeze_layer4
        self.adapter_stages = list(adapter_stages)

        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                stage: nn.ModuleList([
                    ResidualAdapter(
                        STAGE_CHANNELS[f"l{stage[-1]}"],
                        reduction=adapter_reduction,
                        dropout=adapter_dropout,
                    )
                    for _ in getattr(self, stage)
                ])
                for stage in self.adapter_stages
            })
            for d in self.domain_list
        })

        if self.domain_bn_in_adapter:
            self.domain_bns = nn.ModuleDict({
                d: nn.ModuleDict({
                    stage: nn.ModuleList([
                        nn.BatchNorm2d(STAGE_CHANNELS[f"l{stage[-1]}"])
                        for _ in getattr(self, stage)
                    ])
                    for stage in self.adapter_stages
                })
                for d in self.domain_list
            })

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for name, module in self.named_children():
            if name in ("domain_adapters", "domain_bns"):
                continue
            if self.unfreeze_layer4 and name == "layer4":
                for p in module.parameters():
                    p.requires_grad = True
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            else:
                for p in module.parameters():
                    p.requires_grad = False
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in self.named_children():
            if name in ("domain_adapters", "domain_bns"):
                continue
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _run_stage(
        self,
        stage: nn.Sequential,
        adapters: nn.ModuleList | None,
        bns: nn.ModuleList | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for i, block in enumerate(stage):
            x = block(x)
            if bns is not None:
                x = bns[i](x)
            if adapters is not None:
                x = adapters[i](x)
        return x

    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        params = list(self.domain_adapters[domain].parameters())
        if hasattr(self, "domain_bns") and domain in self.domain_bns:
            params += list(self.domain_bns[domain].parameters())
        return params

    def adapter_parameters(self, domain: str | None = None):
        if domain is None:
            for p in self.domain_adapters.parameters():
                yield p
            if hasattr(self, "domain_bns"):
                for p in self.domain_bns.parameters():
                    yield p
        else:
            for p in self.domain_adapters[domain].parameters():
                yield p
            if hasattr(self, "domain_bns") and domain in self.domain_bns:
                for p in self.domain_bns[domain].parameters():
                    yield p

    def extract_multiscale(self, x: torch.Tensor, domain: str) -> Dict[str, torch.Tensor]:
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        ad = self.domain_adapters[domain]
        bns = self.domain_bns[domain] if hasattr(self, "domain_bns") else None

        x = self.stem(x)
        l1 = self._run_stage(self.layer1, ad["layer1"] if "layer1" in ad else None, bns["layer1"] if (bns and "layer1" in bns) else None, x)
        l2 = self._run_stage(self.layer2, ad["layer2"] if "layer2" in ad else None, bns["layer2"] if (bns and "layer2" in bns) else None, l1)
        l3 = self._run_stage(self.layer3, ad["layer3"] if "layer3" in ad else None, bns["layer3"] if (bns and "layer3" in bns) else None, l2)
        l4 = self._run_stage(self.layer4, ad["layer4"] if "layer4" in ad else None, bns["layer4"] if (bns and "layer4" in bns) else None, l3)
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def extract_features(self, x: torch.Tensor, domain: str):
        feats = self.extract_multiscale(x, domain)
        return feats["l3"], feats["l4"]

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.extract_multiscale(x, domain)["l4"]


# Alias kept for callers that import this name from main.
UNetEncoderWithAdapters = ResNetWithAdapters
