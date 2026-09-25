"""U-Net-style encoder: frozen ImageNet ResNet50 pyramid + per-domain adapters.

Two encoder adapter types, both per-domain:

- ``simple``: ResidualAdapter after every bottleneck block, applied to each
  temporal stream independently.
- ``guided``: ChangeGuidedDynamicAdapter, which takes BOTH temporal streams and
  adapts them jointly, conditioned on their difference.  Placed either once per
  ResNet stage (at the stage output) or after every bottleneck block.

A guided adapter needs x1 and x2 at the same point in the network, so the
encoder runs the two timesteps in lockstep, block by block, instead of as two
independent passes.  For ``simple`` adapters this is the same computation as
before: every frozen block and adapter still sees one stream at a time, so
adapter BatchNorm statistics stay per stream.  Only the order in which dropout
masks are drawn changes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.guided_adapter import ChangeGuidedDynamicAdapter, routing_balance_loss

_STAGES = ("layer1", "layer2", "layer3", "layer4")


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


class ResNetWithAdapters(nn.Module):
    """Frozen ResNet50 encoder pyramid + trainable per-domain adapters.

    Outputs ``{l1..l4}`` at H/4, H/8, H/16, H/32 for each timestep, for the
    U-Net skip connections.
    """

    def __init__(
        self,
        base: nn.Module,
        domain_list: Iterable[str],
        adapter_dropout: float = 0.1,
        adapter_reduction: int = 16,
        domain_bn_in_adapter: bool = False,
        unfreeze_layer4: bool = False,
        adapter_stages: Iterable[str] = _STAGES,
        adapter_type: str = "guided",
        guided_granularity: str = "stage",
        num_experts: int = 4,
        guided_reduction: int = 16,
        router_top_k: Optional[int] = 2,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        if adapter_type not in {"simple", "guided"}:
            raise ValueError(f"adapter_type must be 'simple' or 'guided', got {adapter_type!r}")
        if guided_granularity not in {"stage", "block"}:
            raise ValueError(
                f"guided_granularity must be 'stage' or 'block', got {guided_granularity!r}"
            )

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
        self.adapter_type = adapter_type
        self.guided_granularity = guided_granularity
        # Simple adapters always sit after every block; guided adapters follow
        # guided_granularity.
        self.adapter_per_block = adapter_type == "simple" or guided_granularity == "block"
        # Load-balancing loss over guided routing, refreshed on every forward.
        self.routing_balance: Optional[torch.Tensor] = None

        def make_adapter(channels: int) -> nn.Module:
            if adapter_type == "simple":
                return ResidualAdapter(
                    channels, reduction=adapter_reduction, dropout=adapter_dropout
                )
            return ChangeGuidedDynamicAdapter(
                channels,
                num_experts=num_experts,
                reduction=guided_reduction,
                temperature=router_temperature,
                top_k=router_top_k,
            )

        def n_adapters(stage: str) -> int:
            return len(getattr(self, stage)) if self.adapter_per_block else 1

        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                stage: nn.ModuleList([
                    make_adapter(STAGE_CHANNELS[f"l{stage[-1]}"])
                    for _ in range(n_adapters(stage))
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

    def _run_stage_pair(
        self,
        stage: nn.Sequential,
        adapters: nn.ModuleList | None,
        bns: nn.ModuleList | None,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run one ResNet stage on both timesteps in lockstep.

        Returns both stage outputs and the routing weights of any guided
        adapters that ran, for the load-balancing loss.
        """
        routing: List[torch.Tensor] = []
        last = len(stage) - 1
        for i, block in enumerate(stage):
            x1, x2 = block(x1), block(x2)
            if bns is not None:
                x1, x2 = bns[i](x1), bns[i](x2)
            if adapters is None:
                continue
            if self.adapter_per_block:
                adapter = adapters[i]
            elif i == last:
                adapter = adapters[0]
            else:
                continue
            if self.adapter_type == "simple":
                x1, x2 = adapter(x1), adapter(x2)
            else:
                x1, x2, info = adapter(x1, x2)
                routing.append(info["routing_weights"])
        return x1, x2, routing

    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        params = list(self.domain_adapters[domain].parameters())
        if hasattr(self, "domain_bns") and domain in self.domain_bns:
            params += list(self.domain_bns[domain].parameters())
        return params

    def extract_multiscale_pair(
        self, x1: torch.Tensor, x2: torch.Tensor, domain: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pyramids ``{l1..l4}`` for both timesteps, run through the encoder together."""
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        ad = self.domain_adapters[domain]
        bns = self.domain_bns[domain] if hasattr(self, "domain_bns") else None

        x1, x2 = self.stem(x1), self.stem(x2)
        p1: Dict[str, torch.Tensor] = {}
        p2: Dict[str, torch.Tensor] = {}
        routing: List[torch.Tensor] = []
        for k, stage_name in enumerate(_STAGES, start=1):
            adapters = ad[stage_name] if stage_name in ad else None
            stage_bns = bns[stage_name] if (bns is not None and stage_name in bns) else None
            x1, x2, r = self._run_stage_pair(
                getattr(self, stage_name), adapters, stage_bns, x1, x2
            )
            p1[f"l{k}"], p2[f"l{k}"] = x1, x2
            routing.extend(r)

        self.routing_balance = (
            torch.stack([routing_balance_loss(w) for w in routing]).mean()
            if routing else None
        )
        return p1, p2

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, domain: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.extract_multiscale_pair(x1, x2, domain)
