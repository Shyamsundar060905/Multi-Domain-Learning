"""FCN-style encoder: frozen ImageNet ResNet-50 with dilated convolutions + per-domain adapters.

FCN (Fully Convolutional Network) uses a ResNet-50 backbone with dilated (atrous)
convolutions so that the spatial resolution is preserved deeper in the network:

  layer3 uses dilation=2  → output stride stays at H/16  (not H/32)
  layer4 uses dilation=4  → output stride stays at H/16  (not H/32)

This gives denser feature maps at the deepest level compared to the standard ResNet-50,
which is the key difference between a plain ResNet encoder and an FCN-style encoder.

We expose 4 skip levels (l1..l4) matching the U-Net decoder's expectations:
  l1 = layer1 output   256 ch @ H/4
  l2 = layer2 output   512 ch @ H/8
  l3 = layer3 output  1024 ch @ H/16  (dilation=2, same spatial as standard)
  l4 = layer4 output  2048 ch @ H/16  (dilation=4, denser than standard ResNet)

Note: because layer4 now outputs at H/16 (not H/32), the last up-stage in the
decoder needs upsample_stride=4 to go back to H/4 before the final x4 up-stage.
The STAGE_CHANNELS dict and UPSAMPLE_STRIDES dict both advertise this so the
decoder/factory can size itself correctly.

One ResidualAdapter per selected stage, per domain (identical to ResNetWithAdapters).
Both temporal streams are run through the frozen FCN backbone independently (adapters
are per-stream since there is no guided/change-aware routing in this encoder).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


# ---------------------------------------------------------------------------
# Residual adapter
# ---------------------------------------------------------------------------

class ResidualAdapter(nn.Module):
    """1x1 bottleneck adapter with identity init and optional dropout."""

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(channels // reduction, 8)

        self.down    = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(bottleneck)
        self.act     = nn.ReLU(inplace=True)
        self.up      = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.bn(self.down(x)))))


# ---------------------------------------------------------------------------
# Channel widths at each pyramid level
# ---------------------------------------------------------------------------

#: Advertised to ChangeDetectionModel so the decoder sizes itself correctly.
STAGE_CHANNELS: Dict[str, int] = {
    "l1":  256,   # ResNet layer1 output
    "l2":  512,   # ResNet layer2 output
    "l3": 1024,   # ResNet layer3 output (dilation=2)
    "l4": 2048,   # ResNet layer4 output (dilation=4, same spatial as l3)
}

#: How much each up-stage must upsample (consumed by the decoder factory).
#: l4→l3 is ×1 (same size), l3→l2 is ×2, l2→l1 is ×2, l1→full is ×4.
UPSAMPLE_STRIDES: Dict[str, int] = {
    "up0": 1,   # l4 → l3  (both at H/16, no spatial upsampling needed)
    "up1": 2,   # l3 → l2  (H/16 → H/8)
    "up2": 2,   # l2 → l1  (H/8  → H/4)
    "up3": 4,   # l1 → H   (H/4  → H)
}


# ---------------------------------------------------------------------------
# Helper: ensure layer4 is dilated if not already
# ---------------------------------------------------------------------------

def _ensure_layer4_dilated(layer4: nn.Module, dilation: int = 2) -> nn.Module:
    """Ensure layer4 uses dilation instead of stride=2 if a non-dilated base was passed."""
    for module in layer4.modules():
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3) and module.stride == (2, 2):
                module.stride = (1, 1)
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)
            elif module.kernel_size == (1, 1) and module.stride == (2, 2):
                module.stride = (1, 1)
    return layer4


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class FCNWithAdapters(nn.Module):
    """Frozen ImageNet ResNet-50 with dilated layer4 + per-domain adapters.

    This is the FCN-style (Fully Convolutional Network) encoder backbone.
    Compared to the standard ResNet-50 encoder:
    - layer1: H/4 (256 ch)
    - layer2: H/8 (512 ch)
    - layer3: H/16 (1024 ch)
    - layer4: H/16 (2048 ch, dilated with rate=2 instead of downsampling to H/32)

    This preserves denser spatial features at the deepest representation level.
    The decoder uses UPSAMPLE_STRIDES (up0=1, up1=2, up2=2, up3=4) to reconstruct
    back to full image resolution.

    Parameters
    ----------
    base:
        A ``torchvision.models.resnet50`` model (ideally constructed with
        ``replace_stride_with_dilation=[False, False, True]``).
    domain_list:
        Ordered list of domain names (e.g. ``["LEVIR", "WHU"]``).
    adapter_stages:
        Subset of ``{"l1","l2","l3","l4"}`` to insert per-domain adapters at.
        Default: all four levels.
    adapter_reduction:
        Bottleneck reduction factor inside each adapter.
    adapter_dropout:
        Dropout probability inside each adapter (0 = no dropout).
    """

    STAGE_CHANNELS: Dict[str, int] = STAGE_CHANNELS
    UPSAMPLE_STRIDES: Dict[str, int] = UPSAMPLE_STRIDES

    def __init__(
        self,
        base: nn.Module,
        domain_list: Iterable[str],
        adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()

        self.domain_list    = list(domain_list)
        self.adapter_stages = list(adapter_stages)

        # --- Stem and stages -------------------------------------------------
        # Stem: conv1, bn1, relu, maxpool  ->  H/4
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        # Layer 1: output  256 ch @ H/4
        self.layer1 = base.layer1
        # Layer 2: output  512 ch @ H/8
        self.layer2 = base.layer2
        # Layer 3: output 1024 ch @ H/16
        self.layer3 = base.layer3
        # Layer 4: output 2048 ch @ H/16 (dilated)
        self.layer4 = _ensure_layer4_dilated(base.layer4, dilation=2)

        # Internal map: level key -> (module_name, channels)
        self._level_to_stage = {
            "l1": ("layer1", 256),
            "l2": ("layer2", 512),
            "l3": ("layer3", 1024),
            "l4": ("layer4", 2048),
        }

        # --- Per-domain residual adapters, one per selected level ------------
        self.domain_adapters = nn.ModuleDict({
            d: nn.ModuleDict({
                lvl: ResidualAdapter(
                    STAGE_CHANNELS[lvl],
                    reduction=adapter_reduction,
                    dropout=adapter_dropout,
                )
                for lvl in self.adapter_stages
            })
            for d in self.domain_list
        })

        # --- Freeze the backbone ---------------------------------------------
        self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters; leave adapters trainable."""
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for p in module.parameters():
                p.requires_grad = False
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode: bool = True):
        """Keep frozen BN layers in eval mode during training."""
        super().train(mode)
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        """Return only the parameters that belong to ``domain``."""
        return list(self.domain_adapters[domain].parameters())

    # ------------------------------------------------------------------
    def _apply_adapter(
        self, x: torch.Tensor, domain: str, level: str
    ) -> torch.Tensor:
        if level in self.adapter_stages and level in self.domain_adapters[domain]:
            return self.domain_adapters[domain][level](x)
        return x

    # ------------------------------------------------------------------
    def extract_multiscale(
        self, x: torch.Tensor, domain: str
    ) -> Dict[str, torch.Tensor]:
        """Run the dilated FCN pyramid and return ``{l1, l2, l3, l4}``."""
        if domain not in self.domain_adapters:
            raise KeyError(
                f"Unknown domain '{domain}'. Known: {list(self.domain_adapters)}"
            )

        x = self.stem(x)                              # H/4

        l1 = self.layer1(x)                           # 256  @ H/4
        l1 = self._apply_adapter(l1, domain, "l1")

        l2 = self.layer2(l1)                          # 512  @ H/8
        l2 = self._apply_adapter(l2, domain, "l2")

        l3 = self.layer3(l2)                          # 1024 @ H/16 (dilated)
        l3 = self._apply_adapter(l3, domain, "l3")

        l4 = self.layer4(l3)                          # 2048 @ H/16 (dilated)
        l4 = self._apply_adapter(l4, domain, "l4")

        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.extract_multiscale(x, domain)["l4"]
