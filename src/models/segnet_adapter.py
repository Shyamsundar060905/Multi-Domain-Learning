"""U-Net-style encoder: frozen ImageNet SegNet (VGG-16 with BatchNorm) pyramid + per-domain adapters.

SegNet feature-map hierarchy (13 conv layers with BatchNorm + ReLU + MaxPool):
  Stage 1:  64 ch  @ H/2  (conv1_1 -> bn -> relu -> conv1_2 -> bn -> relu -> maxpool)
  Stage 2: 128 ch  @ H/4  (conv2_1 -> bn -> relu -> conv2_2 -> bn -> relu -> maxpool)
  Stage 3: 256 ch  @ H/8  (conv3_1 -> bn -> relu -> conv3_2 -> bn -> relu -> conv3_3 -> bn -> relu -> maxpool)
  Stage 4: 512 ch  @ H/16 (conv4_1 -> bn -> relu -> conv4_2 -> bn -> relu -> conv4_3 -> bn -> relu -> maxpool)
  Stage 5: 512 ch  @ H/32 (conv5_1 -> bn -> relu -> conv5_2 -> bn -> relu -> conv5_3 -> bn -> relu -> maxpool)

We expose 4 skip levels (l1..l4) matching the U-Net decoder's expectations:
  l1 = Stage 2 output  (128 ch @ H/4)
  l2 = Stage 3 output  (256 ch @ H/8)
  l3 = Stage 4 output  (512 ch @ H/16)
  l4 = Stage 5 output  (512 ch @ H/32)

Stage 1 (64 ch @ H/2) is consumed by the stem pass.

One ResidualAdapter sits at the END of each stage (after max-pool), providing
per-domain adaptation at every scale.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Residual adapter
# ---------------------------------------------------------------------------

class ResidualAdapter(nn.Module):
    """1x1 bottleneck adapter with identity init and optional dropout."""

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(channels // reduction, 8)

        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(bottleneck)
        self.act  = nn.ReLU(inplace=True)
        self.up   = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.bn(self.down(x)))))


# ---------------------------------------------------------------------------
# Channel widths for U-Net skip connections (must match UNetDecoder's STAGE_CHANNELS).
# ---------------------------------------------------------------------------

STAGE_CHANNELS: Dict[str, int] = {
    "l1": 128,   # SegNet stage 2 output
    "l2": 256,   # SegNet stage 3 output
    "l3": 512,   # SegNet stage 4 output
    "l4": 512,   # SegNet stage 5 output
}


# ---------------------------------------------------------------------------
# Helper: split VGG16_BN / SegNet features into 5 sequential stage modules
# ---------------------------------------------------------------------------

def _split_segnet_stages(features: nn.Sequential) -> Tuple[
    nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential
]:
    """Dynamically partition features by MaxPool2d into 5 stages."""
    stages = []
    current_stage = []
    for layer in features.children():
        current_stage.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            stages.append(nn.Sequential(*current_stage))
            current_stage = []
    if current_stage:
        stages.append(nn.Sequential(*current_stage))
    if len(stages) != 5:
        raise ValueError(f"Expected 5 MaxPool2d stages in SegNet encoder, found {len(stages)}.")
    return tuple(stages)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class SegNetWithAdapters(nn.Module):
    """Frozen SegNet (VGG-16 with BN) encoder pyramid + trainable per-domain residual adapters.

    Exposes ``{l1, l2, l3, l4}`` feature maps for U-Net skip connections at
    H/4, H/8, H/16 and H/32 respectively (for a 512x512 input).

    One adapter per exported scale, per domain. Adapters are applied after
    the max-pool that ends each SegNet stage, so they see spatially-downsampled
    features -- cheaper and stable.

    The class-level ``STAGE_CHANNELS`` dict is read by ``ChangeDetectionModel``
    to size its decoder correctly for SegNet width (128/256/512/512).

    Parameters
    ----------
    base:
        Pre-trained ``torchvision.models.vgg16_bn`` instance.
    domain_list:
        Ordered list of domain names (e.g. ``["LEVIR", "WHU"]``).
    adapter_dropout:
        Dropout probability inside each adapter (0 = no dropout).
    adapter_reduction:
        Bottleneck reduction factor for adapters.
    adapter_stages:
        Subset of ``{"l1","l2","l3","l4"}`` to insert adapters at.
        Default: all four scales.
    """

    STAGE_CHANNELS: Dict[str, int] = {
        "l1": 128,
        "l2": 256,
        "l3": 512,
        "l4": 512,
    }

    def __init__(
        self,
        base: nn.Module,
        domain_list: Iterable[str],
        adapter_dropout: float = 0.1,
        adapter_reduction: int = 16,
        adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    ):
        super().__init__()

        self.domain_list   = list(domain_list)
        self.adapter_stages = list(adapter_stages)

        # --- Split the flat features Sequential into 5 stage modules ----------
        s1, s2, s3, s4, s5 = _split_segnet_stages(base.features)
        self.stage1 = s1  # -> 64 ch  @ H/2
        self.stage2 = s2  # -> 128 ch @ H/4   (l1)
        self.stage3 = s3  # -> 256 ch @ H/8   (l2)
        self.stage4 = s4  # -> 512 ch @ H/16  (l3)
        self.stage5 = s5  # -> 512 ch @ H/32  (l4)

        self.feature_channels = STAGE_CHANNELS["l4"]  # 512

        # Stage name -> exported level key (l1..l4)
        self._stage_to_level = {
            "stage2": "l1",
            "stage3": "l2",
            "stage4": "l3",
            "stage5": "l4",
        }

        # --- Per-domain adapters (one per exported scale, per domain) ---------
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

        # --- Freeze backbone --------------------------------------------------
        self._freeze_backbone()

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all SegNet stage parameters; keep adapters trainable."""
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for p in module.parameters():
                p.requires_grad = False
            # Keep BN layers in eval mode -- they are frozen
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode: bool = True):
        """Always keep frozen BN layers in eval mode during training."""
        super().train(mode)
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    # Adapter / param helpers (same interface as ResNetWithAdapters)
    # ------------------------------------------------------------------

    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        """Return only the parameters that belong to ``domain``."""
        return list(self.domain_adapters[domain].parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _apply_adapter(
        self, x: torch.Tensor, domain: str, level: str
    ) -> torch.Tensor:
        """Apply domain adapter for ``level`` if that stage is configured."""
        if level in self.adapter_stages and level in self.domain_adapters[domain]:
            return self.domain_adapters[domain][level](x)
        return x

    def extract_multiscale(
        self, x: torch.Tensor, domain: str
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the SegNet pyramid with per-domain adapters.

        Parameters
        ----------
        x:
            Input image tensor of shape ``(B, 3, H, W)``.
        domain:
            Domain name string (must be in ``self.domain_list``).

        Returns
        -------
        dict with keys ``{"l1", "l2", "l3", "l4"}`` at scales
        H/4, H/8, H/16, H/32 with channels 128, 256, 512, 512.
        """
        # Stage 1: stem (H -> H/2, 64 ch) -- no adapter, no skip
        x = self.stage1(x)

        # Stage 2: (H/2 -> H/4, 128 ch) -> l1 skip
        x = self.stage2(x)
        l1 = self._apply_adapter(x, domain, "l1")

        # Stage 3: (H/4 -> H/8, 256 ch) -> l2 skip
        x = self.stage3(l1)
        l2 = self._apply_adapter(x, domain, "l2")

        # Stage 4: (H/8 -> H/16, 512 ch) -> l3 skip
        x = self.stage4(l2)
        l3 = self._apply_adapter(x, domain, "l3")

        # Stage 5: (H/16 -> H/32, 512 ch) -> l4 bottleneck
        x = self.stage5(l3)
        l4 = self._apply_adapter(x, domain, "l4")

        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def forward(
        self, x: torch.Tensor, domain: str
    ) -> Dict[str, torch.Tensor]:
        return self.extract_multiscale(x, domain)
