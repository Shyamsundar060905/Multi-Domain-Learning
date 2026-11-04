"""U-Net encoder: 5-stage DoubleConv pyramid with ImageNet-initialized weights + per-domain adapters.

U-Net encoder feature-map hierarchy:
  Stage 1:  64 ch  @ H/2  (DoubleConv(3, 64)   -> MaxPool2d)
  Stage 2: 128 ch  @ H/4  (DoubleConv(64, 128) -> MaxPool2d) -> l1 skip
  Stage 3: 256 ch  @ H/8  (DoubleConv(128, 256)-> MaxPool2d) -> l2 skip
  Stage 4: 512 ch  @ H/16 (DoubleConv(256, 512)-> MaxPool2d) -> l3 skip
  Stage 5: 512 ch  @ H/32 (DoubleConv(512, 512)-> MaxPool2d) -> l4 bottleneck

We expose 4 skip levels (l1..l4) matching the U-Net decoder's expectations:
  l1 = Stage 2 output (128 ch @ H/4)
  l2 = Stage 3 output (256 ch @ H/8)
  l3 = Stage 4 output (512 ch @ H/16)
  l4 = Stage 5 output (512 ch @ H/32)

Stage 1 (64 ch @ H/2) is consumed by the stem pass.

One ResidualAdapter sits at the END of each stage (after max-pool), providing
per-domain adaptation at every scale.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn


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
# Canonical U-Net DoubleConv Block
# ---------------------------------------------------------------------------

class UNetBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm2d and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        # Kaiming normal init
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


# ---------------------------------------------------------------------------
# Channel widths for U-Net skip connections (must match UNetDecoder's STAGE_CHANNELS).
# ---------------------------------------------------------------------------

STAGE_CHANNELS: Dict[str, int] = {
    "l1": 128,   # Stage 2 output
    "l2": 256,   # Stage 3 output
    "l3": 512,   # Stage 4 output
    "l4": 512,   # Stage 5 output
}


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class UNetWithAdapters(nn.Module):
    """Canonical U-Net encoder pyramid + trainable per-domain residual adapters.

    Exposes ``{l1, l2, l3, l4}`` feature maps for U-Net skip connections at
    H/4, H/8, H/16 and H/32 respectively (for a 512x512 input).

    With ``pretrained=True`` the convs are initialized from ImageNet VGG-16-BN.
    Stages 1-2 match VGG-16-BN exactly.  VGG-16-BN stages 3-5 have three convs
    and a U-Net block has two, so the third conv of those stages (VGG indices
    20, 30, 40) is not used: from stage 3 on, the features are NOT those of
    the pretrained network.  Using all VGG-16-BN layers is the SegNet encoder.

    Parameters
    ----------
    domain_list:
        Ordered list of domain names (e.g. ``["LEVIR", "WHU"]``).
    pretrained:
        Whether to load ImageNet pretrained weights into the DoubleConv blocks.
    unfreeze_backbone:
        If True, base DoubleConv blocks are trainable across domains (shared).
        If False (default), base blocks are frozen and only per-domain adapters train.
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
        domain_list: Iterable[str],
        pretrained: bool = True,
        unfreeze_backbone: bool = False,
        adapter_dropout: float = 0.1,
        adapter_reduction: int = 16,
        adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    ):
        super().__init__()

        self.domain_list = list(domain_list)
        self.adapter_stages = list(adapter_stages)
        self.unfreeze_backbone = unfreeze_backbone

        # --- 5 U-Net Encoder Stages -------------------------------------------
        self.stage1 = UNetBlock(3, 64)       # -> 64 ch
        self.pool1 = nn.MaxPool2d(2, 2)     # -> @ H/2

        self.stage2 = UNetBlock(64, 128)     # -> 128 ch
        self.pool2 = nn.MaxPool2d(2, 2)     # -> @ H/4  (l1)

        self.stage3 = UNetBlock(128, 256)    # -> 256 ch
        self.pool3 = nn.MaxPool2d(2, 2)     # -> @ H/8  (l2)

        self.stage4 = UNetBlock(256, 512)    # -> 512 ch
        self.pool4 = nn.MaxPool2d(2, 2)     # -> @ H/16 (l3)

        self.stage5 = UNetBlock(512, 512)    # -> 512 ch
        self.pool5 = nn.MaxPool2d(2, 2)     # -> @ H/32 (l4)

        self.feature_channels = STAGE_CHANNELS["l4"]  # 512

        # --- Optional ImageNet Pretrained Weight Transfer --------------------
        if pretrained:
            self._load_pretrained_weights()

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

        if not self.unfreeze_backbone:
            self._freeze_backbone()

    # (U-Net stage, conv/bn suffix, VGG16_BN conv index, VGG16_BN BatchNorm index).
    # VGG-16-BN stages 3-5 have a THIRD conv (indices 20, 30, 40) that a
    # two-conv U-Net block has no slot for, so those layers are not used.
    _VGG16_BN_MAP = (
        ("stage1", "1", 0, 1), ("stage1", "2", 3, 4),
        ("stage2", "1", 7, 8), ("stage2", "2", 10, 11),
        ("stage3", "1", 14, 15), ("stage3", "2", 17, 18),
        ("stage4", "1", 24, 25), ("stage4", "2", 27, 28),
        ("stage5", "1", 34, 35), ("stage5", "2", 37, 38),
    )

    def _load_pretrained_weights(self) -> None:
        """Initialize the DoubleConv blocks from ImageNet VGG-16-BN.

        Any failure raises.  The encoder is frozen, so silently falling back to
        random weights would train the whole model on random features.
        """
        try:
            vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Could not load ImageNet VGG-16-BN weights for the U-Net encoder. "
                "The encoder is frozen, so training on random weights would be "
                "meaningless; fix the download/cache and rerun."
            ) from exc
        f = list(vgg.features)
        for stage, k, conv_i, bn_i in self._VGG16_BN_MAP:
            block = getattr(self, stage)
            self._copy_conv_bn(f[conv_i], f[bn_i],
                               getattr(block, f"conv{k}"), getattr(block, f"bn{k}"))
        print("Loaded ImageNet pretrained weights into U-Net encoder "
              "(VGG-16-BN convs 0,3,7,10,14,17,24,27,34,37 with biases folded into "
              "BatchNorm; VGG convs 20,30,40 have no U-Net counterpart and are unused).")

    @staticmethod
    @torch.no_grad()
    def _copy_conv_bn(src_conv, src_bn, dst_conv, dst_bn) -> None:
        """Copy a VGG ``conv(+bias) -> BN`` pair into a U-Net ``conv -> BN`` pair.

        VGG-16-BN convs carry a bias; the U-Net convs are bias-free.  The bias is
        folded into the BatchNorm running mean, which is exact for an eval-mode
        (frozen) BatchNorm:  BN(Wx + b; mean) == BN(Wx; mean - b).  In train
        mode a per-channel constant is removed by the batch mean anyway, so the
        fold is harmless with --unfreeze-backbone too.
        """
        if not (isinstance(src_conv, nn.Conv2d) and isinstance(src_bn, nn.BatchNorm2d)):
            raise TypeError(
                f"expected Conv2d + BatchNorm2d, got "
                f"{type(src_conv).__name__} + {type(src_bn).__name__}"
            )
        if src_conv.weight.shape != dst_conv.weight.shape:
            raise ValueError(
                f"conv shape mismatch: VGG {tuple(src_conv.weight.shape)} "
                f"vs U-Net {tuple(dst_conv.weight.shape)}"
            )
        if src_bn.num_features != dst_bn.num_features or src_bn.eps != dst_bn.eps:
            raise ValueError("BatchNorm mismatch (num_features or eps)")

        bias = (src_conv.bias if src_conv.bias is not None
                else torch.zeros_like(src_bn.running_mean))
        dst_conv.weight.copy_(src_conv.weight)
        if dst_conv.bias is not None:
            dst_conv.bias.copy_(bias)
            dst_bn.running_mean.copy_(src_bn.running_mean)
        else:
            dst_bn.running_mean.copy_(src_bn.running_mean - bias)
        dst_bn.running_var.copy_(src_bn.running_var)
        dst_bn.weight.copy_(src_bn.weight)
        dst_bn.bias.copy_(src_bn.bias)

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all U-Net stage parameters; keep adapters trainable."""
        for name, module in self.named_children():
            if name == "domain_adapters":
                continue
            for p in module.parameters():
                p.requires_grad = False
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode: bool = True):
        """Always keep frozen BN layers in eval mode during training."""
        super().train(mode)
        if not self.unfreeze_backbone:
            for name, module in self.named_children():
                if name == "domain_adapters":
                    continue
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    # ------------------------------------------------------------------
    # Adapter / param helpers
    # ------------------------------------------------------------------

    def domain_parameters(self, domain: str) -> List[torch.nn.Parameter]:
        """Return only the parameters that belong to ``domain``."""
        return list(self.domain_adapters[domain].parameters())

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
        """Forward pass through the U-Net pyramid with per-domain adapters.

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
        # Stage 1: stem (H -> H/2, 64 ch)
        x = self.pool1(self.stage1(x))

        # Stage 2: (H/2 -> H/4, 128 ch) -> l1 skip
        x = self.pool2(self.stage2(x))
        l1 = self._apply_adapter(x, domain, "l1")

        # Stage 3: (H/4 -> H/8, 256 ch) -> l2 skip
        x = self.pool3(self.stage3(l1))
        l2 = self._apply_adapter(x, domain, "l2")

        # Stage 4: (H/8 -> H/16, 512 ch) -> l3 skip
        x = self.pool4(self.stage4(l2))
        l3 = self._apply_adapter(x, domain, "l3")

        # Stage 5: (H/16 -> H/32, 512 ch) -> l4 bottleneck
        x = self.pool5(self.stage5(l3))
        l4 = self._apply_adapter(x, domain, "l4")

        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def forward(
        self, x: torch.Tensor, domain: str
    ) -> Dict[str, torch.Tensor]:
        return self.extract_multiscale(x, domain)
