"""Plain single-domain change detection: frozen ResNet50 encoder + plain
trainable U-Net decoder.

This is the "individual" baseline -- one model trained and evaluated on a
single dataset -- against which the multi-domain adapter architectures are
compared.  It has no per-domain adapters and no per-domain BatchNorm.

Everything else is deliberately identical to ``ChangeDetection.py``: the same
frozen backbone, the same bi-temporal fusion, the same decoder shape and
strides, the same AuxDecoder on the first up-stage, the same prior-biased
heads.  The only difference is that nothing is per-domain.  Keeping the two
models structurally identical is what makes the baseline comparable.

The model exposes the same interface as ChangeDetectionModel -- a ``domain``
argument it ignores, plus ``domain_parameters`` / ``shared_parameters`` -- so
it can be driven by the same MultiDomainTrainer and therefore go through the
identical evaluation, checkpointing and metric code.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.models.ChangeDetection import AuxDecoder, CBAM, build_bitemporal_fusion
from src.models.adapter_resnet import STAGE_CHANNELS

# The AuxDecoder keys its BatchNorms by domain.  With a single key it is an
# ordinary BatchNorm, so the plain model reuses that module rather than
# duplicating it -- identical parameters, identical behaviour.
_SINGLE = "_"


class PlainResNetEncoder(nn.Module):
    """Frozen ResNet50 feature pyramid (l1..l4), no adapters."""

    def __init__(self):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorms must never update their running statistics.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def extract_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


class PlainConvBlock(nn.Module):
    """Conv + BatchNorm + ReLU -- ordinary, no per-domain state."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PlainUpStage(nn.Module):
    """ConvTranspose2d upsample -> concat skip -> plain conv blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, upsample_stride: int = 2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=upsample_stride, stride=upsample_stride
        )
        merge_in = in_ch + skip_ch if skip_ch > 0 else in_ch
        self.merge_conv = nn.Sequential(
            PlainConvBlock(merge_in, out_ch),
            PlainConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upconv(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Upsampled shape {x.shape[-2:]} != skip shape {skip.shape[-2:]}. "
                    "Use an input size divisible by 32."
                )
            x = torch.cat([x, skip], dim=1)
        return self.merge_conv(x)


class SimpleUNetDecoder(nn.Module):
    """Plain trainable U-Net decoder -- ordinary BatchNorm, no adapters.

    Same shape as UNetDecoder: 1x1 bottleneck, four up-stages at strides
    2, 2, 2, 4, and the auxiliary decoder branching off the first up-stage.
    """

    def __init__(
        self,
        prior: float = 0.02,
        fusion_type: str = "abs",
        use_attention: bool = False,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        mult = 4 if fusion_type == "abs_prod" else 3
        fused_channels = {k: mult * STAGE_CHANNELS[k] for k in ("l1", "l2", "l3", "l4")}

        self.attention = CBAM(fused_channels["l4"]) if use_attention else nn.Identity()
        self.bottleneck = PlainConvBlock(fused_channels["l4"], 512, kernel_size=1, padding=0)

        self.up_stages = nn.ModuleList([
            PlainUpStage(512, fused_channels["l3"], 256, upsample_stride=2),
            PlainUpStage(256, fused_channels["l2"], 128, upsample_stride=2),
            PlainUpStage(128, fused_channels["l1"], 64, upsample_stride=2),
            PlainUpStage(64, 0, 32, upsample_stride=4),
        ])

        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        # Auxiliary branch off the FIRST up-stage (256ch at H/16), lifted back to
        # full resolution by four stride-2 transposed convs -- the same AuxDecoder
        # the multi-domain model uses, with a single BatchNorm key.
        self.aux_decoder = (
            AuxDecoder(256, [_SINGLE], widths=(128, 64, 32, 16))
            if use_deep_supervision else None
        )

        prior_bias = math.log(prior / (1.0 - prior))
        heads = [self.classifier]
        if self.aux_decoder is not None:
            heads.append(self.aux_decoder.classifier)
        for head in heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, prior_bias)

    def forward(
        self, fused_skips: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused_l4 = self.attention(fused_skips["l4"])
        x = self.bottleneck(fused_l4)

        x = self.up_stages[0](x, fused_skips["l3"])
        aux = self.aux_decoder(x, _SINGLE) if self.aux_decoder is not None else None

        x = self.up_stages[1](x, fused_skips["l2"])
        x = self.up_stages[2](x, fused_skips["l1"])
        x = self.up_stages[3](x)

        logits = self.classifier(x)
        return logits, aux


class SimpleChangeDetectionModel(nn.Module):
    """Bi-temporal U-Net CD: one frozen encoder + one plain trainable decoder.

    Instantiated fresh and trained independently for each dataset, as the
    "individual" baseline.  ``forward`` takes a ``domain`` argument purely so
    that MultiDomainTrainer can drive it unchanged; the value is ignored.
    """

    def __init__(
        self,
        domain_list: Iterable[str] | None = None,
        prior: float = 0.02,
        use_deep_supervision: bool = True,
        fusion_type: str = "abs",
        use_attention: bool = False,
    ):
        super().__init__()
        self.encoder = PlainResNetEncoder()
        self.decoder = SimpleUNetDecoder(
            prior=prior,
            fusion_type=fusion_type,
            use_attention=use_attention,
            use_deep_supervision=use_deep_supervision,
        )
        self.use_deep_supervision = use_deep_supervision
        self.fusion_type = fusion_type
        self.domain_list: List[str] = list(domain_list) if domain_list else []
        # No guided adapters here; the trainer reads this and finds nothing.
        self.routing_balance: Optional[torch.Tensor] = None

        # Fixed set of trainable parameter ids, decided once.  freeze_domain
        # flips requires_grad on every parameter each step, so this must key
        # off identity rather than the current requires_grad state -- and it
        # must exclude the frozen encoder, which would otherwise be unfrozen.
        self._trainable_ids = {id(p) for p in self.decoder.parameters()}

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor, domain: str | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        p1 = self.encoder.extract_multiscale(img1)
        p2 = self.encoder.extract_multiscale(img2)
        fused = {
            k: build_bitemporal_fusion(p1[k], p2[k], fusion_type=self.fusion_type) for k in p1
        }
        logits, aux = self.decoder(fused)

        if not self.use_deep_supervision:
            aux = None
        elif aux is not None and aux.shape[-2:] != img1.shape[-2:]:
            # AuxDecoder already emits full resolution; this only fires if the
            # input size is not divisible by 32.
            aux = F.interpolate(
                aux, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits, aux

    # -- MultiDomainTrainer interface -------------------------------------
    def domain_parameters(self, domain: str | None = None) -> list:
        """Everything trainable belongs to the single domain being trained."""
        return [p for p in self.parameters() if id(p) in self._trainable_ids]

    def shared_parameters(self) -> list:
        """Nothing is shared: there is only one domain."""
        return []
