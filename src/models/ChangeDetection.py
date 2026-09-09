"""U-Net CD: frozen ResNet encoder + adapters, shared decoder with domain BatchNorm + residual adapters."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import STAGE_CHANNELS, ResidualAdapter


def build_bitemporal_fusion(f1: torch.Tensor, f2: torch.Tensor, fusion_type: str = "abs") -> torch.Tensor:
    """Concatenate time steps based on fusion_type."""
    if fusion_type == "abs_prod":
        return torch.cat([f1, f2, torch.abs(f1 - f2), f1 * f2], dim=1)
    else:
        return torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class DomainConvBlock(nn.Module):
    """Shared conv + per-domain BatchNorm + ReLU + per-domain residual adapter."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        domain_list: Iterable[str],
        kernel_size: int = 3,
        padding: int = 1,
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm = nn.ModuleDict({
            d: nn.BatchNorm2d(out_ch) for d in domain_list
        })
        self.act = nn.ReLU(inplace=True)
        self.adapters = nn.ModuleDict({
            d: ResidualAdapter(out_ch, reduction=adapter_reduction, dropout=adapter_dropout)
            for d in domain_list
        })

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        x = self.act(self.norm[domain](self.conv(x)))
        return self.adapters[domain](x)


class UNetUpStage(nn.Module):
    """ConvTranspose2d upsample → concat skip → shared conv + domain BN + domain adapter."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        domain_list: Iterable[str],
        upsample_stride: int = 2,
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=upsample_stride, stride=upsample_stride
        )
        merge_in = in_ch + skip_ch if skip_ch > 0 else in_ch
        self.merge_conv = nn.Sequential(
            DomainConvBlock(
                merge_in, out_ch, domain_list,
                adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
            ),
            DomainConvBlock(
                out_ch, out_ch, domain_list,
                adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
            ),
        )

    def forward(
        self, x: torch.Tensor, domain: str, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.upconv(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Upsampled shape {x.shape[-2:]} != skip shape {skip.shape[-2:]}. "
                    "Use an input size divisible by 32."
                )
            x = torch.cat([x, skip], dim=1)
        for block in self.merge_conv:
            x = block(x, domain)
        return x


class UNetDecoder(nn.Module):
    """Shared trainable U-Net decoder; BatchNorm + residual adapters are per-domain."""

    def __init__(
        self,
        domain_list: Iterable[str],
        prior: float = 0.02,
        fusion_type: str = "abs",
        use_attention: bool = False,
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.domain_list = list(domain_list)
        self.fusion_type = fusion_type
        self.use_attention = use_attention

        mult = 4 if fusion_type == "abs_prod" else 3
        fused_channels = {k: mult * STAGE_CHANNELS[k] for k in ("l1", "l2", "l3", "l4")}

        if use_attention:
            self.attention = CBAM(fused_channels["l4"])
        else:
            self.attention = nn.Identity()

        self.bottleneck = DomainConvBlock(
            fused_channels["l4"], 512, domain_list, kernel_size=1, padding=0,
            adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
        )

        self.up_stages = nn.ModuleList([
            UNetUpStage(512, fused_channels["l3"], 256, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout),
            UNetUpStage(256, fused_channels["l2"], 128, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout),
            UNetUpStage(128, fused_channels["l1"], 64, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout),
            UNetUpStage(64, 0, 32, domain_list, upsample_stride=4,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout),
        ])

        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        # Deep supervision on the bottleneck output (512ch at H/32), before the
        # first up-stage.  Upsampled to the input size in ChangeDetectionModel.
        self.aux_classifier = nn.Conv2d(512, 1, kernel_size=1)

        prior_bias = math.log(prior / (1.0 - prior))
        for head in (self.classifier, self.aux_classifier):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, prior_bias)

    def forward(
        self, fused_skips: Dict[str, torch.Tensor], domain: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_l4 = self.attention(fused_skips["l4"])
        x = self.bottleneck(fused_l4, domain)
        aux = self.aux_classifier(x)

        x = self.up_stages[0](x, domain, fused_skips["l3"])
        x = self.up_stages[1](x, domain, fused_skips["l2"])
        x = self.up_stages[2](x, domain, fused_skips["l1"])
        x = self.up_stages[3](x, domain)

        logits = self.classifier(x)
        return logits, aux

    def domain_parameters(self, domain: str) -> list:
        """Params private to one domain: per-domain BatchNorm + per-domain adapters.

        Any ``nn.ModuleDict`` keyed by domain name (``norm``, ``adapters``, ...)
        contributes its ``domain`` entry, so new per-domain submodules are
        picked up automatically without touching this method.
        """
        params: list = []
        for module in self.modules():
            if isinstance(module, nn.ModuleDict) and domain in module:
                params += list(module[domain].parameters())
        return params

    def shared_parameters(self) -> list:
        domain_param_ids = {id(p) for d in self.domain_list for p in self.domain_parameters(d)}
        return [p for p in self.parameters() if id(p) not in domain_param_ids]


class ChangeDetectionModel(nn.Module):
    """Bi-temporal U-Net CD: adapter encoder + shared decoder (domain BN + adapters)."""

    def __init__(
        self,
        backbone,
        domain_list: Iterable[str] | None = None,
        prior: float = 0.02,
        use_deep_supervision: bool = True,
        fusion_type: str = "abs",
        use_attention: bool = False,
        decoder_adapter_reduction: int = 16,
        decoder_adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_deep_supervision = use_deep_supervision
        self.fusion_type = fusion_type
        self.use_attention = use_attention

        if domain_list is None:
            domain_list = getattr(backbone, "domain_list", None)
        if domain_list is None:
            raise ValueError(
                "ChangeDetectionModel needs a domain_list (or a backbone with one)."
            )
        self.domain_list: List[str] = list(domain_list)

        self.decoder = UNetDecoder(
            self.domain_list,
            prior=prior,
            fusion_type=fusion_type,
            use_attention=use_attention,
            adapter_reduction=decoder_adapter_reduction,
            adapter_dropout=decoder_adapter_dropout,
        )

        # Fixed (structural) set of shared-parameter ids, decided once at
        # construction time. ``freeze_domain`` toggles ``requires_grad`` on
        # every parameter on every domain switch, so ``shared_parameters``
        # must NOT be derived from ``requires_grad`` (that state is exactly
        # what it's used to restore) — it must key off identity instead.
        shared_ids = {id(p) for p in self.decoder.shared_parameters()}
        if getattr(self.backbone, "unfreeze_layer4", False):
            shared_ids |= {id(p) for p in self.backbone.layer4.parameters()}
        self._shared_param_ids = shared_ids

    def _fuse_pyramid(
        self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            k: build_bitemporal_fusion(p1[k], p2[k], fusion_type=self.fusion_type)
            for k in p1
        }

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor, domain: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pyramid1 = self.backbone.extract_multiscale(img1, domain)
        pyramid2 = self.backbone.extract_multiscale(img2, domain)
        fused = self._fuse_pyramid(pyramid1, pyramid2)

        logits, aux = self.decoder(fused, domain)

        if self.use_deep_supervision and self.training and aux is not None:
            aux = F.interpolate(
                aux, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            aux = None

        return logits, aux

    def domain_parameters(self, domain: str) -> list:
        params = self.decoder.domain_parameters(domain)
        backbone = self.backbone
        if hasattr(backbone, "domain_parameters"):
            params += backbone.domain_parameters(domain)
        else:
            adapters = getattr(backbone, "domain_adapters", None)
            if adapters is not None and domain in adapters:
                params += list(adapters[domain].parameters())
        return params

    def shared_parameters(self) -> list:
        return [p for p in self.parameters() if id(p) in self._shared_param_ids]

