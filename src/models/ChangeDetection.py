"""U-Net CD: frozen ResNet encoder + adapters, shared decoder with domain BatchNorm + residual adapters."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapter_resnet import STAGE_CHANNELS, ResidualAdapter
from src.models.guided_adapter import ContextGuidedAdapter, routing_balance_loss


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
    """Shared conv + per-domain BatchNorm + ReLU + per-domain adapter.

    The adapter is a plain ResidualAdapter by default.  With
    ``adapter_type="guided"`` and a ``context_channels`` count it becomes a
    ContextGuidedAdapter, conditioned on the encoder's change map for this
    scale, which ``forward`` then expects as ``context``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        domain_list: Iterable[str],
        kernel_size: int = 3,
        padding: int = 1,
        adapter_reduction: int = 16,
        adapter_dropout: float = 0.1,
        adapter_type: str = "simple",
        context_channels: int = 0,
        num_experts: int = 4,
        router_top_k: Optional[int] = 2,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.guided = adapter_type == "guided" and context_channels > 0
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm = nn.ModuleDict({
            d: nn.BatchNorm2d(out_ch) for d in domain_list
        })
        self.act = nn.ReLU(inplace=True)
        if self.guided:
            self.adapters = nn.ModuleDict({
                d: ContextGuidedAdapter(
                    out_ch, context_channels, num_experts=num_experts,
                    reduction=adapter_reduction, temperature=router_temperature,
                    top_k=router_top_k,
                )
                for d in domain_list
            })
        else:
            self.adapters = nn.ModuleDict({
                d: ResidualAdapter(out_ch, reduction=adapter_reduction, dropout=adapter_dropout)
                for d in domain_list
            })

    def forward(
        self, x: torch.Tensor, domain: str, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.act(self.norm[domain](self.conv(x)))
        if self.guided and context is not None:
            x, info = self.adapters[domain](x, context)
            return x, info["routing_weights"]
        return self.adapters[domain](x), None


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
        adapter_type: str = "simple",
        context_channels: int = 0,
        num_experts: int = 4,
        router_top_k: Optional[int] = 2,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=upsample_stride, stride=upsample_stride
        )
        merge_in = in_ch + skip_ch if skip_ch > 0 else in_ch
        block_kw = dict(
            adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
            adapter_type=adapter_type, context_channels=context_channels,
            num_experts=num_experts, router_top_k=router_top_k,
            router_temperature=router_temperature,
        )
        self.merge_conv = nn.Sequential(
            DomainConvBlock(merge_in, out_ch, domain_list, **block_kw),
            DomainConvBlock(out_ch, out_ch, domain_list, **block_kw),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain: str,
        skip: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.upconv(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Upsampled shape {x.shape[-2:]} != skip shape {skip.shape[-2:]}. "
                    "Use an input size divisible by 32."
                )
            x = torch.cat([x, skip], dim=1)
        routing: List[torch.Tensor] = []
        for block in self.merge_conv:
            x, r = block(x, domain, context)
            if r is not None:
                routing.append(r)
        return x, routing


class AuxDecoder(nn.Module):
    """Lightweight skip-free decoder for the auxiliary head.

    Lifts its input to full resolution with a stack of stride-2 transposed
    convolutions -- one per entry in ``widths`` -- so the auxiliary prediction
    is made at the same resolution as the main head instead of being a bilinear
    blur of a small map.  Deliberately has no skip connections and thin
    channels: it is meant to be a cheap exit, not a second decoder.

    Shared transposed convs + per-domain BatchNorm, matching DomainConvBlock's
    split, but without the residual adapters.
    """

    def __init__(
        self,
        in_ch: int,
        domain_list: Iterable[str],
        widths: Tuple[int, ...] = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = in_ch
        for w in widths:
            self.ups.append(
                nn.ConvTranspose2d(ch, w, kernel_size=2, stride=2, bias=False)
            )
            self.norms.append(
                nn.ModuleDict({d: nn.BatchNorm2d(w) for d in domain_list})
            )
            ch = w
        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        for up, norm in zip(self.ups, self.norms):
            x = self.act(norm[domain](up(x)))
        return self.classifier(x)


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
        adapter_type: str = "simple",
        num_experts: int = 4,
        router_top_k: Optional[int] = 2,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.domain_list = list(domain_list)
        self.fusion_type = fusion_type
        self.use_attention = use_attention
        self.adapter_type = adapter_type
        # Load-balancing loss over this decoder's guided routing, per forward.
        self.routing_balance: Optional[torch.Tensor] = None

        mult = 4 if fusion_type == "abs_prod" else 3
        fused_channels = {k: mult * STAGE_CHANNELS[k] for k in ("l1", "l2", "l3", "l4")}

        if use_attention:
            self.attention = CBAM(fused_channels["l4"])
        else:
            self.attention = nn.Identity()

        # Each decoder stage is conditioned on the change map of the pyramid
        # level it works at: |f1 - f2| has STAGE_CHANNELS[level] channels and
        # already matches that stage's spatial size.  Up-stage 4 runs at full
        # resolution, where no change map exists, so it keeps simple adapters.
        guided_kw = dict(
            adapter_type=adapter_type, num_experts=num_experts,
            router_top_k=router_top_k, router_temperature=router_temperature,
        )
        self.bottleneck = DomainConvBlock(
            fused_channels["l4"], 512, domain_list, kernel_size=1, padding=0,
            adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
            context_channels=STAGE_CHANNELS["l4"], **guided_kw,
        )

        self.up_stages = nn.ModuleList([
            UNetUpStage(512, fused_channels["l3"], 256, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
                        context_channels=STAGE_CHANNELS["l3"], **guided_kw),
            UNetUpStage(256, fused_channels["l2"], 128, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
                        context_channels=STAGE_CHANNELS["l2"], **guided_kw),
            UNetUpStage(128, fused_channels["l1"], 64, domain_list, upsample_stride=2,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout,
                        context_channels=STAGE_CHANNELS["l1"], **guided_kw),
            UNetUpStage(64, 0, 32, domain_list, upsample_stride=4,
                        adapter_reduction=adapter_reduction, adapter_dropout=adapter_dropout),
        ])

        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        # Auxiliary branch off the FIRST up-stage (256ch at H/16), with its own
        # cheap transposed-conv stack back to full resolution.  Four stride-2
        # steps take H/16 -> H, so the aux logits match the main head exactly
        # and no bilinear upsampling is needed.
        self.aux_decoder = AuxDecoder(256, domain_list, widths=(128, 64, 32, 16))

        prior_bias = math.log(prior / (1.0 - prior))
        for head in (self.classifier, self.aux_decoder.classifier):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, prior_bias)

    def forward(
        self, fused_skips: Dict[str, torch.Tensor], domain: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def change_map(level: str) -> torch.Tensor:
            """The |f1 - f2| block of a fused skip (channels 2C..3C)."""
            c = STAGE_CHANNELS[level]
            return fused_skips[level][:, 2 * c:3 * c]

        routing: List[torch.Tensor] = []
        fused_l4 = self.attention(fused_skips["l4"])
        x, r = self.bottleneck(fused_l4, domain, change_map("l4"))
        if r is not None:
            routing.append(r)

        x, r = self.up_stages[0](x, domain, fused_skips["l3"], change_map("l3"))
        routing.extend(r)
        aux = self.aux_decoder(x, domain)

        x, r = self.up_stages[1](x, domain, fused_skips["l2"], change_map("l2"))
        routing.extend(r)
        x, r = self.up_stages[2](x, domain, fused_skips["l1"], change_map("l1"))
        routing.extend(r)
        x, _ = self.up_stages[3](x, domain)

        self.routing_balance = (
            torch.stack([routing_balance_loss(w) for w in routing]).mean()
            if routing else None
        )
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
        decoder_adapter_type: str = "simple",
        num_experts: int = 4,
        router_top_k: Optional[int] = 2,
        router_temperature: float = 1.0,
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
            adapter_type=decoder_adapter_type,
            num_experts=num_experts,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
        )
        # Mean routing-balance loss over encoder and decoder guided adapters.
        self.routing_balance: Optional[torch.Tensor] = None

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
        # Both timesteps go through the encoder together: change-guided
        # adapters need x1 and x2 at the same point in the network.
        pyramid1, pyramid2 = self.backbone.extract_multiscale_pair(img1, img2, domain)
        fused = self._fuse_pyramid(pyramid1, pyramid2)

        logits, aux = self.decoder(fused, domain)

        parts = [
            t for t in (
                getattr(self.backbone, "routing_balance", None),
                getattr(self.decoder, "routing_balance", None),
            ) if t is not None
        ]
        self.routing_balance = torch.stack(parts).mean() if parts else None

        # ``aux`` is returned in eval mode too, so the auxiliary head can be
        # scored as a standalone predictor alongside the main head.  It is
        # suppressed only when deep supervision is switched off entirely.
        if not self.use_deep_supervision:
            aux = None
        elif aux is not None and aux.shape[-2:] != img1.shape[-2:]:
            # AuxDecoder already emits full resolution; this only fires if the
            # input size is not divisible by 32.
            aux = F.interpolate(
                aux, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )

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

