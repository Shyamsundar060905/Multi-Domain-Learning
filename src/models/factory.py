"""Shared model factory — same architecture for uni- and multi-domain runs."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torchvision.models import ResNet50_Weights, resnet50

from src.models.ChangeDetection import (
    DECODER_STAGES, ChangeDetectionModel, normalize_decoder_stage,
)
from src.models.adapter_resnet import ResNetWithAdapters


def build_change_detection_model(
    domain_list: Iterable[str],
    device: torch.device | None = None,
    prior: float = 0.02,
    fusion_type: str = "abs",
    domain_bn_in_adapter: bool = False,
    unfreeze_layer4: bool = False,
    use_attention: bool = False,
    adapter_stages: Iterable[str] = ("layer4",),
    adapter_type: str = "guided",
    guided_granularity: str = "stage",
    num_experts: int = 4,
    guided_reduction: int = 16,
    router_top_k: Optional[int] = 2,
    decoder_adapter_type: str = "simple",
    decoder_adapter_stages: Optional[Iterable[str]] = None,
    use_deep_supervision: bool = True,
    simple_granularity: str = "block",
    decoder_adapter_granularity: str = "block",
    aux_adapters: bool = True,
) -> ChangeDetectionModel:
    """ResNet50 encoder + per-domain adapters (simple or change-guided) + shared U-Net decoder."""
    domains: List[str] = list(domain_list)
    if not domains:
        raise ValueError("domain_list must contain at least one domain name.")

    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone = ResNetWithAdapters(
        base,
        domains,
        domain_bn_in_adapter=domain_bn_in_adapter,
        unfreeze_layer4=unfreeze_layer4,
        adapter_stages=adapter_stages,
        adapter_type=adapter_type,
        guided_granularity=guided_granularity,
        num_experts=num_experts,
        guided_reduction=guided_reduction,
        router_top_k=router_top_k,
        simple_granularity=simple_granularity,
    )
    model = ChangeDetectionModel(
        backbone,
        domain_list=domains,
        prior=prior,
        fusion_type=fusion_type,
        use_attention=use_attention,
        use_deep_supervision=use_deep_supervision,
        decoder_adapter_type=decoder_adapter_type,
        decoder_adapter_stages=decoder_adapter_stages,
        decoder_adapter_granularity=decoder_adapter_granularity,
        aux_adapters=aux_adapters,
        decoder_adapter_reduction=guided_reduction if decoder_adapter_type == "guided" else 16,
        num_experts=num_experts,
        router_top_k=router_top_k,
    )
    if device is not None:
        model = model.to(device)
    return model


# Bottleneck blocks per ResNet50 stage (used to report adapter counts).
_RESNET50_BLOCKS = {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}
# AuxDecoder widths are fixed at (128, 64, 32, 16): one adapter per step.
_AUX_STEPS = 4


def print_architecture(
    mode: str = "multi",
    adapter_type: str = "guided",
    guided_granularity: str = "stage",
    num_experts: int = 4,
    router_top_k: Optional[int] = 2,
    decoder_adapter_type: str = "simple",
    decoder_adapter_stages: Optional[Iterable[str]] = None,
    adapter_stages: Iterable[str] = ("layer4",),
    simple_granularity: str = "block",
    decoder_adapter_granularity: str = "block",
    aux_adapters: bool = True,
    use_deep_supervision: bool = True,
) -> None:
    """Describe the model that was built, with exact per-domain adapter counts."""
    label = "Uni-domain" if mode == "uni" else "Multi-domain"

    # ---- encoder
    stages = list(adapter_stages)
    per_block = (simple_granularity if adapter_type == "simple" else guided_granularity) == "block"
    n_enc = sum(_RESNET50_BLOCKS[s] if per_block else 1 for s in stages)
    stage_list = ("every ResNet stage" if sorted(stages) == sorted(_RESNET50_BLOCKS)
                  else ", ".join(stages))
    placement = (f"after every bottleneck block of {stage_list}" if per_block
                 else f"once at the output of {stage_list}")
    if not stages:
        encoder = "frozen ImageNet ResNet50 (no domain adapters in backbone)"
    elif adapter_type == "guided":
        routing = f"top-{router_top_k}" if router_top_k else "dense"
        encoder = ("frozen ImageNet ResNet50 + per-domain change-guided adapters "
                   f"({num_experts} experts, {routing} routing) {placement}")
    else:
        encoder = f"frozen ImageNet ResNet50 + per-domain residual adapters {placement}"

    # ---- decoder
    if decoder_adapter_stages is None:
        dec_stages = list(DECODER_STAGES)
    else:
        dec_stages = [normalize_decoder_stage(s) for s in decoder_adapter_stages]
    if decoder_adapter_type == "none":
        dec_stages = []
    per_up_stage = 2 if decoder_adapter_granularity == "block" else 1
    n_dec = sum(1 if s == "bottleneck" else per_up_stage for s in set(dec_stages))
    if not dec_stages:
        decoder = "shared trainable U-Net (shared convs + per-domain BatchNorm, no decoder adapters)"
    else:
        where = "all stages" if set(dec_stages) == set(DECODER_STAGES) else ", ".join(dec_stages)
        gran = ("one per conv block" if decoder_adapter_granularity == "block"
                else "one per stage")
        if decoder_adapter_type == "guided":
            kind = "change-guided adapters conditioned on |f1-f2|"
            if "up4" in dec_stages:
                kind += " (up-stage 4 keeps residual adapters)"
        else:
            kind = "residual adapters"
        decoder = (f"shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain "
                   f"{kind} in {where}, {gran})")

    # ---- aux head
    if not use_deep_supervision:
        aux = "no aux head"
        n_aux = 0
    elif aux_adapters:
        aux = f"aux decoder on 1st up-stage with per-domain residual adapters"
        n_aux = _AUX_STEPS
    else:
        aux = "aux decoder on 1st up-stage without adapters (per-domain BatchNorm only)"
        n_aux = 0

    print(f"{label} change detection:")
    print(f"  Encoder: {encoder}")
    print(f"  Decoder: {decoder}")
    print(f"  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  {aux}")
    print(f"  Adapters per domain: encoder {n_enc} + decoder {n_dec} + aux {n_aux} "
          f"= {n_enc + n_dec + n_aux}")
