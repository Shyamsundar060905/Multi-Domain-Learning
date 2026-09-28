"""Shared model factory — same architecture for uni- and multi-domain runs."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torchvision.models import ResNet50_Weights, resnet50

from src.models.ChangeDetection import ChangeDetectionModel
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
        decoder_adapter_reduction=guided_reduction if decoder_adapter_type == "guided" else 16,
        num_experts=num_experts,
        router_top_k=router_top_k,
    )
    if device is not None:
        model = model.to(device)
    return model


def print_architecture(
    mode: str = "multi",
    adapter_type: str = "guided",
    guided_granularity: str = "stage",
    num_experts: int = 4,
    router_top_k: Optional[int] = 2,
    decoder_adapter_type: str = "simple",
    decoder_adapter_stages: Optional[Iterable[str]] = None,
    adapter_stages: Iterable[str] = ("layer4",),
) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    stages = list(adapter_stages)
    if stages == ["layer4"]:
        stage_desc = "last stage only (layer4)"
    elif len(stages) == 4 and set(stages) == {"layer1", "layer2", "layer3", "layer4"}:
        stage_desc = "every ResNet stage" if guided_granularity == "stage" else "every bottleneck block"
    elif not stages:
        stage_desc = "no backbone stages (backbone fully frozen without adapters)"
    else:
        stage_desc = f"stages: {', '.join(stages)}"

    if not stages:
        encoder = "frozen ImageNet ResNet50 (no domain adapters in backbone)"
    elif adapter_type == "guided":
        routing = f"top-{router_top_k}" if router_top_k else "dense"
        encoder = (
            "frozen ImageNet ResNet50 + per-domain change-guided adapters "
            f"({num_experts} experts, {routing} routing) after {stage_desc}"
        )
    else:
        where = "every bottleneck block" if (guided_granularity == "block" and len(stages) > 1) else "stage output"
        encoder = f"frozen ImageNet ResNet50 + per-domain residual adapters after {stage_desc}"
    print(f"{label} change detection:")
    print(f"  Encoder: {encoder}")
    dec_stages = list(decoder_adapter_stages) if decoder_adapter_stages is not None else None
    if dec_stages is not None:
        dec_stage_desc = f"in {', '.join(dec_stages)} only" if len(dec_stages) < 5 else "across all stages"
    else:
        dec_stage_desc = "across all stages"

    if decoder_adapter_type == "none" or (dec_stages is not None and len(dec_stages) == 0):
        decoder = "shared trainable U-Net (shared convs + per-domain BatchNorm, no decoder adapters)"
    elif decoder_adapter_type == "guided":
        decoder = (f"shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain "
                   f"change-guided adapters [{dec_stage_desc}] conditioned on |f1-f2| at each scale; up-stage 4 "
                   f"keeps residual adapters)")
    else:
        decoder = (f"shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain "
                   f"residual adapters [{dec_stage_desc}])")
    print(f"  Decoder: {decoder}")
    print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  aux decoder on 1st up-stage")
