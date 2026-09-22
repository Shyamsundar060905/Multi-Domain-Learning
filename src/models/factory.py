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
    adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
    adapter_type: str = "guided",
    guided_granularity: str = "stage",
    num_experts: int = 4,
    guided_reduction: int = 16,
    router_top_k: Optional[int] = 2,
    decoder_adapter_type: str = "simple",
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
        decoder_adapter_type=decoder_adapter_type,
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
) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    if adapter_type == "guided":
        where = "every ResNet stage" if guided_granularity == "stage" else "every bottleneck block"
        routing = f"top-{router_top_k}" if router_top_k else "dense"
        encoder = (
            "frozen ImageNet ResNet50 + per-domain change-guided adapters "
            f"({num_experts} experts, {routing} routing) after {where}"
        )
    else:
        encoder = "frozen ImageNet ResNet50 + per-domain residual adapters after every bottleneck block"
    print(f"{label} change detection:")
    print(f"  Encoder: {encoder}")
    if decoder_adapter_type == "guided":
        decoder = ("shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain "
                   "change-guided adapters conditioned on |f1-f2| at each scale; up-stage 4 "
                   "keeps residual adapters)")
    else:
        decoder = "shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain residual adapters)"
    print(f"  Decoder: {decoder}")
    print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  aux decoder on 1st up-stage")
