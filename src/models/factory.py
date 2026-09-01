"""Shared model factory — same architecture for uni- and multi-domain runs."""

from __future__ import annotations

from typing import Iterable, List

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
    use_attention: bool = True,
    adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
) -> ChangeDetectionModel:
    """ResNet50 encoder + per-domain adapters + CBAM, shared U-Net decoder (domain BN + adapters + CBAM)."""
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
        use_attention=use_attention,
    )
    model = ChangeDetectionModel(
        backbone,
        domain_list=domains,
        prior=prior,
        fusion_type=fusion_type,
        use_attention=use_attention,
    )
    if device is not None:
        model = model.to(device)
    return model


def print_architecture(mode: str = "multi", adapter_stages: Iterable[str] = (), use_attention: bool = True) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    stages = list(adapter_stages)
    encoder_desc = (
        f"frozen ImageNet ResNet50 + per-domain residual adapters ({', '.join(stages)})"
        if stages else
        "frozen ImageNet ResNet50, fully shared (no per-domain adapters)"
    )
    if use_attention:
        encoder_desc += " + per-domain CBAM (l4)"
    decoder_desc = "shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain residual adapters"
    decoder_desc += " + shared CBAM (fused l4))" if use_attention else ")"
    print(f"{label} change detection:")
    print(f"  Encoder: {encoder_desc}")
    print(f"  Decoder: {decoder_desc}")
    print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  deep sup on 1st up-stage")
