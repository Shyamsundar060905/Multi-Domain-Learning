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
    use_attention: bool = False,
    adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
) -> ChangeDetectionModel:
    """ResNet50 encoder + per-domain adapters + shared U-Net decoder (domain BN + adapters)."""
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


def print_architecture(mode: str = "multi") -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    print(f"{label} change detection:")
    print("  Encoder: frozen ImageNet ResNet50 + per-domain residual adapters (l1–l4)")
    print("  Decoder: shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain residual adapters)")
    print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  deep sup on 1st up-stage")
