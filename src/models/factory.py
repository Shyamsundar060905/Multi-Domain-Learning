"""Shared model factory — same architecture for uni- and multi-domain runs.

Supports two encoder backbones selected via the ``backbone`` argument:
  - ``"resnet50"``  (default) — frozen ImageNet ResNet-50 + per-domain adapters
  - ``"vgg16"``               — frozen ImageNet VGG-16   + per-domain adapters

Both backbones expose the same interface so the decoder and training loop are
identical regardless of which one is chosen.
"""

from __future__ import annotations

from typing import Iterable, List, Literal

import torch
from torchvision.models import ResNet50_Weights, VGG16_Weights, resnet50, vgg16

from src.models.ChangeDetection import ChangeDetectionModel
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.vgg_adapter import VGGWithAdapters


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_change_detection_model(
    domain_list: Iterable[str],
    device: torch.device | None = None,
    prior: float = 0.02,
    fusion_type: str = "abs",
    use_attention: bool = False,
    # --- backbone selection ---
    backbone: Literal["resnet50", "vgg16"] = "resnet50",
    # --- ResNet-only options (ignored for VGG) ---
    domain_bn_in_adapter: bool = False,
    unfreeze_layer4: bool = False,
    adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
    # --- VGG-only options (ignored for ResNet) ---
    vgg_adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    # --- shared adapter hyper-params ---
    adapter_reduction: int = 16,
    adapter_dropout: float = 0.1,
) -> ChangeDetectionModel:
    """Build a bi-temporal change-detection model with the requested backbone.

    Parameters
    ----------
    domain_list:
        Names of domains to train on jointly (e.g. ``["LEVIR", "WHU"]``).
    backbone:
        ``"resnet50"`` (default) or ``"vgg16"``.
    adapter_stages:
        For ResNet-50: which stages get per-domain adapters
        (``"layer1"``–``"layer4"``).  Ignored when ``backbone="vgg16"``.
    vgg_adapter_stages:
        For VGG-16: which pyramid levels get per-domain adapters
        (``"l1"``–``"l4"``).  Ignored when ``backbone="resnet50"``.
    adapter_reduction:
        Bottleneck reduction factor shared by both encoder and decoder adapters.
    adapter_dropout:
        Dropout rate inside adapters.
    """
    domains: List[str] = list(domain_list)
    if not domains:
        raise ValueError("domain_list must contain at least one domain name.")

    if backbone == "vgg16":
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        enc = VGGWithAdapters(
            base,
            domains,
            adapter_dropout=adapter_dropout,
            adapter_reduction=adapter_reduction,
            adapter_stages=list(vgg_adapter_stages),
        )
    elif backbone == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        enc = ResNetWithAdapters(
            base,
            domains,
            domain_bn_in_adapter=domain_bn_in_adapter,
            unfreeze_layer4=unfreeze_layer4,
            adapter_stages=list(adapter_stages),
            adapter_reduction=adapter_reduction,
            adapter_dropout=adapter_dropout,
        )
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose 'resnet50' or 'vgg16'."
        )

    model = ChangeDetectionModel(
        enc,
        domain_list=domains,
        prior=prior,
        fusion_type=fusion_type,
        use_attention=use_attention,
        decoder_adapter_reduction=adapter_reduction,
        decoder_adapter_dropout=adapter_dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def print_architecture(
    mode: str = "multi",
    backbone: str = "resnet50",
) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    enc_desc = {
        "resnet50": "frozen ImageNet ResNet-50 + per-domain residual adapters (layer1-layer4)",
        "vgg16":    "frozen ImageNet VGG-16   + per-domain residual adapters (l1-l4)",
    }.get(backbone, backbone)
    print(f"{label} change detection:")
    print(f"  Encoder: {enc_desc}")
    print("  Decoder: shared trainable U-Net (shared convs + per-domain BatchNorm + per-domain residual adapters)")
    print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  deep sup on 1st up-stage")
