"""Shared model factory — same architecture for uni- and multi-domain runs.

Architectures selected via the ``backbone`` argument:
  - ``"unet"``        frozen ImageNet VGG-16-BN (all 13 convs) + U-Net decoder
                      (skip connections at every level)
  - ``"segnet"``      the same encoder + SegNet decoder (max-unpooling with the
                      encoder's pooling indices, no skip connections)
  - ``"unet_2conv"``  two-conv-per-stage U-Net encoder initialised from
                      VGG-16-BN (convs 20/30/40 unused) + U-Net decoder
  - ``"fcn"``         frozen ImageNet ResNet-50 with dilated layer4 + U-Net decoder
  - ``"vgg16"``       frozen ImageNet VGG-16 (no BN) + U-Net decoder
  - ``"resnet50"``    frozen ImageNet ResNet-50 + U-Net decoder
Every encoder carries per-domain adapters; every decoder has shared convs with
per-domain BatchNorm and adapters.
"""

from __future__ import annotations

from typing import Iterable, List, Literal

import torch
from torchvision.models import (
    ResNet50_Weights,
    VGG16_BN_Weights,
    VGG16_Weights,
    resnet50,
    vgg16,
    vgg16_bn,
)

from src.models.ChangeDetection import ChangeDetectionModel
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.fcn_adapter import FCNWithAdapters
from src.models.segnet_adapter import SegNetWithAdapters
from src.models.unet_adapter import UNetWithAdapters
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
    backbone: Literal["resnet50", "vgg16", "segnet", "unet", "unet_2conv", "fcn"] = "unet",
    # --- ResNet-only options ---
    domain_bn_in_adapter: bool = False,
    unfreeze_layer4: bool = False,
    adapter_stages: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
    # --- VGG / SegNet / U-Net / FCN options ---
    vgg_adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    segnet_adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    unet_adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    fcn_adapter_stages: Iterable[str] = ("l1", "l2", "l3", "l4"),
    unfreeze_backbone: bool = False,
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
        ``"unet"``, ``"segnet"``, ``"vgg16"``, ``"resnet50"``, or ``"fcn"``.
    adapter_stages:
        For ResNet-50: which stages get per-domain adapters (``"layer1"``–``"layer4"``).
    vgg_adapter_stages / segnet_adapter_stages / unet_adapter_stages / fcn_adapter_stages:
        Which pyramid levels get per-domain adapters (``"l1"``–``"l4"``).
    unfreeze_backbone:
        VGG-based encoders (vgg16, unet, segnet, unet_2conv): train every
        encoder weight as a parameter shared by all domains instead of freezing
        it.  Per-domain adapters are unchanged.
    adapter_reduction:
        Bottleneck reduction factor shared by both encoder and decoder adapters.
    adapter_dropout:
        Dropout rate inside adapters.
    """
    domains: List[str] = list(domain_list)
    if not domains:
        raise ValueError("domain_list must contain at least one domain name.")

    if unfreeze_backbone and backbone in ("resnet50", "fcn"):
        raise ValueError(
            "--unfreeze-backbone is for the VGG-based encoders (vgg16, unet, segnet, "
            f"unet_2conv); it is not supported for {backbone} on this branch"
            + (" (use --unfreeze-layer4)" if backbone == "resnet50" else "")
        )

    decoder = "unet"
    if backbone in ("unet", "segnet"):
        # Same frozen VGG-16-BN encoder for both; only the decoder differs.
        # (``unet`` is constructed exactly as the old ``segnet`` was, so it
        # reproduces those runs and loads their checkpoints.)
        base = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        enc = SegNetWithAdapters(
            base,
            domains,
            adapter_dropout=adapter_dropout,
            adapter_reduction=adapter_reduction,
            adapter_stages=list(unet_adapter_stages if backbone == "unet"
                                else segnet_adapter_stages),
            unfreeze_backbone=unfreeze_backbone,
        )
        decoder = backbone
    elif backbone == "unet_2conv":
        enc = UNetWithAdapters(
            domains,
            pretrained=True,
            unfreeze_backbone=unfreeze_backbone,
            adapter_dropout=adapter_dropout,
            adapter_reduction=adapter_reduction,
            adapter_stages=list(unet_adapter_stages),
        )
    elif backbone == "vgg16":
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        enc = VGGWithAdapters(
            base,
            domains,
            adapter_dropout=adapter_dropout,
            adapter_reduction=adapter_reduction,
            adapter_stages=list(vgg_adapter_stages),
            unfreeze_backbone=unfreeze_backbone,
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
    elif backbone == "fcn":
        base = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, False, True],
        )
        enc = FCNWithAdapters(
            base,
            domains,
            adapter_stages=list(fcn_adapter_stages),
            adapter_reduction=adapter_reduction,
            adapter_dropout=adapter_dropout,
        )
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose 'unet', 'segnet', 'unet_2conv', "
            "'vgg16', 'resnet50', or 'fcn'."
        )

    model = ChangeDetectionModel(
        enc,
        domain_list=domains,
        prior=prior,
        fusion_type=fusion_type,
        use_attention=use_attention,
        decoder_adapter_reduction=adapter_reduction,
        decoder_adapter_dropout=adapter_dropout,
        decoder=decoder,
    )
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def print_architecture(
    mode: str = "multi",
    backbone: str = "unet",
    unfreeze_backbone: bool = False,
) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    vgg_bn = "frozen ImageNet VGG-16-BN (all 13 convs) + per-domain residual adapters (l1-l4)"
    enc_desc = {
        "resnet50":   "frozen ImageNet ResNet-50 + per-domain residual adapters (layer1-layer4)",
        "vgg16":      "frozen ImageNet VGG-16   + per-domain residual adapters (l1-l4)",
        "segnet":     vgg_bn,
        "unet":       vgg_bn,
        "unet_2conv": "2-conv U-Net encoder from VGG-16-BN (convs 20/30/40 unused) "
                      "+ per-domain residual adapters (l1-l4)",
        "fcn":        "frozen ImageNet ResNet-50 with dilated layer4 (FCN-style) + per-domain residual adapters (l1-l4)",
    }.get(backbone, backbone)
    if unfreeze_backbone:
        bn_note = {
            "vgg16": "VGG-16 has no BatchNorm",
            "unet_2conv": "BatchNorm statistics updated during training",
        }.get(backbone, "BatchNorm statistics kept at ImageNet values")
        enc_desc = (enc_desc.replace("frozen ", "", 1)
                    + f"  [ENCODER TRAINABLE: all weights shared across domains; {bn_note}]")
    shared = "shared convs + per-domain BatchNorm + per-domain residual adapters"
    print(f"{label} change detection:")
    print(f"  Encoder: {enc_desc}")
    if backbone == "segnet":
        print(f"  Decoder: SegNet ({shared}); max-unpooling with the encoder's pooling "
              "indices (averaged over both dates), no skip connections")
        print("  Fusion:  concat(f1, f2, |f1-f2|) at the deepest level only  |  "
              "deep sup on 1st decoder stage")
    else:
        print(f"  Decoder: U-Net ({shared}); skip connections at every level")
        print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  deep sup on 1st up-stage")
