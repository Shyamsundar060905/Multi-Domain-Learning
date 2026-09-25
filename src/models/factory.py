"""Shared model factory.

Two model families, both built here so uni-, multi- and individual-domain runs
go through the same entry point (and therefore the same trainer, loss, metrics
and checkpoint selection):

- ``adapter``: frozen ResNet50 + per-domain adapters + shared decoder with
  per-domain BatchNorm and adapters.
- ``plain``:   frozen ResNet50 + one ordinary trainable decoder, no per-domain
  anything.  This is the "individual" baseline and is single-domain by
  construction.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torchvision.models import ResNet50_Weights, resnet50

from src.models.ChangeDetection import ChangeDetectionModel
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.simple_unet import SimpleChangeDetectionModel

MODEL_TYPES = ("adapter", "plain")


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
    use_deep_supervision: bool = True,
    model_type: str = "adapter",
):
    """Build the requested model family (see module docstring)."""
    domains: List[str] = list(domain_list)
    if not domains:
        raise ValueError("domain_list must contain at least one domain name.")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")

    if model_type == "plain":
        # The individual baseline has no per-domain parameters at all, so
        # training it on several domains would mean one set of weights driven by
        # several optimisers with independent momentum -- not the joint baseline
        # anyone means.  Refuse rather than silently do that.
        if len(domains) > 1:
            raise ValueError(
                "model_type='plain' is the individual (single-domain) baseline, but "
                f"{len(domains)} domains were given ({domains}). Run it once per "
                "domain, e.g. --domains LEVIR, then --domains WHU."
            )
        model = SimpleChangeDetectionModel(
            domain_list=domains,
            prior=prior,
            use_deep_supervision=use_deep_supervision,
            fusion_type=fusion_type,
            use_attention=use_attention,
        )
        if device is not None:
            model = model.to(device)
        return model

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
        decoder_adapter_reduction=guided_reduction if decoder_adapter_type == "guided" else 16,
        num_experts=num_experts,
        router_top_k=router_top_k,
    )
    if device is not None:
        model = model.to(device)
    return model


def print_architecture(
    mode: str = "multi",
    model_type: str = "adapter",
    adapter_type: str = "guided",
    guided_granularity: str = "stage",
    num_experts: int = 4,
    router_top_k: Optional[int] = 2,
    decoder_adapter_type: str = "simple",
) -> None:
    label = "Uni-domain" if mode == "uni" else "Multi-domain"
    if model_type == "plain":
        print("Individual (single-domain) change detection baseline:")
        print("  Encoder: frozen ImageNet ResNet50, no adapters")
        print("  Decoder: plain trainable U-Net (ordinary BatchNorm, no adapters)")
        print("  Fusion:  concat(f1, f2, |f1-f2|) at each scale  |  aux decoder on 1st up-stage")
        return
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
