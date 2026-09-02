"""Shared utility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")


def domain_parameters(model, domain: str) -> list:
    """Return parameters owned by a single domain (typically encoder adapters)."""
    if hasattr(model, "domain_parameters"):
        return model.domain_parameters(domain)

    params: list = []
    backbone = getattr(model, "backbone", model)
    adapters = getattr(backbone, "domain_adapters", None)
    if adapters is not None and domain in adapters:
        params += list(adapters[domain].parameters())

    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "domain_parameters"):
        params += dec.domain_parameters(domain)
    return params


def shared_parameters(model) -> list:
    """Return shared trainable parameters (e.g. common decoder)."""
    if hasattr(model, "shared_parameters"):
        return model.shared_parameters()

    dec = getattr(model, "decoder", None)
    if dec is not None and not hasattr(dec, "domain_parameters"):
        return list(dec.parameters())
    return []


def freeze_domain(model, current_domain: str) -> None:
    """Enable gradients for the active domain's adapters + any shared modules."""
    for p in model.parameters():
        p.requires_grad = False
    for p in domain_parameters(model, current_domain):
        p.requires_grad = True
    for p in shared_parameters(model):
        p.requires_grad = True
