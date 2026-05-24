"""Shared utility helpers."""

from __future__ import annotations

import random
from typing import Iterable

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


def has_change(mask: torch.Tensor, threshold: float = 0.01) -> bool:
    return float(mask.sum().item()) / max(int(mask.numel()), 1) > threshold


def count_parameters(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")


def freeze_domain(model, current_domain: str) -> None:
    """Notebook-style domain isolation.

    Turn ``requires_grad`` on for ALL of ``current_domain``'s trainable
    surfaces (its backbone adapters + its full decoder) and off for every
    other domain.  The shared frozen backbone (stem, layer1..4 conv weights)
    is left alone -- those parameters were frozen at construction time and
    never become trainable.

    Designed to be called once at the top of each domain's training block in
    the outer loop, exactly as the notebook does.
    """
    backbone = getattr(model, "backbone", model)
    backbone_adapters = getattr(backbone, "domain_adapters", None)
    decoders = getattr(model, "decoders", None)

    if backbone_adapters is not None:
        for d, module in backbone_adapters.items():
            flag = (d == current_domain)
            for p in module.parameters():
                p.requires_grad = flag

    if decoders is not None:
        for d, module in decoders.items():
            flag = (d == current_domain)
            for p in module.parameters():
                p.requires_grad = flag


def domain_parameters(model, domain: str) -> list:
    """Return the trainable parameters owned by a specific domain.

    Includes the per-domain backbone adapters (from ``ResNetWithAdapters``)
    plus the per-domain decoder (from ``ChangeDetectionModel``).  Pass this
    list directly to a ``torch.optim`` constructor for notebook-style
    per-domain optimisers.
    """
    if hasattr(model, "domain_parameters"):
        return model.domain_parameters(domain)

    params: list = []
    backbone = getattr(model, "backbone", model)
    adapters = getattr(backbone, "domain_adapters", None)
    if adapters is not None and domain in adapters:
        params += list(adapters[domain].parameters())

    decoders = getattr(model, "decoders", None)
    if decoders is not None and domain in decoders:
        params += list(decoders[domain].parameters())
    return params


def to_python_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return int(x.item())
        return [to_python_int(i) for i in x]
    if isinstance(x, (list, tuple)):
        return [to_python_int(i) for i in x]
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Unsupported index type: {type(x)}") from e
