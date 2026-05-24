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
    """Mask gradient flow so only ``current_domain``'s adapters update.

    NOTE: this does NOT touch the backbone -- the backbone is frozen at model
    construction time and must stay frozen.  It also leaves the segmentation
    head trainable.  Only the per-domain ``domain_adapters`` module is touched.
    """
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "domain_adapters"):
        return

    for domain_name, module in backbone.domain_adapters.items():
        flag = (domain_name == current_domain)
        for p in module.parameters():
            p.requires_grad = flag


def domain_parameters(model, domain: str) -> list:
    """Return the trainable parameters owned by a specific domain's adapters."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "domain_adapters") or domain not in backbone.domain_adapters:
        return []
    return list(backbone.domain_adapters[domain].parameters())


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
