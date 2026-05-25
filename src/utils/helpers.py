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


def domain_parameters(model, domain: str) -> list:
    """Return parameters owned by a domain (adapters + main + aux decoder)."""
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

    aux_decoders = getattr(model, "aux_decoders", None)
    if aux_decoders is not None and domain in aux_decoders:
        params += list(aux_decoders[domain].parameters())
    return params


def freeze_domain(model, current_domain: str) -> None:
    """Enable gradients only for ``current_domain``'s adapters + decoders."""
    domains = getattr(model, "domain_list", None)
    if domains is None:
        backbone = getattr(model, "backbone", model)
        domains = getattr(backbone, "domain_list", [])

    for d in domains:
        flag = d == current_domain
        for p in domain_parameters(model, d):
            p.requires_grad = flag


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
