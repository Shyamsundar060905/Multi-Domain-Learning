import torch
import random
import numpy as np

def set_seed(seed=42):
    """Locks random seeds across core mathematical modules for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def has_change(mask, threshold=0.01):
    return (mask.sum() / mask.numel()) > threshold

def count_parameters(model):
    """Prints the total, trainable, and frozen parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

def freeze_domain(model, current_domain: str):
    backbone = getattr(model, 'backbone', model)

    for name, param in backbone.named_parameters():
        # Train current domain adapters
        if f"adapters.{current_domain}" in name:
            param.requires_grad = True

        # Freeze other domain adapters
        elif "adapters" in name:
            param.requires_grad = False

        # Keep shared layers trainable (IMPORTANT)
        else:
            param.requires_grad = True
    

def domain_parameters(model, domain: str):
    """Returns the parameters that are specific to the given domain."""
    # Note: assuming model is PrototypicalNetwork which wraps backbone
    backbone = getattr(model, 'backbone', model)
    return list(backbone.adapters[domain].parameters())





def to_python_int(x):
    """
    Converts different index types to a pure Python int.
    Handles numpy, torch, lists, etc.
    """
    if isinstance(x, int):
        return x

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return int(x.item())
        else:
            return [to_python_int(i) for i in x]

    if isinstance(x, list) or isinstance(x, tuple):
        return [to_python_int(i) for i in x]

    # fallback (very important)
    try:
        return int(x)
    except Exception:
        raise TypeError(f"Unsupported index type: {type(x)}")