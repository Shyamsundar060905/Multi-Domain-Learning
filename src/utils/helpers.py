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

def count_parameters(model):
    """Prints the total, trainable, and frozen parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

def freeze_domain(model, current_domain: str):
    """Unfreezes parameters specific to current_domain and freezes others."""
    for name, param in model.named_parameters():
        if f".{current_domain}." in name and "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def domain_parameters(model, domain: str):
    """Returns the parameters that are specific to the given domain."""
    # Note: assuming model is PrototypicalNetwork which wraps backbone
    backbone = getattr(model, 'backbone', model)
    return list(backbone.adapters[domain].parameters())


def validate_episode_config(n_way: int, k_shot: int, q_query: int, num_episodes: int) -> None:
    if n_way <= 1:
        raise ValueError("n_way must be > 1.")
    if k_shot <= 0:
        raise ValueError("k_shot must be > 0.")
    if q_query <= 0:
        raise ValueError("q_query must be > 0.")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0.")


def macro_f1_from_indices(targets: torch.Tensor, preds: torch.Tensor, n_classes: int) -> float:
    """
    Lightweight macro-F1 without sklearn dependency.
    """
    eps = 1e-12
    f1_scores = []
    for cls_idx in range(n_classes):
        tp = ((preds == cls_idx) & (targets == cls_idx)).sum().item()
        fp = ((preds == cls_idx) & (targets != cls_idx)).sum().item()
        fn = ((preds != cls_idx) & (targets == cls_idx)).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


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