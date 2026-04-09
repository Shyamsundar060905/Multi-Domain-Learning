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
