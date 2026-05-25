"""Elastic Weight Consolidation (EWC) for multi-domain change detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.helpers import domain_parameters, freeze_domain


class EWC:
    """Standard EWC penalty: ``lambda/2 * sum_i F_i * (theta_i - theta*_i)^2``."""

    def __init__(self, model, ewc_lambda: float = 1e4):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params: dict = {}
        self.fisher_information: dict = {}

    def _domain_named_params(self, domain: str):
        owned = {id(p) for p in domain_parameters(self.model, domain)}
        return [(n, p) for n, p in self.model.named_parameters() if id(p) in owned]

    def compute_fisher_information(self, dataloader, device, domain, max_batches: int = 50):
        """Diagonal Fisher estimate using the supervised CD loss gradients."""
        freeze_domain(self.model, domain)
        for _, p in self._domain_named_params(domain):
            p.requires_grad_(True)

        named_params = self._domain_named_params(domain)
        if not named_params:
            raise RuntimeError(f"No trainable parameters found for domain '{domain}'.")

        # Use train mode so adapter BN/dropout behave like training; aux head stays
        # off because forward checks self.training only for deep supervision.
        self.model.train()
        fisher_info = {n: torch.zeros_like(p) for n, p in named_params}

        seen = 0
        for batch in dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                continue
            img1, img2, mask = batch

            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device).float()
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) if mask.shape[0] == img1.shape[0] else mask.unsqueeze(0)
            mask = (mask > 0.5).float()

            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                out = self.model(img1, img2, domain)
                logits = out[0] if isinstance(out, tuple) else out

                if not logits.requires_grad:
                    raise RuntimeError(
                        f"EWC Fisher pass for '{domain}' produced logits without grad. "
                        "Check freeze_domain / domain_parameters."
                    )

                pos = mask.sum().clamp(min=1.0)
                neg = (mask.numel() - mask.sum()).clamp(min=1.0)
                pos_weight = (neg / pos).clamp(max=50.0)
                loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)

                param_list = [p for _, p in named_params]
                grads = torch.autograd.grad(
                    loss, param_list, retain_graph=False, allow_unused=True
                )

            for (n, p), g in zip(named_params, grads):
                if g is not None:
                    fisher_info[n] += g.detach() ** 2

            seen += 1
            if seen >= max_batches:
                break

        if seen == 0:
            raise RuntimeError(f"No valid batches for EWC Fisher on domain '{domain}'.")

        for n in fisher_info:
            fisher_info[n] /= seen

        return fisher_info

    def remember_task(self, task_name, dataloader, device, *_, **__):
        fim = self.compute_fisher_information(dataloader, device, task_name)
        self.fisher_information[task_name] = fim
        self.params[task_name] = {
            n: p.detach().clone() for n, p in self._domain_named_params(task_name)
        }

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.params:
            return loss
        for task_name in self.params:
            fim = self.fisher_information[task_name]
            stored = self.params[task_name]
            for name, param in model.named_parameters():
                if name not in stored:
                    continue
                loss = loss + (fim[name] * (param - stored[name]) ** 2).sum()
        return self.ewc_lambda / 2.0 * loss
