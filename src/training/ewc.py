"""Elastic Weight Consolidation (EWC) for multi-domain change detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EWC:
    """Standard EWC penalty: ``lambda/2 * sum_i F_i * (theta_i - theta*_i)^2``."""

    def __init__(self, model, ewc_lambda: float = 1e4):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params: dict = {}
        self.fisher_information: dict = {}

    def _trainable_named_params(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def compute_fisher_information(self, dataloader, device, domain, max_batches: int = 50):
        """Diagonal Fisher estimate using the supervised CD loss gradients."""
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p) for n, p in self._trainable_named_params()}

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
            logits = self.model(img1, img2, domain)

            pos = mask.sum().clamp(min=1.0)
            neg = (mask.numel() - mask.sum()).clamp(min=1.0)
            pos_weight = (neg / pos).clamp(max=50.0)

            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            loss.backward()

            for n, p in self._trainable_named_params():
                if p.grad is not None:
                    fisher_info[n] += p.grad.detach() ** 2

            seen += 1
            if seen >= max_batches:
                break

        if seen > 0:
            for n in fisher_info:
                fisher_info[n] /= seen

        return fisher_info

    def remember_task(self, task_name, dataloader, device, *_, **__):
        fim = self.compute_fisher_information(dataloader, device, task_name)
        self.fisher_information[task_name] = fim
        self.params[task_name] = {
            n: p.detach().clone() for n, p in self._trainable_named_params()
        }

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.params:
            return loss
        for task_name in self.params:
            fim = self.fisher_information[task_name]
            stored = self.params[task_name]
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in stored:
                    continue
                loss = loss + (fim[name] * (param - stored[name]) ** 2).sum()
        return self.ewc_lambda / 2.0 * loss
