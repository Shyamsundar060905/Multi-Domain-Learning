"""Continual / joint trainer for multi-domain binary change detection."""

from __future__ import annotations

from itertools import cycle
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.ewc import EWC


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.  Stable when target is empty."""
    pred = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Pixel-wise focal BCE.  Down-weights easy background pixels."""
    bce = F.binary_cross_entropy_with_logits(
        logits, target,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    focal = (1.0 - p_t).pow(gamma) * bce
    return focal.mean()


def change_detection_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
    gamma: float = 2.0,
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
) -> torch.Tensor:
    """Combined focal-BCE + Dice loss tuned for severely-imbalanced CD."""
    return bce_weight * focal_bce_loss(logits, target, pos_weight, gamma) \
         + dice_weight * dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_round_robin(loaders: Dict[str, torch.utils.data.DataLoader], steps: int):
    """Yield ``steps`` (domain, batch) tuples balanced equally across domains.

    Each domain contributes the same number of batches; the smaller loader is
    cycled to keep parity with the larger one.
    """
    iterators = {d: cycle(loader) for d, loader in loaders.items()}
    domains = list(loaders.keys())
    for i in range(steps):
        d = domains[i % len(domains)]
        yield d, next(iterators[d])


def _segmentation_metrics(logits: torch.Tensor, target: torch.Tensor):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    target = target.float()

    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)

    dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (inter + 1e-6) / (pred_sum + target_sum - inter + 1e-6)
    acc = (pred == target).float().mean()
    return acc.mean(), dice.mean(), iou.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContinualFewShotTrainer:
    """Joint / continual trainer with EWC for multi-domain change detection."""

    def __init__(
        self,
        model,
        train_loaders,
        test_loaders,
        domain_list,
        device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ewc_lambda: float = 1e4,
        pos_weight: float = 20.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
    ):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = list(domain_list)
        self.device = device

        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, len(self.train_loaders)) * 10
        )

        self.ewc = EWC(model, ewc_lambda=ewc_lambda)

        # Freeze every BatchNorm in the model (running stats + affine).
        # Pretrained backbone BNs are already frozen by the backbone itself,
        # but per-domain adapter BNs are tiny -- we leave them trainable.
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and "domain_adapters" not in name and "head" not in name and "reduce" not in name:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        self._log_trainable()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_trainable(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100.0 * trainable / total:.2f}%)")
        print("Trainable submodules:")
        seen_prefixes = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            prefix = ".".join(name.split(".")[:3])
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                print(f"  - {prefix}")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def train_step(self, batch, domain: str):
        img1, img2, mask = batch
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True).float()

        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0) if mask.shape[0] == img1.shape[0] else mask.unsqueeze(1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        mask = (mask > 0.5).float()

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(img1, img2, domain)

        loss = change_detection_loss(
            logits, mask,
            pos_weight=self.pos_weight,
            gamma=self.focal_gamma,
            dice_weight=self.dice_weight,
            bce_weight=self.bce_weight,
        )
        ewc_loss = self.ewc.penalty(self.model)
        total_loss = loss + ewc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), max_norm=1.0
        )
        self.optimizer.step()

        with torch.no_grad():
            _, dice, _ = _segmentation_metrics(logits, mask)

        return loss.item(), float(ewc_loss), dice.item(), mask.sum().item()

    # ------------------------------------------------------------------
    # Joint training (balanced round-robin)
    # ------------------------------------------------------------------
    def train_joint(self, epochs: int, *_):
        """Train all domains jointly with an equal number of steps per domain."""
        if not self.train_loaders:
            print("No training loaders available.")
            return

        max_len = max(len(l) for l in self.train_loaders.values())
        steps_per_epoch = max_len * len(self.train_loaders)

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(
                _make_round_robin(self.train_loaders, steps_per_epoch),
                total=steps_per_epoch,
                desc=f"[Joint] Epoch {epoch+1}/{epochs}",
            )

            running = {d: [0.0, 0.0, 0] for d in self.domain_list}
            for domain, batch in pbar:
                loss, ewc_loss, dice, msum = self.train_step(batch, domain)
                running[domain][0] += loss
                running[domain][1] += dice
                running[domain][2] += 1

                pbar.set_postfix({
                    "dom": domain,
                    "loss": f"{loss:.4f}",
                    "dice": f"{dice:.4f}",
                    "msum": int(msum),
                })

            self.scheduler.step()
            for d, (lsum, dsum, n) in running.items():
                if n:
                    print(f"  [{d}] epoch {epoch+1}: loss={lsum/n:.4f}  dice={dsum/n:.4f}")

        print("\nConsolidating weights for all domains (EWC)...")
        for domain in self.domain_list:
            if domain in self.train_loaders:
                self.ewc.remember_task(domain, self.train_loaders[domain], self.device)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, domain: str, *_):
        if domain not in self.test_loaders:
            return 0.0

        self.model.eval()
        total_acc = total_dice = total_iou = 0.0
        n = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loaders[domain], desc=f"{domain} Eval", leave=False):
                img1, img2, mask = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                mask = mask.to(self.device).float()

                if img1.dim() == 3:
                    img1 = img1.unsqueeze(0)
                    img2 = img2.unsqueeze(0)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(0) if mask.shape[0] == img1.shape[0] else mask.unsqueeze(1)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                mask = (mask > 0.5).float()

                logits = self.model(img1, img2, domain)
                acc, dice, iou = _segmentation_metrics(logits, mask)
                total_acc += acc.item()
                total_dice += dice.item()
                total_iou += iou.item()
                n += 1

        n = max(n, 1)
        avg_acc = 100.0 * total_acc / n
        avg_dice = total_dice / n
        avg_iou = total_iou / n
        print(f"[{domain}] acc={avg_acc:.2f}%  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
        return avg_dice

    def evaluate_all(self, *_):
        print("\n--- Evaluating all domains ---")
        results = {}
        for d in self.domain_list:
            if d in self.test_loaders:
                results[d] = self.evaluate(d)
        if results:
            avg = sum(results.values()) / len(results)
            print(f"Average Dice across domains: {avg:.4f}")
        return results
