"""Continual / joint trainer for multi-domain binary change detection.

Notebook-style architecture:
- Per-domain optimiser + scheduler (Adam state never crosses domains).
- ``freeze_domain`` called at the start of every domain block so the active
  domain is the only one with ``requires_grad=True``.
- Three schedule modes:
    * ``round_robin``: alternate domains every batch (balanced shared-state
      experiment; with fully per-domain decoders there is no shared state, so
      this becomes equivalent to a fine-grained interleaving).
    * ``sequential``: ``min_len`` batches of the first domain then ``min_len``
      batches of the next, inside one outer epoch.
    * ``per_domain_full_epoch``: full inner epoch of each domain per outer
      epoch (mirrors the notebook's training loop).
"""

from __future__ import annotations

import math
from itertools import cycle
from typing import Dict, Iterable, List, Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.ewc import EWC
from src.utils.helpers import domain_parameters, freeze_domain


PosWeightLike = Union[float, Mapping[str, float]]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + smooth) / (denom + smooth)).mean()


def focal_bce_loss(
    logits: torch.Tensor, target: torch.Tensor,
    pos_weight: float = 10.0, gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, target,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    return ((1.0 - p_t).pow(gamma) * bce).mean()


def change_detection_loss(
    logits: torch.Tensor, target: torch.Tensor,
    pos_weight: float = 10.0, gamma: float = 2.0,
    dice_weight: float = 0.7, bce_weight: float = 0.3,
) -> torch.Tensor:
    return bce_weight * focal_bce_loss(logits, target, pos_weight, gamma) \
         + dice_weight * dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _round_robin(loaders: Dict, steps: int, order: List[str]):
    iters = {d: cycle(loaders[d]) for d in order}
    for i in range(steps):
        d = order[i % len(order)]
        yield d, next(iters[d])


def _sequential(loaders: Dict, batches_per_domain: int, order: List[str]):
    for d in order:
        it = cycle(loaders[d])
        for _ in range(batches_per_domain):
            yield d, next(it)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    """Per-domain optimisers + schedulers for multi-domain CD."""

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
        pos_weight: PosWeightLike = 20.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        deep_supervision_weight: float = 0.4,
        schedule: str = "per_domain_full_epoch",
        domain_order: Iterable[str] | None = None,
        scheduler_step_size: int = 15,
        scheduler_gamma: float = 0.1,
        skip_ewc: bool = False,
    ):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = list(domain_list)
        self.device = device

        if isinstance(pos_weight, Mapping):
            self.pos_weight: Dict[str, float] = {
                d: float(pos_weight.get(d, 1.0)) for d in self.domain_list
            }
        else:
            self.pos_weight = {d: float(pos_weight) for d in self.domain_list}

        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.deep_supervision_weight = deep_supervision_weight

        if schedule not in {"round_robin", "sequential", "per_domain_full_epoch"}:
            raise ValueError(
                "schedule must be one of round_robin, sequential, per_domain_full_epoch"
            )
        self.schedule = schedule
        self.skip_ewc = skip_ewc

        if domain_order is None:
            self.domain_order = list(self.domain_list)
        else:
            self.domain_order = list(domain_order)
            unknown = set(self.domain_order) - set(self.train_loaders)
            if unknown:
                raise ValueError(f"domain_order contains unknown domains: {unknown}")

        # ---------------- Per-domain optimisers + StepLR schedulers ----------
        # Mirrors the notebook: one optimiser holding only that domain's
        # trainable parameters (its backbone adapters + its full decoder).
        # Adam state is therefore never contaminated across domains.
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        for d in self.domain_list:
            params = domain_parameters(self.model, d)
            self.optimizers[d] = torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
            self.schedulers[d] = torch.optim.lr_scheduler.StepLR(
                self.optimizers[d], step_size=scheduler_step_size, gamma=scheduler_gamma
            )

        self.ewc = EWC(model, ewc_lambda=ewc_lambda)

        # Keep all backbone BatchNorms in eval mode (they are frozen).
        self._lock_backbone_bn()

        print(f"Per-domain pos_weight: {self.pos_weight}")
        print(f"Deep supervision weight: {self.deep_supervision_weight}")
        print(f"Domain order:          {self.domain_order}")
        print(f"Schedule:              {self.schedule}")
        self._log_trainable()

    # ------------------------------------------------------------------
    def _lock_backbone_bn(self) -> None:
        """Freeze all BatchNorms inside the backbone (decoder BNs stay trainable)."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return
        for name, m in backbone.named_modules():
            if isinstance(m, nn.BatchNorm2d) and "domain_adapters" not in name:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _log_trainable(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100.0 * trainable / total:.2f}%)")
        for d in self.domain_list:
            n = sum(p.numel() for p in domain_parameters(self.model, d))
            print(f"  - {d} owns {n:,} params")

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

        opt = self.optimizers[domain]
        opt.zero_grad(set_to_none=True)

        logits, aux_logits = self.model(img1, img2, domain)
        loss = change_detection_loss(
            logits, mask,
            pos_weight=self.pos_weight[domain],
            gamma=self.focal_gamma,
            dice_weight=self.dice_weight,
            bce_weight=self.bce_weight,
        )
        if aux_logits is not None:
            aux_loss = change_detection_loss(
                aux_logits, mask,
                pos_weight=self.pos_weight[domain],
                gamma=self.focal_gamma,
                dice_weight=self.dice_weight,
                bce_weight=self.bce_weight,
            )
            loss = loss + self.deep_supervision_weight * aux_loss
        ewc_loss = self.ewc.penalty(self.model)
        total = loss + ewc_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(domain_parameters(self.model, domain), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            _, dice, _ = _segmentation_metrics(logits, mask)
        return loss.item(), float(ewc_loss), dice.item(), mask.sum().item()

    # ------------------------------------------------------------------
    def _train_domain_block(self, domain: str, loader, epoch: int, epochs: int):
        """Run one full domain block (a contiguous run of batches on a single
        domain).  Pre-flips ``requires_grad`` via ``freeze_domain`` exactly as
        the notebook does.
        """
        freeze_domain(self.model, domain)

        running_loss = running_dice = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"[{domain} | epoch {epoch+1}/{epochs}]", leave=False)
        for batch in pbar:
            loss, ewc_loss, dice, msum = self.train_step(batch, domain)
            running_loss += loss
            running_dice += dice
            n += 1
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "dice": f"{dice:.4f}",
                "ewc":  f"{ewc_loss:.4f}",
                "msum": int(msum),
            })

        n = max(n, 1)
        lr = self.optimizers[domain].param_groups[0]["lr"]
        print(f"  [{domain}] epoch {epoch+1}: loss={running_loss/n:.4f}  "
              f"dice={running_dice/n:.4f}  lr={lr:.2e}")
        return running_loss / n, running_dice / n

    def _train_mixed(self, stream, total_steps: int, epoch: int, epochs: int):
        """Used for round_robin / sequential: gradients flow only through the
        domain selected per-batch.  ``freeze_domain`` is called inside the
        loop on every domain switch.
        """
        running = {d: [0.0, 0.0, 0] for d in self.domain_list}
        prev = None
        pbar = tqdm(stream, total=total_steps, desc=f"[{self.schedule} | {epoch+1}/{epochs}]")
        for domain, batch in pbar:
            if domain != prev:
                freeze_domain(self.model, domain)
                prev = domain
            loss, ewc_loss, dice, msum = self.train_step(batch, domain)
            running[domain][0] += loss
            running[domain][1] += dice
            running[domain][2] += 1
            pbar.set_postfix({
                "dom": domain, "loss": f"{loss:.4f}",
                "dice": f"{dice:.4f}", "msum": int(msum),
            })
        for d, (lsum, dsum, n) in running.items():
            if n:
                lr = self.optimizers[d].param_groups[0]["lr"]
                print(f"  [{d}] epoch {epoch+1}: loss={lsum/n:.4f}  "
                      f"dice={dsum/n:.4f}  lr={lr:.2e}")

    # ------------------------------------------------------------------
    def train_joint(self, epochs: int, *_):
        if not self.train_loaders:
            print("No training loaders available.")
            return

        if self.schedule == "per_domain_full_epoch":
            print("Schedule: outer-epoch loops over domains; each gets a full inner pass.")
        else:
            batches_per_domain = min(len(l) for l in self.train_loaders.values())
            total = batches_per_domain * len(self.train_loaders)
            print(
                f"Batches/domain/epoch: {batches_per_domain}  |  total steps/epoch: {total}"
            )

        for epoch in range(epochs):
            self.model.train()

            if self.schedule == "per_domain_full_epoch":
                # Notebook style: full inner epoch per domain, in domain_order.
                for domain in self.domain_order:
                    self._train_domain_block(domain, self.train_loaders[domain], epoch, epochs)
            else:
                batches_per_domain = min(len(l) for l in self.train_loaders.values())
                total = batches_per_domain * len(self.train_loaders)
                if self.schedule == "round_robin":
                    stream = _round_robin(self.train_loaders, total, self.domain_order)
                else:  # sequential
                    stream = _sequential(self.train_loaders, batches_per_domain, self.domain_order)
                self._train_mixed(stream, total, epoch, epochs)

            for d in self.domain_list:
                self.schedulers[d].step()

        if self.skip_ewc:
            print("\nSkipping EWC consolidation (--skip-ewc).")
        else:
            print("\nConsolidating weights for all domains (EWC)...")
            try:
                for domain in self.domain_list:
                    if domain in self.train_loaders:
                        self.ewc.remember_task(domain, self.train_loaders[domain], self.device)
                print("EWC consolidation complete.")
            except Exception as exc:
                print(f"[Warning] EWC consolidation failed: {exc}")
                print("Training weights are kept; continuing to evaluation.")

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

                logits, _ = self.model(img1, img2, domain)
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
