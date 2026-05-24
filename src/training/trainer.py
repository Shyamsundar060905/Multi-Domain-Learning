"""Continual / joint trainer for multi-domain binary change detection."""

from __future__ import annotations

import math
from itertools import cycle
from typing import Dict, Iterable, List, Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.ewc import EWC


PosWeightLike = Union[float, Mapping[str, float]]


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

def _make_round_robin(
    loaders: Dict[str, torch.utils.data.DataLoader],
    steps: int,
    domain_order: List[str],
):
    """Yield ``steps`` (domain, batch) tuples balanced equally across domains.

    Each domain contributes the same number of batches; smaller loaders are
    cycled so every domain reaches ``steps / num_domains`` batches per epoch.
    """
    iterators = {d: cycle(loaders[d]) for d in domain_order}
    for i in range(steps):
        d = domain_order[i % len(domain_order)]
        yield d, next(iterators[d])


def _make_sequential(
    loaders: Dict[str, torch.utils.data.DataLoader],
    batches_per_domain: int,
    domain_order: List[str],
):
    """Sequential schedule: yield ALL ``batches_per_domain`` batches of one
    domain before moving on to the next, in ``domain_order``.
    """
    for d in domain_order:
        it = cycle(loaders[d])
        for _ in range(batches_per_domain):
            yield d, next(it)


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
        pos_weight: PosWeightLike = 20.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        schedule: str = "round_robin",
        domain_order: Iterable[str] | None = None,
    ):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.domain_list = list(domain_list)
        self.device = device

        # ``pos_weight`` may be a single float (applied to every domain) or a
        # mapping ``{domain_name: weight}`` for per-domain class balancing.
        if isinstance(pos_weight, Mapping):
            self.pos_weight: Dict[str, float] = {
                d: float(pos_weight.get(d, 1.0)) for d in self.domain_list
            }
        else:
            self.pos_weight = {d: float(pos_weight) for d in self.domain_list}

        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        if schedule not in {"round_robin", "sequential"}:
            raise ValueError(f"schedule must be 'round_robin' or 'sequential', got {schedule!r}")
        self.schedule = schedule

        if domain_order is None:
            self.domain_order = self.domain_list
        else:
            self.domain_order = list(domain_order)
            unknown = set(self.domain_order) - set(self.train_loaders)
            if unknown:
                raise ValueError(f"domain_order contains unknown domains: {unknown}")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
        self._base_lr = lr
        self.scheduler = None  # built lazily in ``train_joint`` once total epochs is known

        self.ewc = EWC(model, ewc_lambda=ewc_lambda)

        # Freeze every BatchNorm in the model (running stats + affine).
        # Pretrained backbone BNs are already frozen by the backbone itself,
        # but per-domain adapter BNs, per-domain decoder BNs (DomainBN), and
        # the small decoder BNs are tiny -- leave them trainable.
        trainable_bn_keywords = (
            "domain_adapters",     # backbone per-domain adapters
            "decoder_adapters",    # decoder per-domain adapters
            "reduce", "conv1", "conv2",  # shared decoder convs + their per-domain BN
            "bns",                 # DomainBN.bns.<domain>
        )
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and not any(t in name for t in trainable_bn_keywords):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        print(f"Per-domain pos_weight: {self.pos_weight}")
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
            pos_weight=self.pos_weight[domain],
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

        # Equal-sample schedule: every domain contributes ``batches_per_domain``
        # batches per epoch.  The shuffled DataLoader cycles through the larger
        # WHU pool over multiple epochs; the smaller LEVIR pool is consumed
        # ~once per epoch.
        batches_per_domain = min(len(l) for l in self.train_loaders.values())
        steps_per_epoch = batches_per_domain * len(self.train_loaders)

        print(
            f"Schedule: {self.schedule}  |  order: {self.domain_order}  |  "
            f"batches/domain/epoch: {batches_per_domain}  |  total steps/epoch: {steps_per_epoch}"
        )

        # Closed-form warmup + cosine in a single LambdaLR.  Equivalent to
        # SequentialLR([LinearLR, CosineAnnealingLR]) but without the
        # `epoch=...` deprecation warning that SequentialLR triggers.
        warmup_epochs = max(1, min(5, epochs // 10))
        cosine_epochs = max(1, epochs - warmup_epochs)
        min_factor = 0.01

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * (epoch / max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / cosine_epochs
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_factor + (1.0 - min_factor) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        for epoch in range(epochs):
            self.model.train()

            if self.schedule == "round_robin":
                stream = _make_round_robin(self.train_loaders, steps_per_epoch, self.domain_order)
            else:
                stream = _make_sequential(self.train_loaders, batches_per_domain, self.domain_order)

            pbar = tqdm(
                stream,
                total=steps_per_epoch,
                desc=f"[{self.schedule}] Epoch {epoch+1}/{epochs}",
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
            current_lr = self.optimizer.param_groups[0]["lr"]
            for d, (lsum, dsum, n) in running.items():
                if n:
                    print(f"  [{d}] epoch {epoch+1}: loss={lsum/n:.4f}  dice={dsum/n:.4f}  lr={current_lr:.2e}")

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
