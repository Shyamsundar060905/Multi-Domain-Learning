"""Joint trainer for multi-domain binary change detection.

- Per-domain optimiser + scheduler (Adam state never crosses domains), plus one
  shared optimiser for the decoder's shared conv weights.
- ``freeze_domain`` called at the start of every domain block so the active
  domain is the only one with ``requires_grad=True``.
- Three schedule modes:
    * ``round_robin``: alternate domains every batch.
    * ``sequential``: a full block of the first domain then the next, inside one
      outer epoch (each block sized to the largest domain's loader).
    * ``per_domain_full_epoch``: full inner epoch of each domain per outer epoch.
"""

from __future__ import annotations

import math
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.helpers import domain_parameters, freeze_domain, shared_parameters


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

class MultiDomainTrainer:
    """Per-domain optimisers + schedulers for multi-domain CD."""

    def __init__(
        self,
        model,
        train_loaders,
        test_loaders,
        domain_list,
        device,
        eval_loaders=None,
        eval_domain_splits: Dict[str, str] | None = None,
        test_domain_splits: Dict[str, str] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: PosWeightLike = 20.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        deep_supervision_weight: float = 1.0,
        schedule: str = "per_domain_full_epoch",
        domain_order: Iterable[str] | None = None,
        scheduler_step_size: int = 15,
        scheduler_gamma: float = 0.1,
        use_tta: bool = False,
        ckpt_dir: str | None = "checkpoints",
        ckpt_name: str = "best.pt",
    ):
        self.model = model
        self.train_loaders = train_loaders
        self.eval_loaders = eval_loaders if eval_loaders is not None else test_loaders
        self.test_loaders = test_loaders
        self.eval_domain_splits = eval_domain_splits or {d: "test" for d in self.eval_loaders}
        self.test_domain_splits = test_domain_splits or {d: "test" for d in self.test_loaders}
        self.domain_list = list(domain_list)
        self.device = device
        self.use_tta = use_tta

        # Best-checkpoint tracking, keyed on the mean eval Dice across domains.
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_name = ckpt_name
        self.best_dice = -1.0
        self.best_epoch = -1
        if self.ckpt_dir is not None:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

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

        if domain_order is None:
            self.domain_order = list(self.domain_list)
        else:
            self.domain_order = list(domain_order)
            unknown = set(self.domain_order) - set(self.train_loaders)
            if unknown:
                raise ValueError(f"domain_order contains unknown domains: {unknown}")

        # Per-domain optimisers for encoder adapters; one shared decoder optimiser.
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

        decoder_params = shared_parameters(self.model)
        self.decoder_optimizer = None
        self.decoder_scheduler = None
        if decoder_params:
            self.decoder_optimizer = torch.optim.AdamW(
                decoder_params, lr=lr, weight_decay=weight_decay
            )
            self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(
                self.decoder_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
            )

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
            if isinstance(m, nn.BatchNorm2d) and "adapter" not in name and "domain_bn" not in name:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _log_trainable(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100.0 * trainable / total:.2f}%)")
        shared_n = sum(p.numel() for p in shared_parameters(self.model))
        if shared_n:
            print(f"  - shared decoder (trainable): {shared_n:,} params")
        else:
            print("  - shared decoder: frozen")
        for d in self.domain_list:
            n = sum(p.numel() for p in domain_parameters(self.model, d))
            print(f"  - {d} adapters: {n:,} params")

    # ------------------------------------------------------------------
    @property
    def ckpt_path(self) -> Path | None:
        return None if self.ckpt_dir is None else self.ckpt_dir / self.ckpt_name

    def _save_best(self, epoch: int, results: Dict[str, float]) -> bool:
        """Save the model when mean eval Dice improves. Returns True if saved."""
        path = self.ckpt_path
        if path is None or not results:
            return False

        avg = sum(results.values()) / len(results)
        if avg <= self.best_dice:
            return False

        prev = self.best_dice
        self.best_dice, self.best_epoch = avg, epoch

        payload = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "avg_eval_dice": avg,
            "per_domain_dice": dict(results),
            "domain_list": list(self.domain_list),
        }
        # Write to a temp file first: a run killed mid-save leaves the previous
        # best intact instead of a truncated checkpoint.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

        delta = "" if prev < 0 else f" (was {prev:.4f})"
        print(f"  * new best avg Dice {avg:.4f}{delta} -- saved {path}")
        return True

    def _load_best(self) -> bool:
        """Restore the best checkpoint in place. Returns True if one was loaded."""
        path = self.ckpt_path
        if path is None or not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.best_epoch = ckpt.get("epoch", self.best_epoch)
        self.best_dice = ckpt.get("avg_eval_dice", self.best_dice)
        print(
            f"Restored best checkpoint from epoch {self.best_epoch} "
            f"(avg eval Dice {self.best_dice:.4f})"
        )
        return True

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
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad(set_to_none=True)

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

        # Self-distillation: pull the auxiliary head toward the main head.
        # Both are pooled to the aux head's *native* resolution (the bottleneck
        # sits at H/32) -- aux_logits arrives already bilinearly upsampled to
        # full size, so comparing there would only penalise it for blur it
        # cannot represent.  The teacher is detached, so gradients move aux
        # toward main and never the reverse.
        distill_val = 0.0
        if aux_logits is not None:
            size = max(logits.shape[-1] // 32, 1)
            teacher = F.adaptive_avg_pool2d(logits.detach(), size)
            student = F.adaptive_avg_pool2d(aux_logits, size)
            distill = F.mse_loss(student, teacher)
            loss = loss + distill
            distill_val = float(distill.detach())

        loss.backward()
        trainable = domain_parameters(self.model, domain) + shared_parameters(self.model)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.step()

        with torch.no_grad():
            _, dice, _ = _segmentation_metrics(logits, mask)
        return loss.item(), dice.item(), mask.sum().item(), distill_val

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
            loss, dice, msum, dst = self.train_step(batch, domain)
            running_loss += loss
            running_dice += dice
            n += 1
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "dice": f"{dice:.4f}",
                "dst": f"{dst:.4f}",
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
            loss, dice, msum, dst = self.train_step(batch, domain)
            running[domain][0] += loss
            running[domain][1] += dice
            running[domain][2] += 1
            pbar.set_postfix({
                "dom": domain, "loss": f"{loss:.4f}",
                "dice": f"{dice:.4f}", "dst": f"{dst:.4f}",
                "msum": int(msum),
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
            batches_per_domain = max(len(l) for l in self.train_loaders.values())
            total = batches_per_domain * len(self.train_loaders)
            print(
                f"Batches/domain/epoch (max): {batches_per_domain}  |  total steps/epoch: {total}"
            )

        for epoch in range(epochs):
            self.model.train()

            if self.schedule == "per_domain_full_epoch":
                # Notebook style: full inner epoch per domain, in domain_order.
                for domain in self.domain_order:
                    self._train_domain_block(domain, self.train_loaders[domain], epoch, epochs)
            else:
                batches_per_domain = max(len(l) for l in self.train_loaders.values())
                total = batches_per_domain * len(self.train_loaders)
                if self.schedule == "round_robin":
                    stream = _round_robin(self.train_loaders, total, self.domain_order)
                else:  # sequential
                    stream = _sequential(self.train_loaders, batches_per_domain, self.domain_order)
                self._train_mixed(stream, total, epoch, epochs)

            for d in self.domain_list:
                self.schedulers[d].step()
            if self.decoder_scheduler is not None:
                self.decoder_scheduler.step()

            if self.eval_loaders:
                results = self.evaluate_all(
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    loaders=self.eval_loaders,
                    domain_splits=self.eval_domain_splits,
                )
                self._save_best(epoch + 1, results)
            # Also track LEVIR held-out test each epoch (val alone is misleading).
            if (
                self.test_loaders
                and "LEVIR" in self.test_loaders
                and self.test_loaders.get("LEVIR") is not self.eval_loaders.get("LEVIR")
            ):
                self.evaluate(
                    "LEVIR",
                    loaders=self.test_loaders,
                    split_label="test",
                    epoch=epoch + 1,
                    total_epochs=epochs,
                )

        # Final held-out test on the BEST checkpoint, not whatever the last
        # epoch happened to land on.
        if self.test_loaders:
            restored = self._load_best()
            header = (
                f"Final test (held-out) -- best checkpoint, epoch {self.best_epoch}"
                if restored
                else "Final test (held-out) -- last epoch, no checkpoint saved"
            )
            self.evaluate_all(
                loaders=self.test_loaders,
                domain_splits=self.test_domain_splits,
                header=header,
            )

    # ------------------------------------------------------------------
    def evaluate(
        self,
        domain: str,
        loaders: Dict | None = None,
        split_label: str = "test",
        epoch: int | None = None,
        total_epochs: int | None = None,
    ):
        loaders = self.test_loaders if loaders is None else loaders
        if domain not in loaders:
            return 0.0

        was_training = self.model.training
        self.model.eval()
        total_acc = total_dice = total_iou = 0.0
        n = 0
        desc = f"{domain} {split_label}"
        if epoch is not None and total_epochs is not None:
            desc = f"{domain} {split_label} (ep {epoch}/{total_epochs})"
        with torch.no_grad():
            for batch in tqdm(loaders[domain], desc=desc, leave=False):
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

                if self.use_tta:
                    # original prediction
                    logits, _ = self.model(img1, img2, domain)
                    prob = torch.sigmoid(logits)

                    # horizontal flip
                    img1_h = torch.flip(img1, dims=[-1])
                    img2_h = torch.flip(img2, dims=[-1])
                    logits_h, _ = self.model(img1_h, img2_h, domain)
                    prob_h = torch.flip(torch.sigmoid(logits_h), dims=[-1])

                    # vertical flip
                    img1_v = torch.flip(img1, dims=[-2])
                    img2_v = torch.flip(img2, dims=[-2])
                    logits_v, _ = self.model(img1_v, img2_v, domain)
                    prob_v = torch.flip(torch.sigmoid(logits_v), dims=[-2])

                    # average probabilities
                    avg_prob = (prob + prob_h + prob_v) / 3.0
                    eps = 1e-7
                    avg_prob = torch.clamp(avg_prob, eps, 1.0 - eps)
                    logits = torch.log(avg_prob / (1.0 - avg_prob))
                else:
                    logits, _ = self.model(img1, img2, domain)

                acc, dice, iou = _segmentation_metrics(logits, mask)
                total_acc += acc.item()
                total_dice += dice.item()
                total_iou += iou.item()
                n += 1

        if was_training:
            self.model.train()

        n = max(n, 1)
        avg_acc = 100.0 * total_acc / n
        avg_dice = total_dice / n
        avg_iou = total_iou / n
        if epoch is not None:
            print(
                f"  [{domain} {split_label}] dice={avg_dice:.4f}  "
                f"iou={avg_iou:.4f}  acc={avg_acc:.2f}%"
            )
        else:
            print(
                f"[{domain} {split_label}] acc={avg_acc:.2f}%  "
                f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
            )
        return avg_dice

    def evaluate_all(
        self,
        epoch: int | None = None,
        total_epochs: int | None = None,
        loaders: Dict | None = None,
        domain_splits: Dict[str, str] | None = None,
        header: str | None = None,
    ):
        loaders = self.test_loaders if loaders is None else loaders
        domain_splits = domain_splits or {d: "test" for d in loaders}

        if header:
            print(f"\n--- {header} ---")
        elif epoch is not None and total_epochs is not None:
            print(f"\n--- Eval after epoch {epoch}/{total_epochs} ---")
        else:
            print("\n--- Evaluating all domains ---")

        eval_order = [d for d in ("LEVIR", "WHU") if d in loaders]
        eval_order += [d for d in self.domain_list if d in loaders and d not in eval_order]

        results = {}
        for d in eval_order:
            results[d] = self.evaluate(
                d,
                loaders=loaders,
                split_label=domain_splits.get(d, "test"),
                epoch=epoch,
                total_epochs=total_epochs,
            )
        if results:
            avg = sum(results.values()) / len(results)
            if epoch is not None:
                print(f"  Avg eval Dice: {avg:.4f}")
            else:
                print(f"  Average Dice across domains: {avg:.4f}")
        return results
