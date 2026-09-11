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

class _LossAccum:
    """Running means of each loss term over an epoch (or a domain block)."""

    KEYS = ("total", "main", "aux", "distill", "ewc", "dice", "aux_dice")

    def __init__(self):
        self.sums = {k: 0.0 for k in self.KEYS}
        self.n = 0

    def add(self, stats: Dict[str, float]) -> None:
        for k in self.KEYS:
            self.sums[k] += stats.get(k, 0.0)
        self.n += 1

    def mean(self, key: str) -> float:
        return self.sums[key] / max(self.n, 1)

    def summary(self, trainer) -> str:
        """One line showing every term, weighted as it enters the objective."""
        w, a = trainer.deep_supervision_weight, trainer.distill_alpha
        parts = [
            f"loss={self.mean('total'):.4f}",
            f"(main={self.mean('main'):.4f}",
            f"aux={self.mean('aux'):.4f}x{w:g}",
        ]
        if a > 0.0:
            parts.append(f"dst={self.mean('distill'):.4f}x{a:g}")
        e = trainer.ewc_lambda
        if e > 0.0:
            # Raw Fisher-weighted penalty and the lambda/2 it is scaled by.
            parts.append(f"ewc={self.mean('ewc'):.3e}x{e / 2:g}")
        parts[-1] += ")"
        parts.append(f"dice={self.mean('dice'):.4f}")
        if self.sums["aux_dice"] > 0.0:
            parts.append(f"aux_dice={self.mean('aux_dice'):.4f}")
        return "  ".join(parts)


def _tta_logits(model, img1: torch.Tensor, img2: torch.Tensor, domain: str):
    """Average identity / hflip / vflip predictions, for both heads.

    Averaging is done in probability space and converted back to logits so the
    caller can keep using the same metric function.
    """
    eps = 1e-7
    views = [
        (lambda t: t, lambda t: t),                                   # identity
        (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
        (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
    ]

    main_sum, aux_sum, k = None, None, 0
    for fwd, inv in views:
        lo, ax = model(fwd(img1), fwd(img2), domain)
        p = inv(torch.sigmoid(lo))
        main_sum = p if main_sum is None else main_sum + p
        if ax is not None:
            pa = inv(torch.sigmoid(ax))
            aux_sum = pa if aux_sum is None else aux_sum + pa
        k += 1

    def _to_logits(prob_sum):
        if prob_sum is None:
            return None
        p = torch.clamp(prob_sum / k, eps, 1.0 - eps)
        return torch.log(p / (1.0 - p))

    return _to_logits(main_sum), _to_logits(aux_sum)


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
        distill_alpha: float = 0.0,
        ewc_lambda: float = 0.0,
        ewc_fisher_batches: int = 50,
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
        self.distill_alpha = distill_alpha
        # EWC over the shared decoder weights: a diagonal Fisher and an anchor
        # snapshot per domain, refreshed by _ewc_consolidate.
        self.ewc_lambda = ewc_lambda
        self.ewc_fisher_batches = ewc_fisher_batches
        self._ewc: Dict[str, Dict[str, list]] = {}
        self._ewc_shared: list = []

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
        print(f"Distill alpha:         {self.distill_alpha}"
              f"{'  (off)' if self.distill_alpha <= 0 else ''}")
        if self.ewc_lambda > 0:
            print(f"EWC lambda:            {self.ewc_lambda}  "
                  f"({self.ewc_fisher_batches} Fisher batches/domain, shared params only)")
        else:
            print(f"EWC lambda:            {self.ewc_lambda}  (off)")
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
    def _prepare_batch(self, batch):
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
        return img1, img2, (mask > 0.5).float()

    # ------------------------------------------------------------------
    # EWC over the shared parameters
    # ------------------------------------------------------------------
    def _ewc_consolidate(self, domains: Iterable[str]) -> None:
        """Estimate each domain's diagonal Fisher over the SHARED parameters and
        snapshot their current values as that domain's anchor.

        Only shared weights are protected: per-domain adapters / BatchNorm are
        private (no other domain can overwrite them) and the backbone is frozen.
        Runs in eval mode so the Fisher passes neither update BatchNorm running
        statistics nor apply dropout.  Uses the batch-level empirical Fisher
        (squared gradient of the batch loss), the usual cheap approximation.
        """
        if self.ewc_lambda <= 0.0:
            return
        shared = shared_parameters(self.model)
        if not shared:
            return
        self._ewc_shared = shared

        was_training = self.model.training
        self.model.eval()
        for d in domains:
            loader = self.train_loaders.get(d)
            if loader is None:
                continue
            fisher = [torch.zeros_like(p) for p in shared]
            n = 0
            for batch in loader:
                if n >= self.ewc_fisher_batches:
                    break
                img1, img2, mask = self._prepare_batch(batch)
                self.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits, aux_logits = self.model(img1, img2, d)
                    kw = dict(pos_weight=self.pos_weight[d], gamma=self.focal_gamma,
                              dice_weight=self.dice_weight, bce_weight=self.bce_weight)
                    obj = change_detection_loss(logits, mask, **kw)
                    if aux_logits is not None:
                        obj = obj + self.deep_supervision_weight * change_detection_loss(
                            aux_logits, mask, **kw)
                    obj.backward()
                for f, p in zip(fisher, shared):
                    if p.grad is not None:
                        f += p.grad.detach() ** 2
                n += 1
            if n == 0:
                continue
            for f in fisher:
                f /= n
            self._ewc[d] = {
                "fisher": fisher,
                "anchor": [p.detach().clone() for p in shared],
            }
            numel = sum(f.numel() for f in fisher)
            mean_f = sum(float(f.sum()) for f in fisher) / max(numel, 1)
            print(f"  [EWC] consolidated {d}: {n} batches, mean Fisher {mean_f:.3e}")

        # Leave no stale gradients for the optimisers.
        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()

    def _ewc_penalty(self, domain: str):
        """sum over the OTHER domains d' of  sum_i F_d'[i] * (theta_i - theta*_d'[i])^2.

        A domain's own Fisher never restrains its own steps; it only protects the
        shared weights from being moved by the other domains.
        """
        others = [s for d, s in self._ewc.items() if d != domain]
        if not others or not self._ewc_shared:
            return None
        total = self._ewc_shared[0].new_zeros(())
        for s in others:
            for p, f, a in zip(self._ewc_shared, s["fisher"], s["anchor"]):
                total = total + (f * (p - a) ** 2).sum()
        return total

    def _postfix(self, stats: Dict[str, float]) -> Dict[str, str]:
        out = {
            "loss": f"{stats['total']:.4f}",
            "main": f"{stats['main']:.4f}",
            "aux": f"{stats['aux']:.4f}",
        }
        if self.distill_alpha > 0.0:
            out["dst"] = f"{stats['distill']:.4f}"
        if self.ewc_lambda > 0.0:
            out["ewc"] = f"{stats['ewc']:.2e}"
        out["dice"] = f"{stats['dice']:.4f}"
        out["msum"] = int(stats["msum"])
        return out

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
        main_loss = change_detection_loss(
            logits, mask,
            pos_weight=self.pos_weight[domain],
            gamma=self.focal_gamma,
            dice_weight=self.dice_weight,
            bce_weight=self.bce_weight,
        )
        loss = main_loss
        aux_val = 0.0
        if aux_logits is not None:
            aux_loss = change_detection_loss(
                aux_logits, mask,
                pos_weight=self.pos_weight[domain],
                gamma=self.focal_gamma,
                dice_weight=self.dice_weight,
                bce_weight=self.bce_weight,
            )
            loss = loss + self.deep_supervision_weight * aux_loss
            aux_val = float(aux_loss.detach())

        # Self-distillation: pull the auxiliary head toward the main head.
        # Both predict at full resolution now (AuxDecoder upsamples with its
        # own transposed convs), so the comparison is direct -- no pooling.
        # The teacher is detached, so gradients move aux toward main and
        # never the reverse.
        distill_val = 0.0
        if aux_logits is not None and self.distill_alpha > 0.0:
            distill = F.mse_loss(aux_logits, logits.detach())
            loss = loss + self.distill_alpha * distill
            distill_val = float(distill.detach())

        # EWC: keep the shared weights near the values the OTHER domains rely on.
        # Zero until the first consolidation (end of epoch 1 / first block).
        ewc_val = 0.0
        if self.ewc_lambda > 0.0:
            pen = self._ewc_penalty(domain)
            if pen is not None:
                loss = loss + 0.5 * self.ewc_lambda * pen
                ewc_val = float(pen.detach())

        loss.backward()
        trainable = domain_parameters(self.model, domain) + shared_parameters(self.model)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.step()

        with torch.no_grad():
            _, dice, _ = _segmentation_metrics(logits, mask)
            aux_dice = 0.0
            if aux_logits is not None:
                _, ad, _ = _segmentation_metrics(aux_logits, mask)
                aux_dice = ad.item()

        # Raw (unweighted) values for each term, so the epoch summary can show
        # how the three parts of the objective actually balance.
        return {
            "total": loss.item(),
            "main": float(main_loss.detach()),
            "aux": aux_val,
            "distill": distill_val,
            "ewc": ewc_val,
            "dice": dice.item(),
            "aux_dice": aux_dice,
            "msum": mask.sum().item(),
        }

    # ------------------------------------------------------------------
    def _train_domain_block(self, domain: str, loader, epoch: int, epochs: int):
        """Run one full domain block (a contiguous run of batches on a single
        domain).  Pre-flips ``requires_grad`` via ``freeze_domain`` exactly as
        the notebook does.
        """
        freeze_domain(self.model, domain)

        acc = _LossAccum()
        pbar = tqdm(loader, desc=f"[{domain} | epoch {epoch+1}/{epochs}]", leave=False)
        for batch in pbar:
            stats = self.train_step(batch, domain)
            acc.add(stats)
            pbar.set_postfix(self._postfix(stats))

        lr = self.optimizers[domain].param_groups[0]["lr"]
        print(f"  [{domain}] epoch {epoch+1}: {acc.summary(self)}  lr={lr:.2e}")
        return acc.mean("total"), acc.mean("dice")

    def _train_mixed(self, stream, total_steps: int, epoch: int, epochs: int):
        """Used for round_robin / sequential: gradients flow only through the
        domain selected per-batch.  ``freeze_domain`` is called inside the
        loop on every domain switch.
        """
        running = {d: _LossAccum() for d in self.domain_list}
        prev = None
        pbar = tqdm(stream, total=total_steps, desc=f"[{self.schedule} | {epoch+1}/{epochs}]")
        for domain, batch in pbar:
            if domain != prev:
                freeze_domain(self.model, domain)
                prev = domain
            stats = self.train_step(batch, domain)
            running[domain].add(stats)
            pbar.set_postfix({"dom": domain, **self._postfix(stats)})
        for d, acc in running.items():
            if acc.n:
                lr = self.optimizers[d].param_groups[0]["lr"]
                print(f"  [{d}] epoch {epoch+1}: {acc.summary(self)}  lr={lr:.2e}")

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
                    # Consolidate right after this domain's block, so the next
                    # domain is anchored to the weights this one just produced.
                    self._ewc_consolidate([domain])
            else:
                batches_per_domain = max(len(l) for l in self.train_loaders.values())
                total = batches_per_domain * len(self.train_loaders)
                if self.schedule == "round_robin":
                    stream = _round_robin(self.train_loaders, total, self.domain_order)
                else:  # sequential
                    stream = _sequential(self.train_loaders, batches_per_domain, self.domain_order)
                self._train_mixed(stream, total, epoch, epochs)
                # Domains are interleaved, so there is no per-domain boundary:
                # consolidate every domain once per epoch.
                self._ewc_consolidate(self.domain_list)

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
        aux_acc = aux_dice = aux_iou = 0.0
        has_aux = False
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
                    logits, aux_logits = _tta_logits(self.model, img1, img2, domain)
                else:
                    logits, aux_logits = self.model(img1, img2, domain)

                acc, dice, iou = _segmentation_metrics(logits, mask)
                total_acc += acc.item()
                total_dice += dice.item()
                total_iou += iou.item()

                if aux_logits is not None:
                    has_aux = True
                    a_acc, a_dice, a_iou = _segmentation_metrics(aux_logits, mask)
                    aux_acc += a_acc.item()
                    aux_dice += a_dice.item()
                    aux_iou += a_iou.item()
                n += 1

        if was_training:
            self.model.train()

        n = max(n, 1)
        avg_acc = 100.0 * total_acc / n
        avg_dice = total_dice / n
        avg_iou = total_iou / n
        if epoch is not None:
            print(
                f"  [{domain} {split_label}] main: dice={avg_dice:.4f}  "
                f"iou={avg_iou:.4f}  acc={avg_acc:.2f}%"
            )
        else:
            print(
                f"[{domain} {split_label}] main: acc={avg_acc:.2f}%  "
                f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
            )

        if has_aux:
            a_acc = 100.0 * aux_acc / n
            a_dice = aux_dice / n
            a_iou = aux_iou / n
            gap = avg_dice - a_dice
            indent = "  " if epoch is not None else ""
            print(
                f"{indent}[{domain} {split_label}] aux : dice={a_dice:.4f}  "
                f"iou={a_iou:.4f}  acc={a_acc:.2f}%  (gap {gap:+.4f})"
            )

        # Checkpoint selection tracks the MAIN head -- that is the deployed
        # prediction; the aux head is reported for the early-exit comparison.
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
