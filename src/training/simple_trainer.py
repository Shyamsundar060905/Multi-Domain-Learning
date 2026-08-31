"""Plain single-domain trainer -- one model, one dataset.

No per-domain optimisers, no ``freeze_domain``, no EWC, no domain scheduling.
Reuses the same loss functions and metrics as the multi-domain trainer so
numbers stay comparable.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from src.training.trainer import _segmentation_metrics, change_detection_loss


class SimpleTrainer:
    """Standard train/eval loop for :class:`SimpleChangeDetectionModel`."""

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 20.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        deep_supervision_weight: float = 0.4,
        scheduler_step_size: int = 15,
        scheduler_gamma: float = 0.1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.deep_supervision_weight = deep_supervision_weight

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

    def _prep_batch(self, batch):
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
        return img1, img2, mask

    def train_step(self, batch):
        img1, img2, mask = self._prep_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        logits, aux_logits = self.model(img1, img2)
        loss = change_detection_loss(
            logits, mask,
            pos_weight=self.pos_weight, gamma=self.focal_gamma,
            dice_weight=self.dice_weight, bce_weight=self.bce_weight,
        )
        if aux_logits is not None:
            aux_loss = change_detection_loss(
                aux_logits, mask,
                pos_weight=self.pos_weight, gamma=self.focal_gamma,
                dice_weight=self.dice_weight, bce_weight=self.bce_weight,
            )
            loss = loss + self.deep_supervision_weight * aux_loss

        loss.backward()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            _, dice, _ = _segmentation_metrics(logits, mask)
        return loss.item(), dice.item()

    def evaluate(self, loader=None, desc: str = "test"):
        loader = loader if loader is not None else self.test_loader
        was_training = self.model.training
        self.model.eval()
        total_acc = total_dice = total_iou = 0.0
        n = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                img1, img2, mask = self._prep_batch(batch)
                logits, _ = self.model(img1, img2)
                acc, dice, iou = _segmentation_metrics(logits, mask)
                total_acc += acc.item()
                total_dice += dice.item()
                total_iou += iou.item()
                n += 1
        if was_training:
            self.model.train()
        n = max(n, 1)
        return 100.0 * total_acc / n, total_dice / n, total_iou / n

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            running_loss = running_dice = 0.0
            n = 0
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
            for batch in pbar:
                loss, dice = self.train_step(batch)
                running_loss += loss
                running_dice += dice
                n += 1
                pbar.set_postfix({"loss": f"{loss:.4f}", "dice": f"{dice:.4f}"})
            self.scheduler.step()

            n = max(n, 1)
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch + 1}: loss={running_loss / n:.4f}  dice={running_dice / n:.4f}  lr={lr:.2e}")

            if self.test_loader:
                acc, dice, iou = self.evaluate(desc=f"eval ep{epoch + 1}")
                print(f"  eval: dice={dice:.4f}  iou={iou:.4f}  acc={acc:.2f}%")

        if self.test_loader:
            acc, dice, iou = self.evaluate(desc="final test")
            print(f"\nFinal test: dice={dice:.4f}  iou={iou:.4f}  acc={acc:.2f}%")
            return acc, dice, iou
        return None
