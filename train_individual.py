"""Individual baseline: one plain encoder + plain decoder, one dataset.

No per-domain adapters, no per-domain BatchNorm, no continual/EWC machinery
-- this trains and evaluates a single model on LEVIR or WHU independently, as
a baseline to compare against the multi-domain adapter architectures in
``main.py`` (branches ``Optimizations`` / ``Shared-Encoder``).

Usage:
    python train_individual.py --config configs/individual_levir.json
    python train_individual.py --config configs/individual_whu.json
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.data.LEVIR_dataset import LEVIRFewShotDataset, list_image_names, verify_levir_splits
from src.data.WHU_dataset import WHUDataset
from src.data.transforms import get_test_transform, get_train_transform
from src.models.simple_unet import SimpleChangeDetectionModel
from src.training.simple_trainer import SimpleTrainer
from src.utils.helpers import count_parameters, set_seed


def build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(description="Individual (single-domain) change detection baseline")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--domain", type=str, choices=["LEVIR", "WHU"], default=defaults.get("domain", "LEVIR"))
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 20))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    p.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-4))
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("--image-size", type=int, default=defaults.get("image_size", 512))
    p.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument("--pos-weight", type=float, default=defaults.get("pos_weight", 20.0))
    p.add_argument("--focal-gamma", type=float, default=defaults.get("focal_gamma", 2.0))
    p.add_argument("--dice-weight", type=float, default=defaults.get("dice_weight", 0.7))
    p.add_argument("--bce-weight", type=float, default=defaults.get("bce_weight", 0.3))
    p.add_argument("--deep-supervision-weight", type=float,
                    default=defaults.get("deep_supervision_weight", 0.4))
    p.add_argument("--scheduler-step-size", type=int, default=defaults.get("scheduler_step_size", 15))
    p.add_argument("--scheduler-gamma", type=float, default=defaults.get("scheduler_gamma", 0.1))
    p.add_argument("--positive-only", action="store_true",
                    help="Train only on samples that contain change (recommended).")
    p.add_argument("--use-color-jitter", action="store_true")
    p.add_argument("--use-attention", action="store_true",
                    help="Add CBAM attention block on the fused l4 features.")
    p.add_argument("--fusion-type", type=str, default=defaults.get("fusion_type", "abs"),
                    choices=["abs", "abs_prod"])
    p.add_argument("--whu-dir", type=str, default=defaults.get("whu_dir", "./Data/WHU"))
    p.add_argument("--levir-dir", type=str, default=defaults.get("levir_dir", "./Data/LEVIR CD"))
    p.add_argument("--device", type=str,
                    default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if defaults.get("positive_only"):
        p.set_defaults(positive_only=True)
    if defaults.get("use_color_jitter"):
        p.set_defaults(use_color_jitter=True)
    if defaults.get("use_attention"):
        p.set_defaults(use_attention=True)
    return p


def _make_loaders(args):
    train_transform = get_train_transform(args.image_size, use_color_jitter=args.use_color_jitter)
    test_transform = get_test_transform(args.image_size)

    if args.domain == "WHU":
        train_ds = WHUDataset(
            root_dir=args.whu_dir, split="train", transform=train_transform,
            positive_only=args.positive_only, image_size=args.image_size,
        )
        test_ds = WHUDataset(
            root_dir=args.whu_dir, split="test", transform=test_transform, image_size=args.image_size,
        )
        print(f"WHU loaded: {len(train_ds)} train / {len(test_ds)} test")
    else:
        levir_splits = {
            split: list_image_names(args.levir_dir, split, image_subdir="A")
            for split in ("train", "val", "test")
        }
        verify_levir_splits(args.levir_dir, levir_splits)
        train_ds = LEVIRFewShotDataset(
            root_dir=args.levir_dir, split="train", transform=train_transform,
            positive_only=args.positive_only, image_size=args.image_size,
        )
        test_ds = LEVIRFewShotDataset(
            root_dir=args.levir_dir, split="test", transform=test_transform, image_size=args.image_size,
        )
        print(f"LEVIR loaded: {len(train_ds)} train / {len(test_ds)} test")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    return train_loader, test_loader


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    defaults = {}
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            defaults = json.load(f)

    args = build_parser(defaults=defaults).parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    print(f"Domain: {args.domain}  (individual baseline -- one plain encoder + one plain decoder)")
    train_loader, test_loader = _make_loaders(args)

    print("Initializing model...")
    model = SimpleChangeDetectionModel(
        fusion_type=args.fusion_type, use_attention=args.use_attention,
    ).to(device)
    print("  Encoder: frozen ImageNet ResNet50 (plain, no adapters)")
    print("  Decoder: plain trainable U-Net (ordinary BatchNorm, no adapters)")
    count_parameters(model)

    trainer = SimpleTrainer(
        model, train_loader, test_loader, device,
        lr=args.lr, weight_decay=args.weight_decay, pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma, dice_weight=args.dice_weight, bce_weight=args.bce_weight,
        deep_supervision_weight=args.deep_supervision_weight,
        scheduler_step_size=args.scheduler_step_size, scheduler_gamma=args.scheduler_gamma,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
