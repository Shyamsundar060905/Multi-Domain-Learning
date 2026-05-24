import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from src.data.LEVIR_dataset import LEVIRFewShotDataset
from src.data.WHU_dataset import WHUDataset
from src.data.transforms import get_test_transform, get_train_transform
from src.models.ChangeDetection import ChangeDetectionModel
from src.models.adapter_resnet import ResNetWithAdapters
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import count_parameters, set_seed


def build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(description="Multi-Domain Change Detection (adapters + EWC)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 10))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    p.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-4))
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 8))
    p.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument("--n-way", type=int, default=defaults.get("n_way", 5))
    p.add_argument("--k-shot", type=int, default=defaults.get("k_shot", 1))
    p.add_argument("--q-query", type=int, default=defaults.get("q_query", 15))
    p.add_argument("--ewc-lambda", type=float, default=defaults.get("ewc_lambda", 1000.0))
    p.add_argument("--pos-weight", type=float, default=defaults.get("pos_weight", 20.0))
    p.add_argument("--focal-gamma", type=float, default=defaults.get("focal_gamma", 2.0))
    p.add_argument("--dice-weight", type=float, default=defaults.get("dice_weight", 0.7))
    p.add_argument("--bce-weight", type=float, default=defaults.get("bce_weight", 0.3))
    p.add_argument("--positive-only", action="store_true",
                   help="Train only on samples that contain change (recommended).")
    p.add_argument("--schedule", type=str, default=defaults.get("schedule", "sequential"),
                   choices=["round_robin", "sequential"],
                   help="round_robin: alternate WHU/LEVIR every batch. "
                        "sequential: all batches of the first domain then all of the next.")
    p.add_argument("--domain-order", type=str, nargs="+",
                   default=defaults.get("domain_order", ["LEVIR", "WHU"]),
                   help="Order in which domains are trained (sequential mode) "
                        "or rotated (round_robin mode).")
    p.add_argument("--whu-dir", type=str, default="./Data/WHU")
    p.add_argument("--levir-dir", type=str, default="./Data/LEVIR CD")
    p.add_argument("--use-change-datasets", action="store_true")
    return p


def _make_loaders(args):
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_loaders, test_loaders = {}, {}

    if not args.use_change_datasets:
        return train_loaders, test_loaders

    print("Loading WHU + LEVIR datasets...")

    try:
        whu_train = WHUDataset(
            root_dir=args.whu_dir, split="train", transform=train_transform,
            positive_only=args.positive_only,
        )
        whu_test = WHUDataset(
            root_dir=args.whu_dir, split="test", transform=test_transform,
        )
        train_loaders["WHU"] = DataLoader(
            whu_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        test_loaders["WHU"] = DataLoader(
            whu_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"WHU loaded: {len(whu_train)} train / {len(whu_test)} test")
    except Exception as e:
        print(f"[Warning] WHU loading failed: {e}")

    try:
        levir_train = LEVIRFewShotDataset(
            root_dir=args.levir_dir, split="train", transform=train_transform,
            positive_only=args.positive_only,
        )
        levir_test = LEVIRFewShotDataset(
            root_dir=args.levir_dir, split="test", transform=test_transform,
        )
        train_loaders["LEVIR"] = DataLoader(
            levir_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        test_loaders["LEVIR"] = DataLoader(
            levir_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"LEVIR loaded: {len(levir_train)} train / {len(levir_test)} test")
    except Exception as e:
        print(f"[Warning] LEVIR loading failed: {e}")

    return train_loaders, test_loaders


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    defaults = {}
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            defaults = json.load(f)

    args = build_parser(defaults=defaults).parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    train_loaders, test_loaders = _make_loaders(args)
    domain_list = list(train_loaders.keys())
    if not domain_list:
        raise SystemExit("No domains loaded -- enable --use-change-datasets and check data paths.")

    print("Initializing model...")
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone = ResNetWithAdapters(base_model, domain_list)
    model = ChangeDetectionModel(backbone, domain_list=domain_list).to(device)

    count_parameters(model)

    domain_order = [d for d in args.domain_order if d in train_loaders]
    if not domain_order:
        domain_order = domain_list

    trainer = ContinualFewShotTrainer(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        domain_list=domain_list,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        schedule=args.schedule,
        domain_order=domain_order,
    )

    print("Starting training...")
    trainer.train_joint(args.epochs)
    trainer.evaluate_all()


if __name__ == "__main__":
    main()
