import argparse
import json

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.data.LEVIR_dataset import LEVIRFewShotDataset, list_image_names, verify_levir_splits
from src.data.WHU_dataset import WHUDataset
from src.data.transforms import get_test_transform, get_train_transform
from src.models.factory import build_change_detection_model, print_architecture
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import count_parameters, set_seed


def build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(description="Multi-Domain Change Detection (adapters + EWC)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 10))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    p.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-4))
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("--image-size", type=int, default=defaults.get("image_size", 512))
    p.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument("--n-way", type=int, default=defaults.get("n_way", 5))
    p.add_argument("--k-shot", type=int, default=defaults.get("k_shot", 1))
    p.add_argument("--q-query", type=int, default=defaults.get("q_query", 15))
    p.add_argument("--ewc-lambda", type=float, default=defaults.get("ewc_lambda", 1000.0))
    p.add_argument("--skip-ewc", action="store_true",
                   help="Skip post-training EWC Fisher consolidation (safe for joint training).")
    p.add_argument("--pos-weight", type=float, default=defaults.get("pos_weight", 20.0),
                   help="Global pos_weight; overridden by --pos-weight-per-domain if set.")
    p.add_argument("--pos-weight-per-domain", type=str, nargs="*",
                   default=defaults.get("pos_weight_per_domain"),
                   help="Per-domain pos_weight overrides, e.g. WHU=7 LEVIR=28.")
    p.add_argument("--focal-gamma", type=float, default=defaults.get("focal_gamma", 2.0))
    p.add_argument("--dice-weight", type=float, default=defaults.get("dice_weight", 0.7))
    p.add_argument("--bce-weight", type=float, default=defaults.get("bce_weight", 0.3))
    p.add_argument("--deep-supervision-weight", type=float,
                   default=defaults.get("deep_supervision_weight", 0.4),
                   help="Weight on the layer3 auxiliary segmentation loss.")
    p.add_argument("--oversample-cap", type=float,
                   default=defaults.get("oversample_cap", 4.0),
                   help="Max oversampling factor for smaller domains (e.g. 4 = "
                        "LEVIR repeated at most 4x per epoch).")
    p.add_argument("--positive-only", action="store_true",
                   help="Train only on samples that contain change (recommended).")
    p.add_argument("--balance-domain-samples", dest="balance_domain_samples",
                   action="store_true", default=defaults.get("balance_domain_samples", True),
                   help="Oversample smaller train domains with replacement so every "
                        "domain yields the same number of batches per epoch.")
    p.add_argument("--no-balance-domain-samples", dest="balance_domain_samples",
                   action="store_false",
                   help="Disable domain oversampling; each loader yields its natural length.")
    p.add_argument("--schedule", type=str,
                   default=defaults.get("schedule", "per_domain_full_epoch"),
                   choices=["round_robin", "sequential", "per_domain_full_epoch"],
                   help="round_robin: alternate domains every batch. "
                        "sequential: min_len batches of first then second domain per outer epoch. "
                        "per_domain_full_epoch: full inner epoch of each domain per outer epoch "
                        "(notebook style; recommended with per-domain optimisers).")
    p.add_argument("--scheduler-step-size", type=int,
                   default=defaults.get("scheduler_step_size", 15),
                   help="Step-LR step size (epochs) for the per-domain schedulers.")
    p.add_argument("--scheduler-gamma", type=float,
                   default=defaults.get("scheduler_gamma", 0.1),
                   help="Step-LR gamma for the per-domain schedulers.")
    p.add_argument("--domain-order", type=str, nargs="+",
                   default=defaults.get("domain_order", ["LEVIR", "WHU"]),
                   help="Order in which domains are trained (sequential mode) "
                        "or rotated (round_robin mode).")
    p.add_argument("--domains", type=str, nargs="+",
                   default=defaults.get("domains"),
                   choices=["WHU", "LEVIR"],
                   help="Datasets to train on (default: both). "
                        "Use one name for uni-domain baselines, e.g. --domains WHU.")
    p.add_argument("--whu-dir", type=str, default=defaults.get("whu_dir", "./Data/WHU"))
    p.add_argument("--levir-dir", type=str, default=defaults.get("levir_dir", "./Data/LEVIR CD"))
    p.add_argument("--use-change-datasets", action="store_true")
    p.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                   help="Device to run on, e.g. cuda, cuda:0, cuda:7, or cpu.")
    p.add_argument("--fusion-type", type=str, default=defaults.get("fusion_type", "abs"),
                   choices=["abs", "abs_prod"], help="Change detection feature fusion strategy.")
    p.add_argument("--domain-bn-in-adapter", action="store_true",
                   help="Add domain-specific BN in the adapter path.")
    p.add_argument("--unfreeze-layer4", action="store_true",
                   help="Unfreeze Layer4 in the ResNet backbone.")
    p.add_argument("--use-color-jitter", action="store_true",
                   help="Apply independent color jitter augmentation to images.")
    p.add_argument("--use-attention", dest="use_attention", action="store_true",
                   default=defaults.get("use_attention", True),
                   help="Add per-domain CBAM in the encoder (l4) and shared CBAM in the "
                        "decoder (fused l4). On by default.")
    p.add_argument("--no-attention", dest="use_attention", action="store_false",
                   help="Disable CBAM in both encoder and decoder.")
    p.add_argument("--adapter-stages", type=str, nargs="+",
                   default=defaults.get("adapter_stages", ["layer1", "layer2", "layer3", "layer4"]),
                   choices=["layer1", "layer2", "layer3", "layer4"],
                   help="Stages in ResNet backbone to place adapters.")
    p.add_argument("--use-tta", action="store_true",
                   help="Enable Test-Time Augmentation (hflip/vflip averaging) during eval.")
    
    if defaults.get("skip_ewc"):
        p.set_defaults(skip_ewc=True)
    if defaults.get("positive_only"):
        p.set_defaults(positive_only=True)
    if "balance_domain_samples" in defaults:
        p.set_defaults(balance_domain_samples=defaults["balance_domain_samples"])
    if defaults.get("domain_bn_in_adapter"):
        p.set_defaults(domain_bn_in_adapter=True)
    if defaults.get("unfreeze_layer4"):
        p.set_defaults(unfreeze_layer4=True)
    if defaults.get("use_color_jitter"):
        p.set_defaults(use_color_jitter=True)
    if defaults.get("use_tta"):
        p.set_defaults(use_tta=True)
    return p


def _make_loaders(args):
    train_transform = get_train_transform(args.image_size, use_color_jitter=args.use_color_jitter)
    test_transform = get_test_transform(args.image_size)
    train_loaders, eval_loaders, test_loaders = {}, {}, {}

    if not args.use_change_datasets:
        return train_loaders, eval_loaders, test_loaders

    requested = args.domains or ["WHU", "LEVIR"]
    print(f"Loading datasets: {', '.join(requested)}...")

    if "WHU" in requested:
        try:
            whu_train = WHUDataset(
                root_dir=args.whu_dir, split="train", transform=train_transform,
                positive_only=args.positive_only, image_size=args.image_size,
            )
            whu_test = WHUDataset(
                root_dir=args.whu_dir, split="test", transform=test_transform,
                image_size=args.image_size,
            )
            train_loaders["WHU"] = DataLoader(
                whu_train, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
            )
            test_loaders["WHU"] = DataLoader(
                whu_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )
            eval_loaders["WHU"] = test_loaders["WHU"]
            print(f"WHU loaded: {len(whu_train)} train / {len(whu_test)} test (eval each epoch)")
        except Exception as e:
            print(f"[Warning] WHU loading failed: {e}")

    if "LEVIR" in requested:
        try:
            levir_splits = {
                split: list_image_names(args.levir_dir, split, image_subdir="A")
                for split in ("train", "val", "test")
            }
            verify_levir_splits(args.levir_dir, levir_splits)
            print(
                f"[LEVIR] splits OK (all disjoint): "
                f"{len(levir_splits['train'])} train, "
                f"{len(levir_splits['val'])} val, "
                f"{len(levir_splits['test'])} test"
            )

            levir_train = LEVIRFewShotDataset(
                root_dir=args.levir_dir, split="train", transform=train_transform,
                positive_only=args.positive_only, image_size=args.image_size,
            )
            levir_val = LEVIRFewShotDataset(
                root_dir=args.levir_dir, split="val", transform=test_transform,
                image_size=args.image_size,
            )
            levir_test = LEVIRFewShotDataset(
                root_dir=args.levir_dir, split="test", transform=test_transform,
                image_size=args.image_size,
            )
            train_loaders["LEVIR"] = DataLoader(
                levir_train, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
            )
            eval_loaders["LEVIR"] = DataLoader(
                levir_val, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )
            test_loaders["LEVIR"] = DataLoader(
                levir_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )
            print(
                f"LEVIR loaded: {len(levir_train)} train | "
                f"{len(levir_val)} val (eval each epoch) | "
                f"{len(levir_test)} test (final only)"
            )
        except Exception as e:
            print(f"[Warning] LEVIR loading failed: {e}")

    if args.balance_domain_samples and len(train_loaders) > 1:
        train_loaders = _balance_domain_samples(train_loaders, args)

    return train_loaders, eval_loaders, test_loaders


def _balance_domain_samples(train_loaders, args):
    """Balance domains to the same sample count per epoch.

    Target = min(largest_domain, oversample_cap * smallest_domain).
    Smaller domains are oversampled with replacement; larger domains are
    randomly subsampled without replacement.  Both yield the same batch count.
    """
    sizes = {d: len(ld.dataset) for d, ld in train_loaders.items()}
    max_samples = max(sizes.values())
    min_samples = min(sizes.values())
    target = min(max_samples, int(args.oversample_cap * min_samples))
    print(f"[balance] target samples/domain/epoch = {target} "
          f"(cap={args.oversample_cap}x on smallest domain)")

    for d, ld in list(train_loaders.items()):
        n = len(ld.dataset)
        if n == target:
            continue
        replacement = n < target
        sampler = RandomSampler(
            ld.dataset, replacement=replacement, num_samples=target
        )
        train_loaders[d] = DataLoader(
            ld.dataset,
            batch_size=ld.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if replacement:
            print(f"[balance] {d}: {n} -> {target} samples ({target/n:.1f}x oversample)")
        else:
            print(f"[balance] {d}: {n} -> {target} samples ({100*target/n:.0f}% subsample)")
    return train_loaders


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

    train_loaders, eval_loaders, test_loaders = _make_loaders(args)
    domain_list = list(train_loaders.keys())
    if not domain_list:
        raise SystemExit("No domains loaded -- enable --use-change-datasets and check data paths.")

    print(f"Image size: {args.image_size}x{args.image_size}")
    mode = "uni" if len(domain_list) == 1 else "multi"
    print(f"Mode: {mode} ({', '.join(domain_list)})")
    print("Initializing model...")
    model = build_change_detection_model(
        domain_list,
        device=device,
        fusion_type=args.fusion_type,
        domain_bn_in_adapter=args.domain_bn_in_adapter,
        unfreeze_layer4=args.unfreeze_layer4,
        use_attention=args.use_attention,
        adapter_stages=args.adapter_stages,
    )
    print_architecture(mode=mode, adapter_stages=args.adapter_stages, use_attention=args.use_attention)

    count_parameters(model)

    domain_order = [d for d in args.domain_order if d in train_loaders]
    if not domain_order:
        domain_order = domain_list

    if args.pos_weight_per_domain:
        pos_weight_arg = {d: args.pos_weight for d in domain_list}
        for kv in args.pos_weight_per_domain:
            if "=" not in kv:
                raise SystemExit(f"--pos-weight-per-domain expects DOMAIN=VALUE, got {kv!r}")
            d, v = kv.split("=", 1)
            if d not in domain_list:
                print(f"[warn] --pos-weight-per-domain entry {kv!r} references unknown domain; ignored.")
                continue
            pos_weight_arg[d] = float(v)
    else:
        # Sensible defaults tuned for WHU (~13% pos) vs LEVIR (~3% pos).
        defaults_pw = {"WHU": 7.0, "LEVIR": 45.0}
        pos_weight_arg = {d: defaults_pw.get(d, args.pos_weight) for d in domain_list}

    trainer = ContinualFewShotTrainer(
        model=model,
        train_loaders=train_loaders,
        eval_loaders=eval_loaders,
        test_loaders=test_loaders,
        eval_domain_splits={"LEVIR": "val", "WHU": "test"},
        test_domain_splits={"LEVIR": "test", "WHU": "test"},
        domain_list=domain_list,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        pos_weight=pos_weight_arg,
        focal_gamma=args.focal_gamma,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        deep_supervision_weight=args.deep_supervision_weight,
        schedule=args.schedule,
        domain_order=domain_order,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        skip_ewc=args.skip_ewc,
        use_tta=args.use_tta,
    )

    print("Starting training...")
    trainer.train_joint(args.epochs)


if __name__ == "__main__":
    main()
