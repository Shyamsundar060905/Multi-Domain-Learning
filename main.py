import os
import json
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_train_transform, get_test_transform
from src.data.builtin_datasets import get_hf_dataloader, get_advance_dataloader, get_mlrs_dataloader
from src.data.episodic_sampler import EpisodicBatchSampler
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.prototypical_net import PrototypicalNetwork
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import count_parameters, domain_parameters, validate_episode_config, set_seed


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Multi-Domain Few-Shot Continual Learning")
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--mlrs-dir', type=str, default=defaults.get('mlrs_dir', './Dataset/MLRSNet'), help='Path to MLRS dataset')
    parser.add_argument('--hf-cache-dir', type=str, default=defaults.get('hf_cache_dir', './.hf_cache'), help='HuggingFace cache directory')
    parser.add_argument('--hf-max-samples', type=int, default=defaults.get('hf_max_samples', None), help='Optional cap on samples per HF dataset')
    parser.add_argument('--epochs', type=int, default=defaults.get('epochs', 5), help='Number of epochs per domain')
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 1e-3), help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=defaults.get('num_workers', max((os.cpu_count() or 2) // 2, 0)), help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 42), help='Global random seed')
    parser.add_argument('--dry-run', action='store_true', default=defaults.get('dry_run', False), help='Run with synthetic data to verify pipeline')
    parser.add_argument('--skip-mlrs', action='store_true', default=defaults.get('skip_mlrs', False), help='Skip local MLRS dataset')
    parser.add_argument('--n-way', type=int, default=defaults.get('n_way', 5), help='Number of classes per episode')
    parser.add_argument('--k-shot', type=int, default=defaults.get('k_shot', 1), help='Number of support samples per class')
    parser.add_argument('--q-query', type=int, default=defaults.get('q_query', 15), help='Number of query samples per class')
    parser.add_argument('--num-episodes', type=int, default=defaults.get('num_episodes', 100), help='Number of episodes per epoch')
    parser.add_argument('--ewc-lambda', type=float, default=defaults.get('ewc_lambda', 1000.0), help='EWC regularization strength')
    parser.add_argument('--output-dir', type=str, default=defaults.get('output_dir', './checkpoints'), help='Directory to save model checkpoints')
    return parser


class SyntheticEpisodeDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rng = np.random.default_rng(seed=idx + 7)
        img_arr = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
        image = Image.fromarray(img_arr).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def create_fake_episodic_loader(transform, n_way, k_shot, q_query, num_episodes, num_workers):
    min_samples = n_way * (k_shot + q_query) * 4
    labels = [i % n_way for i in range(min_samples)]
    dataset = SyntheticEpisodeDataset(labels=labels, transform=transform)
    sampler = EpisodicBatchSampler(labels, n_way, k_shot, q_query, num_episodes)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, persistent_workers=num_workers > 0)

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = {}
    if pre_args.config:
        with open(pre_args.config, 'r', encoding='utf-8') as f:
            defaults = json.load(f)

    parser = build_parser(defaults=defaults)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    os.makedirs(args.hf_cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    validate_episode_config(args.n_way, args.k_shot, args.q_query, args.num_episodes)

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    print("Loading datasets in Episodic modes...")
    episodic_kwargs = {
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'q_query': args.q_query,
        'num_episodes': args.num_episodes
    }

    train_loaders = {}
    test_loaders = {}
    if args.dry_run:
        print("Dry run enabled: using synthetic episodic data.")
        train_loaders["Synthetic"] = create_fake_episodic_loader(
            train_transform, args.n_way, args.k_shot, args.q_query, args.num_episodes, args.num_workers
        )
        test_loaders["Synthetic"] = create_fake_episodic_loader(
            test_transform, args.n_way, args.k_shot, args.q_query, max(1, args.num_episodes // 2), args.num_workers
        )
    else:
        for name, hf_id in [("EuroSAT", "blanchon/EuroSAT_RGB"), ("PatternNet", "blanchon/PatternNet")]:
            try:
                tr, te = get_hf_dataloader(
                    hf_id, train_transform, test_transform, None, args.num_workers,
                    cache_dir=args.hf_cache_dir, max_samples=args.hf_max_samples, **episodic_kwargs
                )
                train_loaders[name] = tr
                test_loaders[name] = te
            except Exception as e:
                print(f"[Warning] Failed to load {name} ({hf_id}): {e}")

        try:
            advance_train, advance_test = get_advance_dataloader(
                train_transform, test_transform, None, args.num_workers,
                cache_dir=args.hf_cache_dir, max_samples=args.hf_max_samples, **episodic_kwargs
            )
            train_loaders["Advance"] = advance_train
            test_loaders["Advance"] = advance_test
        except Exception as e:
            print(f"[Warning] Failed to load Advance dataset: {e}")

        if not args.skip_mlrs:
            try:
                mlrs_train, mlrs_test = get_mlrs_dataloader(
                    args.mlrs_dir, train_transform, test_transform, None, args.num_workers, **episodic_kwargs
                )
                train_loaders['MLRS'] = mlrs_train
                test_loaders['MLRS'] = mlrs_test
            except FileNotFoundError as e:
                print(f"Warning: {e}. Skipping MLRS dataset.")

    if not train_loaders:
        raise RuntimeError("No datasets could be loaded. Use --dry-run to validate code path without downloads.")

    domain_list = list(train_loaders.keys())

    print("Initializing model...")
    try:
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[Warning] Could not load pretrained ResNet50 weights: {e}. Falling back to random init.")
        base_model = resnet50(weights=None)
    model_backbone = ResNetWithAdapters(base_model, domain_list)
    model = PrototypicalNetwork(model_backbone)

    for param in model.backbone.stem.parameters():
        param.requires_grad = False
    for layer in model.backbone.base_layers.values():
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.backbone.avgpool.parameters():
        param.requires_grad = False

    model = model.to(device)
    count_parameters(model)

    optimizers = {}
    schedulers = {}

    for domain in domain_list:
        optimizers[domain] = optim.Adam(domain_parameters(model, domain), lr=args.lr)
        schedulers[domain] = StepLR(optimizers[domain], step_size=15, gamma=0.1)

    trainer = ContinualFewShotTrainer(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        domain_list=domain_list,
        optimizers=optimizers,
        schedulers=schedulers,
        device=device,
        ewc_lambda=args.ewc_lambda,
        output_dir=args.output_dir
    )

    print("Starting Continual Training...")
    for domain in domain_list:
        try:
            trainer.train_task(domain, args.epochs, args.n_way, args.k_shot, args.q_query)
        except Exception as e:
            print(f"[Error] Training failed for domain {domain}: {e}")
            continue
        trainer.evaluate_all(args.n_way, args.k_shot, args.q_query)

    final_model_path = os.path.join(args.output_dir, 'final_continual_model.pth')
    print(f"\nSaving final model to {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    main()
