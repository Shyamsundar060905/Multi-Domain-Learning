import os
import argparse
import torch
from torchvision.models import resnet50

from src.data.transforms import get_train_transform, get_test_transform
from src.data.builtin_datasets import get_hf_dataloader, get_advance_dataloader, get_mlrs_dataloader
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.prototypical_net import PrototypicalNetwork
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate Continual Learning Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth model checkpoint')
    parser.add_argument('--mlrs-dir', type=str, default='./Dataset/MLRSNet', help='Path to MLRS dataset')
    parser.add_argument('--hf-cache-dir', type=str, default='./.hf_cache', help='HuggingFace cache directory')
    parser.add_argument('--hf-max-samples', type=int, default=None, help='Optional cap on samples per HF dataset')
    parser.add_argument('--num-workers', type=int, default=max((os.cpu_count() or 2) // 2, 0), help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--n-way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k-shot', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--q-query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes per epoch')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    episodic_kwargs = {
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'q_query': args.q_query,
        'num_episodes': args.num_episodes
    }

    print("Loading test datasets...")
    test_loaders = {}
    
    for name, hf_id in [("EuroSAT", "blanchon/EuroSAT_RGB"), ("PatternNet", "blanchon/PatternNet")]:
        try:
            _, te = get_hf_dataloader(
                hf_id, train_transform, test_transform, None, args.num_workers,
                cache_dir=args.hf_cache_dir, max_samples=args.hf_max_samples, **episodic_kwargs
            )
            test_loaders[name] = te
        except Exception as e:
            print(f"[Warning] Failed to load {name}: {e}")

    try:
        _, advance_test = get_advance_dataloader(
            train_transform, test_transform, None, args.num_workers,
            cache_dir=args.hf_cache_dir, max_samples=args.hf_max_samples, **episodic_kwargs
        )
        test_loaders["Advance"] = advance_test
    except Exception as e:
        print(f"[Warning] Failed to load Advance dataset: {e}")

    try:
        _, mlrs_test = get_mlrs_dataloader(
            args.mlrs_dir, train_transform, test_transform, None, args.num_workers, **episodic_kwargs
        )
        test_loaders['MLRS'] = mlrs_test
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping MLRS dataset.")

    if not test_loaders:
        raise RuntimeError("No datasets could be loaded.")

    domain_list = list(test_loaders.keys())

    print("Initializing model architecture...")
    base_model = resnet50(weights=None)
    model_backbone = ResNetWithAdapters(base_model, domain_list)
    model = PrototypicalNetwork(model_backbone)

    print(f"Loading checkpoint from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # We reuse the trainer for evaluation logging, no need for train_loaders/optimizers
    trainer = ContinualFewShotTrainer(
        model=model,
        train_loaders={},
        test_loaders=test_loaders,
        domain_list=domain_list,
        optimizers={},
        schedulers={},
        device=device,
        ewc_lambda=0,
        output_dir='./checkpoints'
    )

    print("\nStarting Evaluation...")
    trainer.evaluate_all(args.n_way, args.k_shot, args.q_query)

if __name__ == "__main__":
    main()
