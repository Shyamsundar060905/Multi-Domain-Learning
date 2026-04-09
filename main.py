import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights

from src.data.transforms import get_train_transform, get_test_transform
from src.data.builtin_datasets import get_hf_dataloader, get_advance_dataloader, get_mlrs_dataloader
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.prototypical_net import PrototypicalNetwork
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import count_parameters, domain_parameters

def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Few-Shot Continual Learning")
    parser.add_argument('--mlrs-dir', type=str, default='./data/MLRSNet', help='Path to MLRS dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per domain')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() // 2, help='Number of dataloader workers')
    
    # Few-Shot params
    parser.add_argument('--n-way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k-shot', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--q-query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes per epoch')
    
    # EWC params
    parser.add_argument('--ewc-lambda', type=float, default=1000.0, help='EWC regularization strength')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    print("Loading datasets in Episodic modes...")
    episodic_kwargs = {
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'q_query': args.q_query,
        'num_episodes': args.num_episodes
    }

    euroSAT_train, euroSAT_test = get_hf_dataloader("blanchon/EuroSAT_RGB", train_transform, test_transform, None, args.num_workers, **episodic_kwargs)
    patternNet_train, patternNet_test = get_hf_dataloader("blanchon/PatternNet", train_transform, test_transform, None, args.num_workers, **episodic_kwargs)
    advance_train, advance_test = get_advance_dataloader(train_transform, test_transform, None, args.num_workers, **episodic_kwargs)
    
    try:
        mlrs_train, mlrs_test = get_mlrs_dataloader(args.mlrs_dir, train_transform, test_transform, None, args.num_workers, **episodic_kwargs)
        has_mlrs = True
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping MLRS dataset.")
        has_mlrs = False

    train_loaders = {
        'EuroSAT': euroSAT_train,
        'PatternNet': patternNet_train,
        'Advance': advance_train
    }
    
    test_loaders = {
        'EuroSAT': euroSAT_test,
        'PatternNet': patternNet_test,
        'Advance': advance_test
    }
    
    if has_mlrs:
        train_loaders['MLRS'] = mlrs_train
        test_loaders['MLRS'] = mlrs_test

    domain_list = list(train_loaders.keys())

    print("Initializing model...")
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
        ewc_lambda=args.ewc_lambda
    )

    print("Starting Continual Training...")
    for domain in domain_list:
        trainer.train_task(domain, args.epochs, args.n_way, args.k_shot, args.q_query)
        trainer.evaluate_all(args.n_way, args.k_shot, args.q_query)

if __name__ == "__main__":
    main()
