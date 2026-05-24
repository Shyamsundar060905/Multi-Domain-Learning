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
from src.models.adapter_resnet import ResNetWithAdapters
from src.models.prototypical_net import PrototypicalNetwork
from src.training.trainer import ContinualFewShotTrainer
from src.utils.helpers import count_parameters, domain_parameters, set_seed
from src.models.ChangeDetection import ChangeDetectionModel
from src.data.WHU_dataset import WHUDataset
from src.data.LEVIR_dataset import LEVIRFewShotDataset

from torchvision import transforms

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def collate_fn(batch):
    return batch[0]


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Multi-Domain Few-Shot Continual Learning")
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--epochs', type=int, default=defaults.get('epochs', 5), help='Number of epochs per domain')
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 1e-3), help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=defaults.get('num_workers', 8), help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 42), help='Global random seed')
    parser.add_argument('--n-way', type=int, default=defaults.get('n_way', 5), help='Number of classes per episode')
    parser.add_argument('--k-shot', type=int, default=defaults.get('k_shot', 1), help='Number of support samples per class')
    parser.add_argument('--q-query', type=int, default=defaults.get('q_query', 15), help='Number of query samples per class')
    parser.add_argument('--ewc-lambda', type=float, default=defaults.get('ewc_lambda', 1000.0), help='EWC regularization strength')
    parser.add_argument('--whu-dir', type=str, default='./Data/WHU', help='Path to WHU dataset')
    parser.add_argument('--levir-dir', type=str, default='./Data/LEVIR CD', help='Path to LEVIR dataset')
    parser.add_argument('--use-change-datasets', action='store_true', help='Include WHU + LEVIR datasets')
    return parser


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

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_loaders = {}
    test_loaders = {}
    # -------------------------
    # ADD: WHU + LEVIR
    # -------------------------
    if args.use_change_datasets:
        print("Loading WHU + LEVIR datasets...")
    
        try:
            whu_train = WHUDataset(
                root_dir=args.whu_dir,
                split="train",
                transform=train_transform,
                k_shot=args.k_shot,
                q_query=args.q_query
            )
    
            whu_test = WHUDataset(
                root_dir=args.whu_dir,
                split="test",
                transform=test_transform,
                k_shot=args.k_shot,
                q_query=args.q_query
            )
    
            train_loaders["WHU"] = DataLoader(
                whu_train,
                batch_size=1,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
    
            test_loaders["WHU"] = DataLoader(
                whu_test,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
    
            print("WHU loaded ✅")
    
        except Exception as e:
            print(f"[Warning] WHU loading failed: {e}")
    
        try:
            levir_train = LEVIRFewShotDataset(
                root_dir=args.levir_dir,
                split="train",
                transform=train_transform,
                k_shot=args.k_shot,
                q_query=args.q_query
            )
            
            levir_test = LEVIRFewShotDataset(
                root_dir=args.levir_dir,
                split="test",
                transform=test_transform,
                k_shot=args.k_shot,
                q_query=args.q_query
            )    
            train_loaders["LEVIR"] = DataLoader(
                levir_train,
                batch_size=1,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=True
            )
    
            test_loaders["LEVIR"] = DataLoader(
                levir_test,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
    
            print("LEVIR loaded ✅")
    
        except Exception as e:
            print(f"[Warning] LEVIR loading failed: {e}")


    domain_list = list(train_loaders.keys())


    print("Initializing model...")
    try:
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[Warning] Could not load pretrained ResNet50 weights: {e}. Falling back to random init.")
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    model_backbone = ResNetWithAdapters(base_model, domain_list)

    for name, param in model_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    model = ChangeDetectionModel(model_backbone)

    # Freeze stem
    for param in model.backbone.stem.parameters():
        param.requires_grad = False
    
    # Freeze layer1 and layer2 manually
    for param in model.backbone.layer1.parameters():
        param.requires_grad = False
    
    for param in model.backbone.layer2.parameters():
        param.requires_grad = False
    
    # Keep adapters trainable (handled by freeze_domain)

    model = model.to(device)
    count_parameters(model)

    # optimizers = {}
    # schedulers = {}

    # for domain in domain_list:
    #     optimizers[domain] = optim.Adam(domain_parameters(model, domain), lr=args.lr)
    #     schedulers[domain] = StepLR(optimizers[domain], step_size=15, gamma=0.1)

    trainer = ContinualFewShotTrainer(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        domain_list=domain_list,
        device=device,
        ewc_lambda=args.ewc_lambda
    )    
    
    print("Starting Continual Training...")
    trainer.train_joint(args.epochs, args.n_way, args.k_shot, args.q_query)
    trainer.evaluate_all(args.n_way, args.k_shot, args.q_query)
if __name__ == "__main__":
    main()
