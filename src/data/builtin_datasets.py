import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision.datasets import ImageFolder
from src.data.wrapper import HFDatasetWrapper
from src.data.episodic_sampler import EpisodicBatchSampler
from src.utils.helpers import to_python_int

class ApplyTransform(Dataset):
    """
    Dataset wrapper to apply a specific transform.
    Useful for applying different transforms to train/test splits derived from random_split.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        idx_converted = to_python_int(idx)
        
        if isinstance(idx_converted, list):
            return [self.__getitem__(i) for i in idx_converted]
            
        x, y = self.dataset[idx_converted]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

def extract_labels(dataset):
    """Extracts all labels from a dataset for the sampler."""
    if isinstance(dataset, HFDatasetWrapper):
        labels = [dataset.dataset[i][dataset.label_key] for i in range(len(dataset.dataset))]
        if dataset.label_to_idx is not None:
            labels = [dataset.label_to_idx[l] for l in labels]
        return labels
    elif isinstance(dataset, ApplyTransform):
        underlying = dataset.dataset
        return extract_labels(underlying)
    elif hasattr(dataset, 'targets'):
        return dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        return [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label if isinstance(label, int) else label.item())
        return labels

def _create_dataloader(dataset, batch_size, num_workers, shuffle, **kwargs):
    n_way = kwargs.get('n_way')
    k_shot = kwargs.get('k_shot')
    q_query = kwargs.get('q_query')
    num_episodes = kwargs.get('num_episodes')
    
    persistent_workers = num_workers > 0
    pin_memory = torch.cuda.is_available()

    if n_way is not None and k_shot is not None and q_query is not None and num_episodes is not None:
        labels = extract_labels(dataset)
        sampler = EpisodicBatchSampler(labels, n_way, k_shot, q_query, num_episodes)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

def get_hf_dataloader(dataset_name: str, train_transform, test_transform, batch_size: int, 
                      num_workers: int, test_size: float = 0.6, seed: int = 42,
                      cache_dir: str = None, max_samples: int = None, **kwargs):
    """Generic function to load HuggingFace datasets and create dataloaders."""
    dataset = load_dataset(
        dataset_name,
        split="train",
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists"
    )
    if "image" not in dataset.column_names:
        raise ValueError(f"Dataset {dataset_name} does not expose an 'image' column.")
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    
    train_ds = split_dataset['train']
    test_ds = split_dataset['test']
    
    wrapped_train = HFDatasetWrapper(train_ds, transform=train_transform)
    wrapped_test = HFDatasetWrapper(test_ds, transform=test_transform)
    
    train_loader = _create_dataloader(wrapped_train, batch_size, num_workers, shuffle=True, **kwargs)
    test_loader = _create_dataloader(wrapped_test, batch_size, num_workers, shuffle=False, **kwargs)
    
    return train_loader, test_loader

def get_advance_dataloader(train_transform, test_transform, batch_size: int, num_workers: int,
                           cache_dir: str = None, max_samples: int = None, **kwargs):
    """Specific function for Advance dataset to handle weighted sampling or episodic."""
    dataset = load_dataset(
        "blanchon/ADVANCE",
        split='train',
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists"
    )
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    
    advance_train = split_dataset['train']
    advance_test = split_dataset['test']
    
    all_labels = set(example['label'] for example in advance_train)
    sorted_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)} 
    
    wrapped_advance_train = HFDatasetWrapper(
        advance_train,
        transform=train_transform,
        label_key='label',
        label_to_idx=label_to_idx
    )
    
    wrapped_advance_test = HFDatasetWrapper(
        advance_test,
        transform=test_transform,
        label_key='label',
        label_to_idx=label_to_idx
    )
    
    n_way = kwargs.get('n_way')
    if n_way is not None:
        train_loader = _create_dataloader(wrapped_advance_train, batch_size, num_workers, shuffle=True, **kwargs)
        test_loader = _create_dataloader(wrapped_advance_test, batch_size, num_workers, shuffle=False, **kwargs)
        return train_loader, test_loader
    else:
        labels = [label_to_idx[example['label']] for example in advance_train]
        class_sample_counts = [labels.count(i) for i in range(len(sorted_labels))]
        weights = [1.0 / class_sample_counts[label] for label in labels]
        
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        advance_train_loader = DataLoader(
            wrapped_advance_train, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available()
        )
        
        advance_test_loader = DataLoader(
            wrapped_advance_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available()
        )
        
        return advance_train_loader, advance_test_loader

def get_mlrs_dataloader(data_dir: str, train_transform, test_transform, batch_size: int, num_workers: int, **kwargs):
    """Loads MLRS dataset from a local directory."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"MLRS dataset directory not found at {data_dir}")
        
    dataset = ImageFolder(root=data_dir) 
    
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Properly decouple augmentations!
    train_dataset = ApplyTransform(train_dataset, transform=train_transform)
    test_dataset = ApplyTransform(test_dataset, transform=test_transform)
    
    train_loader = _create_dataloader(train_dataset, batch_size, num_workers, shuffle=True, **kwargs)
    test_loader = _create_dataloader(test_dataset, batch_size, num_workers, shuffle=False, **kwargs)
    
    return train_loader, test_loader
