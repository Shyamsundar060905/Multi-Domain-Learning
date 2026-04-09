import numbers
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable
class HFDatasetWrapper(Dataset):
    """
    A PyTorch Dataset wrapper for HuggingFace datasets.
    Handles:
    - Index type safety (numpy, torch, etc.)
    - Image conversion (array → PIL → RGB)
    - Label mapping (single-label + multi-label)
    """

    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        label_key: str = "label",
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.label_key = label_key
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from src.utils.helpers import to_python_int
        idx_converted = to_python_int(idx)
        
        if isinstance(idx_converted, list):
            return [self.__getitem__(i) for i in idx_converted]
            
        idx = idx_converted

        # 🔥 Step 2: Get item safely
        item = self.dataset[idx]

        # 🔥 Step 3: Image handling
        image = item["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        # Force RGB (important for ResNet / CNNs)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 🔥 Step 4: Label handling
        label = item[self.label_key]

        if self.label_to_idx is not None:
            # Multi-label case
            if isinstance(label, list):
                label_tensor = torch.zeros(
                    len(self.label_to_idx), dtype=torch.float32
                )
                for lbl in label:
                    label_tensor[self.label_to_idx[lbl]] = 1.0

            # Single-label case
            else:
                label_tensor = torch.tensor(
                    self.label_to_idx[label], dtype=torch.long
                )
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor