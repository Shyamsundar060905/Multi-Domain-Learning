"""WHU change-detection dataset."""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_DEFAULT_SIZE = 224

_img_transform = transforms.Compose([
    transforms.Resize((_DEFAULT_SIZE, _DEFAULT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_mask_transform = transforms.Compose([
    transforms.Resize((_DEFAULT_SIZE, _DEFAULT_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


class WHUDataset(Dataset):
    """WHU Building CD dataset (``root_dir/<split>/{A,B,OUT}``)."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        k_shot: int = 1,
        q_query: int = 1,
        positive_only: bool = False,
        min_change_pixels: int = 1,
    ):
        self.root = os.path.join(root_dir, split)
        self.transform = _img_transform if transform is None else transform
        self.mask_transform = _mask_transform
        self.k_shot = k_shot
        self.q_query = q_query

        self.A_dir = os.path.join(self.root, "A")
        self.B_dir = os.path.join(self.root, "B")
        self.label_dir = os.path.join(self.root, "OUT")

        all_names = sorted(os.listdir(self.A_dir))

        if positive_only and split == "train":
            self.img_names = self._filter_positives(all_names, min_change_pixels)
            if not self.img_names:
                print("[WHU] No positive-change samples found; falling back to all samples.")
                self.img_names = all_names
        else:
            self.img_names = all_names

    def _filter_positives(self, names, min_change_pixels: int):
        kept = []
        for name in names:
            try:
                m = Image.open(os.path.join(self.label_dir, name)).convert("L")
            except FileNotFoundError:
                continue
            t = self.mask_transform(m)
            if (t > 0).sum().item() >= min_change_pixels:
                kept.append(name)
        print(f"[WHU] positive-only filter: kept {len(kept)}/{len(names)} samples.")
        return kept

    def load_triplet(self, idx: int):
        name = self.img_names[idx]
        img1 = Image.open(os.path.join(self.A_dir, name)).convert("RGB")
        img2 = Image.open(os.path.join(self.B_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, name)).convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()
        return img1, img2, mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.load_triplet(idx)
