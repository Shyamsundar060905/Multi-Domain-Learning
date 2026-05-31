"""LEVIR-CD change-detection dataset."""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from src.data.split_utils import list_image_names, validate_triplet_files
from src.data.transforms import PairedCDTransform, get_train_transform, get_test_transform


class LEVIRFewShotDataset(Dataset):
    """LEVIR-CD dataset (``root_dir/<split>/{A,B,label}``).

    Uses the official ``train/``, ``val/``, and ``test/`` folders.  All splits
    must contain disjoint image filenames — tiles in ``train/`` are the only
    ones used for gradient updates; ``val/`` is for per-epoch monitoring;
    ``test/`` is held out for final evaluation.

    Uses a paired transform so the two bitemporal images and the mask share
    identical spatial augmentation parameters every step.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[PairedCDTransform] = None,
        k_shot: int = 1,
        q_query: int = 1,
        positive_only: bool = False,
        min_change_pixels: int = 1,
        image_size: int = 512,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"LEVIR split must be 'train', 'val', or 'test', got {split!r}"
            )

        self.root_dir = root_dir
        self.split = split
        self.root = os.path.join(root_dir, split)
        self.k_shot = k_shot
        self.q_query = q_query

        if transform is None:
            transform = get_train_transform(image_size) if split == "train" else get_test_transform(image_size)
        elif not isinstance(transform, PairedCDTransform):
            raise TypeError(
                "LEVIRFewShotDataset requires a PairedCDTransform (or None) so "
                "image/mask augmentations stay synchronised."
            )
        self.paired_transform = transform

        self.A_dir = os.path.join(self.root, "A")
        self.B_dir = os.path.join(self.root, "B")
        self.label_dir = os.path.join(self.root, "label")

        all_names = list_image_names(root_dir, split, image_subdir="A")

        if positive_only and split == "train":
            self.img_names = self._filter_positives(all_names, min_change_pixels, image_size)
            if not self.img_names:
                print("[LEVIR] No positive-change samples found; falling back to all samples.")
                self.img_names = all_names
        else:
            self.img_names = all_names

        validate_triplet_files(root_dir, split, self.img_names, "LEVIR", mask_subdir="label")

    def _filter_positives(self, names, min_change_pixels: int, image_size: int):
        from torchvision import transforms
        mask_check = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        kept = []
        for name in names:
            try:
                m = Image.open(os.path.join(self.label_dir, name)).convert("L")
            except FileNotFoundError:
                continue
            if (mask_check(m) > 0).sum().item() >= min_change_pixels:
                kept.append(name)
        print(f"[LEVIR] positive-only filter: kept {len(kept)}/{len(names)} samples.")
        return kept

    def load_triplet(self, idx: int):
        name = self.img_names[idx]
        img1 = Image.open(os.path.join(self.A_dir, name)).convert("RGB")
        img2 = Image.open(os.path.join(self.B_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, name)).convert("L")
        return self.paired_transform(img1, img2, mask)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.load_triplet(idx)
