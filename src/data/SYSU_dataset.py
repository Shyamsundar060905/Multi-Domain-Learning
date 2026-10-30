"""SYSU-CD change-detection dataset.

SYSU-CD (Shi et al., IEEE TGRS 2021): 20,000 pairs of 0.5 m aerial images,
256x256, Hong Kong 2007-2014, official split 12,000 / 4,000 / 4,000.
Layout: ``root_dir/<split>/{time1,time2,label}/<name>.png`` with single-channel
0/255 labels.  Unlike LEVIR-CD and WHU-CD it is not building-only: changes also
include vegetation, roads, groundwork and sea construction.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import PairedCDTransform, get_test_transform, get_train_transform


class SYSUDataset(Dataset):
    """SYSU-CD (``root_dir/<split>/{time1,time2,label}``).

    Same behaviour as WHUDataset: a paired transform keeps the two images and
    the mask on identical spatial augmentation, and ``positive_only`` (train
    split only) keeps tiles with at least ``min_change_pixels`` changed pixels
    after resizing to ``image_size``.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[PairedCDTransform] = None,
        positive_only: bool = False,
        min_change_pixels: int = 1,
        image_size: int = 512,
    ):
        self.root = os.path.join(root_dir, split)

        if transform is None:
            transform = get_train_transform(image_size) if split == "train" else get_test_transform(image_size)
        elif not isinstance(transform, PairedCDTransform):
            raise TypeError(
                "SYSUDataset requires a PairedCDTransform (or None) so image/mask "
                "augmentations stay synchronised."
            )
        self.paired_transform = transform

        self.t1_dir = os.path.join(self.root, "time1")
        self.t2_dir = os.path.join(self.root, "time2")
        self.label_dir = os.path.join(self.root, "label")

        for name, path in (("time1", self.t1_dir), ("time2", self.t2_dir), ("label", self.label_dir)):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"SYSU [{split}]: expected directory not found: {path}")

        all_names = sorted(os.listdir(self.t1_dir))
        missing = [
            n for n in all_names
            if not (os.path.isfile(os.path.join(self.t2_dir, n))
                    and os.path.isfile(os.path.join(self.label_dir, n)))
        ]
        if missing:
            raise FileNotFoundError(
                f"SYSU [{split}]: {len(missing)} sample(s) missing a time2 or label counterpart. "
                f"Examples: {missing[:8]}"
            )

        if positive_only and split == "train":
            self.img_names = self._filter_positives(all_names, min_change_pixels, image_size)
            if not self.img_names:
                print("[SYSU] No positive-change samples found; falling back to all samples.")
                self.img_names = all_names
        else:
            self.img_names = all_names

    def _filter_positives(self, names, min_change_pixels: int, image_size: int):
        from torchvision import transforms
        mask_check = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        kept = []
        for name in names:
            m = Image.open(os.path.join(self.label_dir, name)).convert("L")
            if (mask_check(m) > 0).sum().item() >= min_change_pixels:
                kept.append(name)
        print(f"[SYSU] positive-only filter: kept {len(kept)}/{len(names)} samples.")
        return kept

    def load_triplet(self, idx: int):
        name = self.img_names[idx]
        img1 = Image.open(os.path.join(self.t1_dir, name)).convert("RGB")
        img2 = Image.open(os.path.join(self.t2_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, name)).convert("L")
        return self.paired_transform(img1, img2, mask)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.load_triplet(idx)
