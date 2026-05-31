"""LEVIR-CD change-detection dataset."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import PairedCDTransform, get_train_transform, get_test_transform


# ---------------------------------------------------------------------------
# Split helpers (kept in this file to avoid extra module sync issues)
# ---------------------------------------------------------------------------

def normalize_image_name(name: str) -> str:
    return os.path.normcase(name.strip())


def assert_disjoint_splits(
    names_a: Iterable[str],
    names_b: Iterable[str],
    label: str,
) -> None:
    set_a = {normalize_image_name(n) for n in names_a}
    set_b = {normalize_image_name(n) for n in names_b}
    overlap = set_a & set_b
    if overlap:
        examples = sorted(overlap)[:8]
        raise ValueError(
            f"{label}: {len(overlap)} image(s) appear in BOTH splits. "
            f"Examples: {examples}. "
            "Each split must contain disjoint tiles."
        )


def list_image_names(root_dir: str, split: str, image_subdir: str = "A") -> List[str]:
    image_dir = os.path.join(root_dir, split, image_subdir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Expected directory not found: {image_dir}")
    return sorted(os.listdir(image_dir))


def validate_triplet_files(
    root_dir: str,
    split: str,
    names: Iterable[str],
    dataset_name: str,
    mask_subdir: str = "label",
) -> None:
    base = os.path.join(root_dir, split)
    a_dir = os.path.join(base, "A")
    b_dir = os.path.join(base, "B")
    m_dir = os.path.join(base, mask_subdir)
    missing: List[str] = []
    for name in names:
        if not all(os.path.isfile(os.path.join(d, name)) for d in (a_dir, b_dir, m_dir)):
            missing.append(name)
    if missing:
        examples = missing[:8]
        raise FileNotFoundError(
            f"{dataset_name} [{split}]: {len(missing)} sample(s) missing A/B/{mask_subdir}. "
            f"Examples: {examples}"
        )


def verify_levir_splits(root_dir: str, splits: Dict[str, List[str]]) -> None:
    """Validate LEVIR-CD train/val/test are pairwise disjoint with complete triplets."""
    split_names = list(splits.keys())
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            assert_disjoint_splits(splits[a], splits[b], f"LEVIR ({a} vs {b})")
    for split, names in splits.items():
        validate_triplet_files(root_dir, split, names, "LEVIR", mask_subdir="label")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
