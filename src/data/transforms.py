"""Paired image / mask transforms for change detection.

The two bitemporal images and the change mask must share the *same* spatial
augmentation parameters every step.  ``torchvision.transforms.Compose`` cannot
guarantee that, so we wrap ``torchvision.transforms.functional`` calls into a
small paired transform helper.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PairedCDTransform:
    """Apply identical spatial ops to (img1, img2, mask).

    Train mode: resize -> random horizontal flip -> random vertical flip ->
    light random rotation -> ToTensor + Normalise.

    Test mode: resize -> ToTensor + Normalise.

    Both bitemporal images go through the same random parameters; the mask is
    resampled with nearest-neighbour so labels stay binary.
    """

    def __init__(self, size: int = 224, train: bool = True):
        self.size = size
        self.train = train

    def _resize(self, img: Image.Image, is_mask: bool) -> Image.Image:
        interp = TF.InterpolationMode.NEAREST if is_mask else TF.InterpolationMode.BILINEAR
        return TF.resize(img, [self.size, self.size], interpolation=interp)

    def __call__(
        self, img1: Image.Image, img2: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1 = self._resize(img1, is_mask=False)
        img2 = self._resize(img2, is_mask=False)
        mask = self._resize(mask, is_mask=True)

        if self.train:
            if random.random() < 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img1 = TF.rotate(img1, angle)
                img2 = TF.rotate(img2, angle)
                mask = TF.rotate(mask, angle)

        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        mask = TF.to_tensor(mask)

        img1 = TF.normalize(img1, IMAGENET_MEAN, IMAGENET_STD)
        img2 = TF.normalize(img2, IMAGENET_MEAN, IMAGENET_STD)
        mask = (mask > 0).float()
        return img1, img2, mask


def get_train_transform(size: int = 512) -> PairedCDTransform:
    return PairedCDTransform(size=size, train=True)


def get_test_transform(size: int = 512) -> PairedCDTransform:
    return PairedCDTransform(size=size, train=False)
