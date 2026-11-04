"""Check the U-Net encoder's ImageNet weight transfer.  Run on the cluster:

    python check_unet_weights.py
    python check_unet_weights.py --levir-dir "./Data/LEVIR CD"

1. Every transferred conv+BN pair must reproduce VGG-16-BN exactly (bias folded
   into BatchNorm).  Also reports how far off the previous copy code was.
2. Compares the U-Net encoder's l1..l4 features with the full VGG-16-BN (the
   SegNet encoder) on real LEVIR images.  l1 must match exactly; l2..l4 differ
   because VGG convs 20, 30, 40 have no slot in a two-conv U-Net block.
"""
import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG16_BN_Weights, vgg16_bn

from src.models.unet_adapter import UNetWithAdapters

p = argparse.ArgumentParser()
p.add_argument("--levir-dir", default="./Data/LEVIR CD")
p.add_argument("--n-images", type=int, default=4)
args = p.parse_args()

torch.manual_seed(0)
vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()
f = list(vgg.features)
unet = UNetWithAdapters(["LEVIR", "WHU"], pretrained=True).eval()
ok = True

print("\n1. conv+BN pairs: U-Net vs VGG-16-BN")
with torch.no_grad():
    for stage, k, ci, bi in UNetWithAdapters._VGG16_BN_MAP:
        block = getattr(unet, stage)
        conv, bn = getattr(block, f"conv{k}"), getattr(block, f"bn{k}")
        x = torch.randn(2, conv.in_channels, 32, 32)
        ref = f[bi](f[ci](x))
        new = bn(conv(x))
        # What the previous code computed: conv bias dropped, BN mean unchanged.
        old = F.batch_norm(F.conv2d(x, f[ci].weight, None, padding=1),
                           f[bi].running_mean, f[bi].running_var,
                           f[bi].weight, f[bi].bias, False, 0.0, f[bi].eps)
        rel_new = ((new - ref).abs().max() / ref.abs().max()).item()
        off_old = ((old - ref).abs().mean() / ref.std()).item()
        good = rel_new < 1e-5
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  VGG conv {ci:2d} -> {stage}.conv{k}:  "
              f"now rel. error {rel_new:.1e}   (old code: off by {off_old:.2f} sd on average)")

print("\n2. encoder features vs full VGG-16-BN (= SegNet encoder)")
paths = sorted(glob.glob(os.path.join(args.levir_dir, "test", "A", "*.png")))[: args.n_images]
tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
if paths:
    x = torch.stack([tf(Image.open(pth).convert("RGB")) for pth in paths])
    print(f"  using {len(paths)} LEVIR test images")
else:
    x = torch.randn(args.n_images, 3, 512, 512)
    print("  [no LEVIR images found -- using random input; pass --levir-dir]")

with torch.no_grad():
    ref, h = {}, x
    for lvl, (a, b) in zip(("l1", "l2", "l3", "l4"), ((0, 14), (14, 24), (24, 34), (34, 44))):
        h = vgg.features[a:b](h)
        ref[lvl] = h
    u = {}
    h = unet.pool1(unet.stage1(x))
    h = unet.pool2(unet.stage2(h)); u["l1"] = h
    h = unet.pool3(unet.stage3(h)); u["l2"] = h
    h = unet.pool4(unet.stage4(h)); u["l3"] = h
    h = unet.pool5(unet.stage5(h)); u["l4"] = h

for lvl in ("l1", "l2", "l3", "l4"):
    r, n = ref[lvl].flatten(1), u[lvl].flatten(1)
    cos = F.cosine_similarity(r, n, dim=1).mean().item()
    rel = ((n - r).norm() / r.norm()).item()
    if lvl == "l1":
        good = rel < 1e-4
        ok &= good
        tag, note = ("PASS" if good else "FAIL"), "must match: stages 1-2 are identical to VGG"
    else:
        tag, note = "info", "differs: VGG convs 20/30/40 are not in the U-Net"
    print(f"  {tag}  {lvl}: cosine similarity {cos:.4f}   relative error {rel:.4f}   ({note})")

print("\nALL TRANSFER CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
