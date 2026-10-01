"""Check the U-Net and (real) SegNet models before training.  Run on the cluster:

    python check_segnet_unet.py

U-Net  : must be the same model as the old "segnet" runs -- loads
         checkpoints/segnet/best.pt with strict=True if that file exists.
SegNet : decoder unpools with the encoder's pooling indices; checks shapes,
         gradient flow, domain isolation and the frozen encoder.
Both   : prints parameter counts against the hand-computed values.
"""
import os
import sys

import torch

from src.models.ChangeDetection import SegNetDecoder, UNetDecoder
from src.models.factory import build_change_detection_model
from src.models.segnet_adapter import SegNetWithAdapters
from src.utils.helpers import domain_parameters, freeze_domain, shared_parameters

DOMAINS = ["LEVIR", "WHU"]
ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok = ok and bool(cond)


def counts(m):
    total = sum(p.numel() for p in m.parameters())
    per = sum(p.numel() for p in domain_parameters(m, "LEVIR"))
    shared = sum(p.numel() for p in shared_parameters(m))
    trainable = shared + len(DOMAINS) * per
    return total, trainable, total - trainable, per


EXPECTED = {  # (total, trainable, frozen, per-domain), computed by hand from the code
    "unet": (24_393_234, 9_670_098, 14_723_136, 135_920),
    "segnet": (31_149_554, 16_426_418, 14_723_136, 312_752),
}

torch.manual_seed(0)
models = {}
for name in ("unet", "segnet"):
    print(f"\n=== {name} ===")
    m = build_change_detection_model(DOMAINS, backbone=name)
    models[name] = m
    want_dec = UNetDecoder if name == "unet" else SegNetDecoder
    check(isinstance(m.backbone, SegNetWithAdapters), "encoder is the full VGG-16-BN (SegNetWithAdapters)")
    check(type(m.decoder) is want_dec, f"decoder is {want_dec.__name__}")
    check(all(not p.requires_grad for n, p in m.backbone.named_parameters()
              if not n.startswith("domain_adapters")), "VGG-16-BN encoder is frozen")
    got = counts(m)
    exp = EXPECTED[name]
    labels = ("total", "trainable", "frozen", "per-domain")
    for lab, g, e in zip(labels, got, exp):
        check(g == e, f"{lab:10s} {g:>12,}  (expected {e:,})")

print("\n=== U-Net is the same model as the old 'segnet' runs ===")
old = "checkpoints/segnet/best.pt"
if os.path.exists(old):
    ckpt = torch.load(old, map_location="cpu")
    try:
        models["unet"].load_state_dict(ckpt["model"], strict=True)
        check(True, f"{old} (epoch {ckpt.get('epoch')}) loads into the new U-Net with strict=True")
    except RuntimeError as exc:
        check(False, f"{old} does not load: {str(exc)[:300]}")
else:
    print(f"  skip  {old} not found (nothing to compare against)")

print("\n=== SegNet forward / backward ===")
m = models["segnet"]
x1, x2 = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
m.eval()
with torch.no_grad():
    l4_a = m.backbone.extract_multiscale(x1, "LEVIR")["l4"]
    l4_b, idx, sizes = m.backbone.extract_with_indices(x1, "LEVIR")
check(torch.equal(l4_a, l4_b), "extract_with_indices gives exactly the same features")
check([tuple(i.shape[1:2]) for i in idx] == [(64,), (128,), (256,), (512,), (512,)],
      f"pooling indices for all 5 stages: channels {[i.shape[1] for i in idx]}")

m.train()
freeze_domain(m, "LEVIR")
logits, aux = m(x1, x2, "LEVIR")
check(tuple(logits.shape) == (2, 1, 64, 64), f"main logits {tuple(logits.shape)}")
check(aux is not None and tuple(aux.shape) == (2, 1, 64, 64), "aux logits at full resolution")
(logits.mean() + aux.mean()).backward()

def has_grad(p):
    return p.grad is not None and p.grad.abs().sum().item() > 0

dec = m.decoder
check(has_grad(dec.bottleneck.conv.weight), "grad reaches the fused bottleneck conv")
check(all(has_grad(b.conv.weight) for st in dec.stages for b in st),
      "grad reaches every decoder conv (unpooling passes gradient)")
check(has_grad(dec.stages[0][0].adapters["LEVIR"].up.weight), "grad reaches LEVIR decoder adapters")
check(has_grad(m.backbone.domain_adapters["LEVIR"]["l4"].up.weight), "grad reaches LEVIR encoder adapters")
check(all(p.grad is None for p in m.backbone.domain_adapters["WHU"].parameters()),
      "WHU adapters get no grad from a LEVIR step")
check(all(p.grad is None for n, p in m.backbone.named_parameters()
          if not n.startswith("domain_adapters")), "frozen VGG-16-BN gets no grad")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
