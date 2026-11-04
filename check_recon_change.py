"""Check the reconstruction-change branch before training.  Run on the cluster
(pytorch env), from the repo root:

    python check_recon_change.py

1. Builds the LEVIR model with and without --recon-change and checks the
   parameter counts (the flag-off model must equal the individual LEVIR
   reconstruction run's logged 24,957,544).
2. Forward/backward: output shapes, gradient reaches every new part.
3. The reconstruction-branch change signal is |s1 - s2|, so swapping the two
   dates must give the same change logits.
4. Runs the real main.main() for 1 epoch on tiny synthetic data, so the
   trainer's new loss term and the "rchg aggregate" scores are exercised.
"""
import contextlib
import io
import re
import sys
import tempfile

import torch

import src.models.factory as factory
from src.utils.helpers import domain_parameters, freeze_domain, shared_parameters

ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok = ok and bool(cond)


def build(recon_change):
    torch.manual_seed(0)
    return factory.build_change_detection_model(
        ["LEVIR"], backbone="unet", unfreeze_backbone=True,
        use_recon=True, recon_change=recon_change,
    )


def count(m):
    return sum(p.numel() for p in m.parameters())


print("=== 1. parameter counts ===")
base, new = build(False), build(True)
cb, cn = count(base), count(new)
check(cb == 24_957_544, f"flag OFF: {cb:,} (individual LEVIR recon run logged 24,957,544)")
check(cn == 26_213_721, f"flag ON:  {cn:,} (expected 26,213,721 = +1,256,177: change head 174,577 "
                         f"+ wider up-stage 2 1,081,600)")
check(base.decoder.recon_change_head is None and new.decoder.recon_change_head is not None,
      "change head exists only with the flag")
check(base.decoder.up_stages[1].upconv.in_channels == 256
      and new.decoder.up_stages[1].upconv.in_channels == 512,
      "up-stage 2 input: 256 without the flag, 256+256 with it")
check(new.decoder.recon_decoder.ups[0].in_channels == 512,
      "reconstruction decoder takes the encoder's l4 (512 ch) per date")
dp = sum(p.numel() for p in domain_parameters(new, "LEVIR"))
sp = sum(p.numel() for p in shared_parameters(new))
check(dp + sp == cn, f"every parameter is either LEVIR-specific ({dp:,}) or shared ({sp:,})")

print("\n=== 2. forward / backward ===")
m = new
x1, x2 = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
m.train()
freeze_domain(m, "LEVIR")
logits, aux, recons = m(x1, x2, "LEVIR", return_recon=True)
rc = m.recon_change_logits
check(tuple(logits.shape) == (2, 1, 64, 64), f"main logits {tuple(logits.shape)}")
check(aux is not None and tuple(aux.shape) == (2, 1, 64, 64), "aux logits at full resolution")
check(recons is not None and all(tuple(r.shape) == (2, 3, 64, 64) for r in recons),
      "T1' and T2' reconstructions at full resolution")
check(rc is not None and tuple(rc.shape) == (2, 1, 64, 64), "recon-change logits at full resolution")
(logits.mean() + aux.mean() + rc.mean() + recons[0].mean() + recons[1].mean()).backward()


def has_grad(p):
    return p.grad is not None and p.grad.abs().sum().item() > 0


d = m.decoder
check(has_grad(d.recon_change_head.classifier.weight), "grad reaches the recon-change head")
check(has_grad(d.recon_decoder.ups[0].weight), "grad reaches reconstruction stage 1")
check(has_grad(d.up_stages[1].upconv.weight), "grad reaches the widened U-Net up-stage 2")
check(has_grad(m.backbone.stage1[0].weight), "grad reaches the (trainable) encoder")
check(has_grad(m.backbone.domain_adapters["LEVIR"]["l4"].up.weight), "grad reaches LEVIR encoder adapter")

print("\n=== 3. swapping the dates gives the same recon-change signal ===")
m.eval()
with torch.no_grad():
    m(x1, x2, "LEVIR")
    a = m.recon_change_logits.clone()
    m(x2, x1, "LEVIR")
    b = m.recon_change_logits.clone()
check(torch.allclose(a, b, atol=1e-5), f"max |difference| = {(a - b).abs().max().item():.2e}")

print("\n=== 4. real main.main(), 1 epoch, synthetic LEVIR data ===")
import main as M


class FakeLEVIR(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, split="train", transform=None, positive_only=False,
                 image_size=64, **kw):
        g = torch.Generator().manual_seed(len(split))
        self.a = torch.randn(4, 3, image_size, image_size, generator=g)
        self.b = torch.randn(4, 3, image_size, image_size, generator=g)
        self.m = (torch.rand(4, 1, image_size, image_size, generator=g) > 0.9).float()

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.m[i]


M.LEVIRDataset = FakeLEVIR
M.list_image_names = lambda root, split, image_subdir="A": [f"{split}_{i}" for i in range(4)]
M.verify_levir_splits = lambda root, splits: None
sys.argv = ["main.py", "--config", "configs/indiv_unet_recon_trainable_levir_rchg.json",
            "--epochs", "1", "--device", "cpu", "--image-size", "64", "--batch-size", "2",
            "--num-workers", "0", "--ckpt-dir", tempfile.mkdtemp()]
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    M.main()
log = buf.getvalue()
check("Recon-change loss:     ON (weight=0.5" in log, "trainer reports the recon-change loss (weight 0.5)")
check("Recon-change:" in log, "architecture header shows the recon-change branch")
check(re.search(r"rchg=[\d.]+x0\.5", log) is not None, "per-epoch summary includes rchg=...x0.5")
check(re.search(r"\[LEVIR val\] rchg aggregate: F1=", log) is not None, "val prints rchg aggregate scores")
check(re.search(r"\[LEVIR test\] rchg aggregate: F1=", log) is not None, "final test prints rchg aggregate scores")
check(re.search(r"Trainable parameters:\s+26,213,721", log) is not None, "logged trainable params = 26,213,721")
for line in log.splitlines():
    if re.search(r"Recon|rchg|Trainable parameters|LEVIR test\] main aggregate", line):
        print("  | " + line.strip())

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
