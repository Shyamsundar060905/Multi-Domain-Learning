"""Check the reconstruction-change branch before training.  Run on the cluster
(pytorch env), from the repo root:

    python check_recon_change.py

1. Builds the LEVIR model for several --recon-change-stages choices and checks
   the parameter counts (flag off must equal the individual LEVIR
   reconstruction run's logged 24,957,544; stage 1 must equal the first
   recon-change run's 26,213,721).
2. Forward/backward with stages 1+2+3: shapes, gradient reaches every new part.
3. The change signal is |date-1 features - date-2 features|, so swapping the
   two dates must give the same change logits at every stage.
4. Runs the real main.main() for 1 epoch on tiny synthetic data with stages
   2 and 3, so the trainer's loss term and per-stage "rchg sK aggregate"
   scores are exercised.
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


def build(recon_change, stages=(1,)):
    torch.manual_seed(0)
    return factory.build_change_detection_model(
        ["LEVIR"], backbone="unet", unfreeze_backbone=True,
        use_recon=True, recon_change=recon_change, recon_change_stages=stages,
    )


def count(m):
    return sum(p.numel() for p in m.parameters())


# Expected counts, worked out by hand from the layer definitions:
#   stage 1: head 174,577 + up-stage 2 widened 256->512 ch   1,081,600
#   stage 2: head  43,249 + up-stage 3 widened 128->256 ch     270,464
#   stage 3: head  10,353 + up-stage 4 widened  64->128 ch     215,104
EXPECTED = {
    None: 24_957_544,
    (1,): 26_213_721,
    (2,): 25_271_257,
    (3,): 25_183_001,
    (2, 3): 25_496_714,
    (1, 2, 3): 26_752_891,
}
UP_IN_BASE = {1: 256, 2: 128, 3: 64}
RECON_W = {1: 256, 2: 128, 3: 64}

print("=== 1. parameter counts ===")
for stages, exp in EXPECTED.items():
    m = build(stages is not None, stages or (1,))
    c = count(m)
    label = "flag OFF" if stages is None else f"stages {list(stages)}"
    check(c == exp, f"{label:15s} {c:,} (expected {exp:,})")
    if stages is not None:
        d = m.decoder
        check(sorted(int(k) for k in d.recon_change_heads) == list(stages),
              f"{label:15s} one change head per stage")
        widths = {k: d.up_stages[k].upconv.in_channels for k in (1, 2, 3)}
        want = {k: UP_IN_BASE[k] + (RECON_W[k] if k in stages else 0) for k in (1, 2, 3)}
        check(widths == want, f"{label:15s} U-Net up-stage 2/3/4 inputs {widths}")
        dp = sum(p.numel() for p in domain_parameters(m, "LEVIR"))
        sp = sum(p.numel() for p in shared_parameters(m))
        check(dp + sp == c, f"{label:15s} every parameter is LEVIR-specific or shared")

print("\n=== 2. forward / backward, stages 1+2+3 ===")
m = build(True, (1, 2, 3))
x1, x2 = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
m.train()
freeze_domain(m, "LEVIR")
logits, aux, recons = m(x1, x2, "LEVIR", return_recon=True)
rc = m.recon_change_logits
check(tuple(logits.shape) == (2, 1, 64, 64), f"main logits {tuple(logits.shape)}")
check(aux is not None and tuple(aux.shape) == (2, 1, 64, 64), "aux logits at full resolution")
check(recons is not None and all(tuple(r.shape) == (2, 3, 64, 64) for r in recons),
      "T1' and T2' reconstructions at full resolution")
check(isinstance(rc, dict) and sorted(rc) == [1, 2, 3]
      and all(tuple(v.shape) == (2, 1, 64, 64) for v in rc.values()),
      "recon-change logits for stages 1, 2, 3, each at full resolution")
(logits.mean() + aux.mean() + sum(v.mean() for v in rc.values())
 + recons[0].mean() + recons[1].mean()).backward()


def has_grad(p):
    return p.grad is not None and p.grad.abs().sum().item() > 0


d = m.decoder
for k in (1, 2, 3):
    check(has_grad(d.recon_change_heads[str(k)].classifier.weight), f"grad reaches stage-{k} change head")
    check(has_grad(d.recon_decoder.ups[k - 1].weight), f"grad reaches reconstruction stage {k}")
    check(has_grad(d.up_stages[k].upconv.weight), f"grad reaches widened U-Net up-stage {k + 1}")
check(has_grad(m.backbone.stage1[0].weight), "grad reaches the (trainable) encoder")
check(has_grad(m.backbone.domain_adapters["LEVIR"]["l4"].up.weight), "grad reaches LEVIR encoder adapter")

print("\n=== 3. swapping the dates gives the same recon-change signal ===")
m.eval()
with torch.no_grad():
    m(x1, x2, "LEVIR")
    a = {k: v.clone() for k, v in m.recon_change_logits.items()}
    m(x2, x1, "LEVIR")
    b = m.recon_change_logits
for k in (1, 2, 3):
    check(torch.allclose(a[k], b[k], atol=1e-5),
          f"stage {k}: max |difference| = {(a[k] - b[k]).abs().max().item():.2e}")

print("\n=== 4. real main.main(), 1 epoch, synthetic LEVIR data, stages 2 3 ===")
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
            "--recon-change-stages", "2", "3",
            "--epochs", "1", "--device", "cpu", "--image-size", "64", "--batch-size", "2",
            "--num-workers", "0", "--ckpt-dir", tempfile.mkdtemp()]
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    M.main()
log = buf.getvalue()
check("Recon-change loss:     ON (weight=0.5, stages [2, 3]" in log,
      "trainer reports the recon-change loss (weight 0.5, stages [2, 3])")
check("Recon-change stage 2:" in log and "Recon-change stage 3:" in log
      and "Recon-change stage 1:" not in log, "architecture header lists stages 2 and 3 only")
check(re.search(r"rchg=[\d.]+x0\.5", log) is not None, "per-epoch summary includes rchg=...x0.5")
for split in ("val", "test"):
    for k in (2, 3):
        check(re.search(rf"\[LEVIR {split}\] rchg s{k} aggregate: F1=", log) is not None,
              f"{split} prints rchg s{k} aggregate scores")
check(re.search(r"Trainable parameters:\s+25,496,714", log) is not None,
      "logged trainable params = 25,496,714")
for line in log.splitlines():
    if re.search(r"Recon|rchg s|Trainable parameters|LEVIR test\] main aggregate", line):
        print("  | " + line.strip())

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
