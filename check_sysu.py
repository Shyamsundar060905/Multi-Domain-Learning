"""One-time SYSU-CD checks.  Run on the cluster (pytorch env), from the repo root:

    python check_sysu.py
    python check_sysu.py --sysu-dir ./Data/SYSU-CD --levir-dir "./Data/LEVIR CD" --whu-dir ./Data/WHU

1. Every split has matching time1 / time2 / label files and 0/255 labels.
2. Changed-pixel percentage per split, and how many train tiles have no change
   at all (those are dropped by positive_only).  For comparison it prints the
   same for the LEVIR and WHU train sets, since their pos_weights (10 and 3)
   were chosen for their change fractions.
3. No image pair appears in more than one split, checked by file CONTENT
   (MD5), not by name -- SYSU names restart at 00000 in every split.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--sysu-dir", default="./Data/SYSU-CD")
p.add_argument("--levir-dir", default="./Data/LEVIR CD")
p.add_argument("--whu-dir", default="./Data/WHU")
args = p.parse_args()
ok = True


def label_stats(label_dir, names):
    pos = tot = empty = 0
    bad_values = set()
    for n in names:
        m = np.array(Image.open(os.path.join(label_dir, n)).convert("L"))
        vals = set(np.unique(m).tolist())
        bad_values |= vals - {0, 255}
        c = int((m > 0).sum())
        pos += c
        tot += m.size
        empty += c == 0
    return 100.0 * pos / max(tot, 1), empty, bad_values


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


print("=== 1-2. SYSU-CD files, labels, change fraction ===")
pair_hash = {}
for split in ("train", "val", "test"):
    root = os.path.join(args.sysu_dir, split)
    t1, t2, lb = (os.path.join(root, d) for d in ("time1", "time2", "label"))
    n1, n2, nl = (sorted(os.listdir(d)) for d in (t1, t2, lb))
    match = n1 == n2 == nl
    ok &= match
    pct, empty, bad = label_stats(lb, nl)
    ok &= not bad
    print(f"  {split:5s}: {len(n1):5d} pairs | names match: {match} | "
          f"changed pixels {pct:5.2f}% | tiles with no change {empty:5d} "
          f"({100.0 * empty / len(nl):.1f}%) | non-0/255 label values: {sorted(bad) or 'none'}")
    for n in n1:
        pair_hash.setdefault(md5(os.path.join(t1, n)) + md5(os.path.join(t2, n)), []).append(f"{split}/{n}")

print("\n=== For comparison: LEVIR and WHU train sets ===")
for name, lab in (("LEVIR", os.path.join(args.levir_dir, "train", "label")),
                  ("WHU", os.path.join(args.whu_dir, "train", "OUT"))):
    if os.path.isdir(lab):
        pct, empty, _ = label_stats(lab, sorted(os.listdir(lab)))
        print(f"  {name:5s} train: changed pixels {pct:5.2f}% | tiles with no change {empty}")
    else:
        print(f"  {name:5s}: label folder not found at {lab} (skipped)")

print("\n=== 3. Same image pair in more than one split (by content) ===")
cross = [v for v in pair_hash.values() if len({s.split('/')[0] for s in v}) > 1]
dup_within = [v for v in pair_hash.values() if len(v) > 1 and v not in cross]
ok &= not cross
print(f"  pairs shared between splits: {len(cross)}" + (f"  e.g. {cross[:3]}" if cross else ""))
print(f"  duplicate pairs within a split: {len(dup_within)}" + (f"  e.g. {dup_within[:3]}" if dup_within else ""))

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
