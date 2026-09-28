"""Parse training logs and print a clean comparison table.

Usage:
    python extract_results.py logs/exp1_full_guided.log logs/exp2_last_layer_guided.log ...

Expects log lines from trainer.py that look like:
    [LEVIR test] main: acc=...  dice=0.8762  iou=0.7793
    [LEVIR test] main aggregate: F1=0.8741  IoU=0.7789  P=0.8812  R=0.8671
    [WHU test] main: acc=...  dice=0.8212  iou=0.7231
    [WHU test] main aggregate: F1=0.8195  IoU=0.7218  P=...  R=...
    Total params:     47,509,694
    Trainable params: 24,001,662
"""

import re
import sys
import csv
import os
from pathlib import Path


# --- patterns ---------------------------------------------------------------
_PAT_F1     = re.compile(r"\[(\w+)\s+test\]\s+main aggregate:.*?F1=([\d.]+).*?IoU=([\d.]+).*?P=([\d.]+).*?R=([\d.]+)")
_PAT_DICE   = re.compile(r"\[(\w+)\s+test\]\s+main:.*?dice=([\d.]+).*?iou=([\d.]+)")
_PAT_TOTAL  = re.compile(r"Total params:\s+([\d,]+)")
_PAT_TRAIN  = re.compile(r"Trainable params:\s+([\d,]+)")


def parse_log(path: str) -> dict:
    """Extract the FINAL test block from a training log."""
    data = {
        "log": os.path.basename(path),
        "LEVIR_F1": None, "LEVIR_IoU": None,
        "LEVIR_P": None, "LEVIR_R": None, "LEVIR_Dice": None,
        "WHU_F1": None,   "WHU_IoU": None,
        "WHU_P": None,    "WHU_R": None,   "WHU_Dice": None,
        "total_params": None, "trainable_params": None,
    }

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"[warn] File not found: {path}")
        return data

    # Find the LAST occurrence of "Final test" block
    final_idx = text.rfind("Final test")
    if final_idx == -1:
        final_idx = text.rfind("Evaluating all domains")
    block = text[final_idx:] if final_idx != -1 else text

    # Domain metrics
    for m in _PAT_F1.finditer(block):
        dom = m.group(1).upper()
        data[f"{dom}_F1"]  = float(m.group(2))
        data[f"{dom}_IoU"] = float(m.group(3))
        data[f"{dom}_P"]   = float(m.group(4))
        data[f"{dom}_R"]   = float(m.group(5))

    for m in _PAT_DICE.finditer(block):
        dom = m.group(1).upper()
        data[f"{dom}_Dice"] = float(m.group(2))

    # Param counts (from anywhere in the log)
    m = _PAT_TOTAL.search(text)
    if m:
        data["total_params"] = int(m.group(1).replace(",", ""))
    m = _PAT_TRAIN.search(text)
    if m:
        data["trainable_params"] = int(m.group(1).replace(",", ""))

    return data


def _fmt(v, M=False):
    if v is None:
        return "--"
    if M:
        return f"{v/1e6:.1f}M"
    return f"{v:.4f}"


def print_table(rows):
    header = (
        f"{'Experiment':<38} | {'LEVIR F1':>8} | {'LEVIR IoU':>9} | "
        f"{'WHU F1':>6} | {'WHU IoU':>7} | "
        f"{'Avg F1':>6} | {'Trainable':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        lf1 = r["LEVIR_F1"]
        wf1 = r["WHU_F1"]
        avg = (lf1 + wf1) / 2 if (lf1 is not None and wf1 is not None) else None
        print(
            f"{r['log']:<38} | {_fmt(r['LEVIR_F1']):>8} | {_fmt(r['LEVIR_IoU']):>9} | "
            f"{_fmt(r['WHU_F1']):>6} | {_fmt(r['WHU_IoU']):>7} | "
            f"{_fmt(avg):>6} | {_fmt(r['trainable_params'], M=True):>10}"
        )
    print(sep + "\n")


def write_csv(rows, out="results_summary.csv"):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to: {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py logs/*.log")
        sys.exit(0)

    logs = sys.argv[1:]
    rows = [parse_log(p) for p in logs]
    print_table(rows)
    write_csv(rows)


if __name__ == "__main__":
    main()
