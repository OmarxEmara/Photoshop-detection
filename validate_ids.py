#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import subprocess
from collections import Counter

def pick_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Run full validation pipeline: batch_eval -> apply_threshold -> summary")
    ap.add_argument("--images", required=True, help="Folder with images to evaluate")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created if missing)")
    ap.add_argument("--card-weights", required=False, help="Path to ID card detector weights (passed to batch_eval if supported)")
    ap.add_argument("--obj-weights", required=False, help="Path to object detector weights (passed to batch_eval if supported)")
    ap.add_argument("--calibration", required=False, help="Path to calibration_stats.json (passed to batch_eval if supported)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Resolve tool paths from either root or tools/
    batch_eval_path = pick_first_existing([
        os.path.join(repo_root, "tools", "batch_eval.py"),
        os.path.join(repo_root, "batch_eval.py"),
    ])
    apply_thresh_path = pick_first_existing([
        os.path.join(repo_root, "tools", "apply_threshold_to_csv.py"),
        os.path.join(repo_root, "apply_threshold_to_csv.py"),
    ])

    if not batch_eval_path:
        print("ERROR: Could not find batch_eval.py (looked in tools/ and repo root).", file=sys.stderr)
        sys.exit(1)
    if not apply_thresh_path:
        print("ERROR: Could not find apply_threshold_to_csv.py (looked in tools/ and repo root).", file=sys.stderr)
        sys.exit(1)

    raw_csv   = os.path.join(args.outdir, "raw_metrics.csv")
    final_csv = os.path.join(args.outdir, "final_metrics.csv")

    # ------------------------
    # 1) Run batch_eval.py
    # ------------------------
    print("Running batch_eval.py...")
    cmd1 = [sys.executable, batch_eval_path, "--images", args.images, "--out", raw_csv]
    # Pass-through optional flags if your batch_eval supports them (safe to include)
    if args.card_weights: cmd1 += ["--card-weights", args.card_weights]
    if args.obj_weights:  cmd1 += ["--obj-weights",  args.obj_weights]
    if args.calibration:  cmd1 += ["--calibration",  args.calibration]

    subprocess.run(cmd1, check=True)

    if not os.path.isfile(raw_csv):
        print(f"ERROR: batch_eval did not produce {raw_csv}", file=sys.stderr)
        sys.exit(1)

    # ------------------------
    # 2) Run apply_threshold_to_csv.py
    # ------------------------
    print("Running apply_threshold_to_csv.py...")
    cmd2 = [sys.executable, apply_thresh_path, "--csv", raw_csv, "--out", final_csv]
    subprocess.run(cmd2, check=True)

    if not os.path.isfile(final_csv):
        print(f"ERROR: thresholding did not produce {final_csv}", file=sys.stderr)
        sys.exit(1)

    # ------------------------
    # 3) Print summary
    # ------------------------
    total = 0
    counts = Counter()
    # Prefer decision_override if present & non-empty, else fall back to decision
    with open(final_csv, newline="") as f:
        reader = csv.DictReader(f)
        has_override = "decision_override" in reader.fieldnames
        for row in reader:
            total += 1
            d = ""
            if has_override:
                d = (row.get("decision_override") or "").strip()
            if not d:
                d = (row.get("decision") or "").strip()
            if d:
                counts[d] += 1

    dataset_hint = "FAKE" if "fake" in args.images.lower() else ("REAL" if "real" in args.images.lower() else "UNKNOWN")
    print("\n----------------------------------------")
    print(f"Dataset: {dataset_hint}")
    print(f"Images:  {args.images}")
    print(f"Output:  {final_csv}")
    print("----------------------------------------")
    print(f"Summary: {total} images → " + ", ".join(f"{k.upper()}={v}" for k, v in counts.items()))
    if total > 0:
        pass_count = counts.get("pass", 0)
        print(f"Pass rate: {pass_count/total*100:.2f}%")
    print("----------------------------------------")

if __name__ == "__main__":
    main()
