#!/usr/bin/env python3
import argparse, json, math, os, sys
from pathlib import Path

import pandas as pd
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Compute fraud threshold from a REAL-only CSV by target false reject rate.")
    p.add_argument("--csv", required=True, help="Path to CSV (has column 'fraud_score').")
    p.add_argument("--target-frr", type=float, default=0.01,
                   help="Target false-reject-rate on real IDs (e.g., 0.01 = 1%% of real will be classified as FAIL).")
    p.add_argument("--out", default="config/decision_config.json", help="Where to write the config JSON.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "fraud_score" not in df.columns:
        # try forgiving column name
        cand = [c for c in df.columns if c.strip().lower() == "fraud_score"]
        if not cand:
            print("ERROR: column 'fraud_score' not found in CSV.", file=sys.stderr)
            print(f"Columns found: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)

    scores = pd.to_numeric(df["fraud_score"], errors="coerce").dropna().values
    if len(scores) == 0:
        print("ERROR: No numeric fraud_score values.", file=sys.stderr)
        sys.exit(2)

    # We assume: higher score => more likely fraud (this matches your data: fails ~0.57-0.60, passes ~0.02-0.06)
    # Choose threshold at the (1 - FRR) quantile so only FRR% of real are above it and thus FAIL.
    q = 1.0 - float(args.target_frr)
    q = min(max(q, 0.5), 0.9999)  # clamp to a safe range
    threshold = float(np.quantile(scores, q))

    # Sanity print
    print(f"Computed threshold = {threshold:.6f}  (target FRR={args.target_frr:.2%})")
    print(f"Min={scores.min():.6f}  Median={np.median(scores):.6f}  Max={scores.max():.6f}")

    # Write JSON config
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "higher_is_fraud": True,     # <- if you ever discover it's inverted, set this False
        "threshold": threshold,      # <- decision: FAIL if higher_is_fraud and score > threshold, else PASS
        "source": str(Path(args.csv).resolve()),
        "target_frr": float(args.target_frr),
    }
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
