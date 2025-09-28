#!/usr/bin/env python3
"""
Unify pass/fail decision to a single 'decision' column.

Behavior:
- If the CSV has a 'decision_override' column, that becomes the final 'decision'.
- Otherwise, compute decision from 'fraud_score' vs a calibrated threshold
  loaded from --config (default: config/decision_config.json) or provided
  via --threshold.

By default, only one 'decision' column is written. Use --keep-raw to also keep
the original 'decision' (as 'raw_decision') for debugging.

Examples
--------
# Real set (already has decision_override)
python3 apply_threshold_to_csv.py \
  --csv outputs/real_metrics_with_decision.csv \
  --out outputs/real_metrics_final.csv

# Fake set (no override) → uses threshold from config
python3 apply_threshold_to_csv.py \
  --csv outputs/test_fake_metrics.csv \
  --out outputs/test_fake_metrics_final.csv
"""

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd


def load_threshold(threshold_arg: Optional[float], config_path: str) -> Optional[float]:
    """
    Decide which fraud threshold to use.

    Priority:
      1) --threshold argument if provided
      2) "fraud_threshold" from config JSON if exists
      3) None (caller must handle)
    """
    if threshold_arg is not None:
        return float(threshold_arg)

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "fraud_threshold" in cfg:
                return float(cfg["fraud_threshold"])
        except Exception as e:
            print(f"[warn] Failed to read config '{config_path}': {e}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description="Unify final decision to a single 'decision' column.")
    ap.add_argument("--csv", required=True, help="Input CSV with metrics and decisions.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument(
        "--config",
        default="config/decision_config.json",
        help="JSON config file containing {'fraud_threshold': <float>}. Default: %(default)s",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override fraud threshold (pass if fraud_score < threshold).",
    )
    ap.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep original 'decision' as 'raw_decision' for debugging.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Track whether we have an existing raw decision column
    had_raw_decision = "decision" in df.columns

    # If user wants to keep the raw value, rename it to raw_decision
    if had_raw_decision and args.keep_raw:
        # Avoid double-rename if 'raw_decision' already exists for some reason
        if "raw_decision" not in df.columns:
            df = df.rename(columns={"decision": "raw_decision"})
    # If not keeping raw, and we will replace 'decision', we'll drop/overwrite below.

    # Final decision logic
    if "decision_override" in df.columns:
        # Use the calibrated override as the final decision
        final_decision = df["decision_override"].astype(str).str.lower()
        source = "override"
    else:
        # Compute from threshold
        thr = load_threshold(args.threshold, args.config)
        if thr is None:
            print(
                "[error] No 'decision_override' column and no threshold found. "
                "Provide --threshold or a config with {'fraud_threshold': ...}.",
                file=sys.stderr,
            )
            sys.exit(2)

        if "fraud_score" not in df.columns:
            print("[error] CSV missing 'fraud_score' column.", file=sys.stderr)
            sys.exit(2)

        final_decision = df["fraud_score"].apply(lambda x: "pass" if float(x) < thr else "fail")
        source = f"threshold={thr:.6f}"

    # Inject/replace unified 'decision' column
    df["decision"] = final_decision

    # Remove helper columns unless asked to keep
    cols_to_drop = []
    if "decision_override" in df.columns:
        cols_to_drop.append("decision_override")
    if had_raw_decision and not args.keep_raw:
        # If there was an original 'decision' and we're not keeping it, we already overwrote it.
        # Nothing to drop here; it is now the final column. But if we renamed earlier, drop raw_decision?
        if "raw_decision" in df.columns and not args.keep_raw:
            cols_to_drop.append("raw_decision")

    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Optional: put 'decision' right after 'image' if 'image' exists
    if "image" in df.columns:
        cols = list(df.columns)
        cols.remove("decision")
        insert_at = 1 if cols and cols[0] == "image" else 0
        cols.insert(insert_at, "decision")
        df = df[cols]

    # Small summary
    counts = df["decision"].value_counts().to_dict()
    total = int(df.shape[0])
    print(f"Unified decisions from {source}: {counts} (total {total})")

    # Write out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
