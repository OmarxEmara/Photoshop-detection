#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"photo": 0.55, "ela_global": 0.15, "jpeg_global": 0.15, "illum": 0.05, "adv_forensics": 0.10}
DEFAULT_SCORE_THR = 0.60

def p(df, col, q):
    s = df[col].dropna()
    return float(np.percentile(s, q)) if len(s) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-metrics", required=True, help="CSV from tools/batch_eval.py")
    ap.add_argument("--calibration", required=True, help="calibration_stats.json for geometry bands")
    ap.add_argument("--out", required=True, help="Output thresholds JSON")
    ap.add_argument("--inflate", type=float, default=1.05, help="Safety multiplier on p99 for hard-fail")
    args = ap.parse_args()

    df = pd.read_csv(args.real_metrics)
    with open(args.calibration, "r", encoding="utf-8") as f:
        calib = json.load(f)

    thr = {
        "weights": DEFAULT_WEIGHTS,
        "score_threshold": DEFAULT_SCORE_THR,
        "geometry": {
            "ar_lo": float(calib["photo_ar"]["p05"]),
            "ar_hi": float(calib["photo_ar"]["p95"]),
            "area_lo": float(calib["photo_area"]["p05"]),
            "area_hi": float(calib["photo_area"]["p95"]),
            "pad_ar": 0.02,
            "pad_area": 0.01
        },
        "hard_fail": {},
        "soft_starts": {}
    }

    # Collect percentiles from real-only
    metrics = ["ela_mean","ela_max","jpeg_std","illum_std","photo_ela_z","photo_jpeg_z",
               "residual","cfa_std_z","lbp_var_z","copy_move"]
    stats = {m: {"p95": p(df, m, 95), "p99": p(df, m, 99)} for m in metrics}

    # Hard-fail at inflated p99; soft starts around p95
    for k, v in stats.items():
        if v["p99"] is not None:
            thr["hard_fail"][k] = float(v["p99"] * args.inflate)
        if v["p95"] is not None:
            thr["soft_starts"][k] = float(v["p95"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(thr, f, ensure_ascii=False, indent=2)
    print(f"Wrote thresholds to {args.out}")

if __name__ == "__main__":
    main()
