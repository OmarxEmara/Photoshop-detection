#!/usr/bin/env python3
import argparse, csv, math, sys
from pathlib import Path

def read_scores(csv_path):
    """Return list of fraud_score floats from a metrics CSV."""
    scores = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "fraud_score" not in r.fieldnames:
            raise SystemExit(f"[error] {csv_path} has no 'fraud_score' column.")
        for row in r:
            try:
                scores.append(float(row["fraud_score"]))
            except Exception:
                # skip bad rows quietly but keep going
                continue
    return scores

def sweep_metrics(real_scores, fake_scores, thresholds):
    """
    For each threshold t:
      predict FAIL (fake) if score >= t, else PASS (real)
    Returns list of dicts with metrics for each threshold.
    """
    n_real = len(real_scores)
    n_fake = len(fake_scores)
    if n_real == 0 or n_fake == 0:
        raise SystemExit("[error] Need non-empty real and fake score lists.")

    results = []
    for t in thresholds:
        # Confusion matrix:
        # For REAL images (should PASS): FP = failed reals, TN = passed reals
        FP = sum(1 for s in real_scores if s >= t)      # real flagged as fake
        TN = n_real - FP

        # For FAKE images (should FAIL): TP = caught fakes, FN = missed fakes (passed)
        TP = sum(1 for s in fake_scores if s >= t)      # fake flagged as fake
        FN = n_fake - TP

        # Metrics
        accuracy = (TP + TN) / (n_real + n_fake)

        precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        recall    = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Rates (as proportions, not %)
        # FAR: fake accepted (i.e., passed) rate = FN / total fakes
        # FRR: real rejected (i.e., failed) rate = FP / total reals
        far = FN / n_fake
        frr = FP / n_real

        results.append(dict(
            threshold=t,
            TP=TP, FP=FP, TN=TN, FN=FN,
            FAR=far, FRR=frr,
            Accuracy=accuracy,
            Precision=precision,
            Recall=recall,
            F1=f1
        ))
    return results

def write_csv(rows, out_csv):
    fieldnames = ["threshold","TP","FP","TN","FN","FAR","FRR","Accuracy","Precision","Recall","F1"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fieldnames})

def pick_best(rows, mode="f1", alpha=0.5, target_frr=None):
    """
    Return the best row by:
      - mode="f1": maximize F1
      - mode="sum": minimize (alpha*FAR + (1-alpha)*FRR)
      - mode="target_frr": among rows with FRR <= target_frr, minimize FAR (fallback: closest FRR above target)
    """
    if mode == "f1":
        return max(rows, key=lambda r: (r["F1"], -r["FAR"]))  # tie-breaker: smaller FAR
    elif mode == "sum":
        return min(rows, key=lambda r: alpha*r["FAR"] + (1-alpha)*r["FRR"])
    elif mode == "target_frr":
        if target_frr is None:
            raise ValueError("target_frr must be provided for mode 'target_frr'")
        eligible = [r for r in rows if r["FRR"] <= target_frr]
        if eligible:
            return min(eligible, key=lambda r: (r["FAR"], -r["F1"]))  # lowest FAR, tie-breaker highest F1
        # fallback: closest FRR above target
        return min(rows, key=lambda r: (r["FRR"] - target_frr) if r["FRR"] >= target_frr else math.inf)
    else:
        raise ValueError("Unknown mode")

def main():
    p = argparse.ArgumentParser(description="Sweep thresholds over fraud_score for real & fake metrics CSVs.")
    p.add_argument("--real", required=True, help="CSV of real IDs (has 'fraud_score').")
    p.add_argument("--fake", required=True, help="CSV of fake IDs (has 'fraud_score').")
    p.add_argument("--out",  required=True, help="Output CSV with metrics per threshold.")
    p.add_argument("--points", type=int, default=1001, help="How many thresholds to sweep (default 1001).")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for FAR in (alpha*FAR + (1-alpha)*FRR).")
    p.add_argument("--target-frr", type=float, default=None, help="If set, also compute best threshold with FRR<=target.")
    args = p.parse_args()

    real_scores = read_scores(args.real)
    fake_scores = read_scores(args.fake)

    all_scores = sorted(set(real_scores + fake_scores))
    if len(all_scores) == 0:
        raise SystemExit("[error] No scores found.")
    lo, hi = all_scores[0], all_scores[-1]
    # Expand a bit to explore outside the exact score set
    thresholds = [lo + (hi - lo) * i / (max(1,args.points-1)) for i in range(args.points)]

    rows = sweep_metrics(real_scores, fake_scores, thresholds)
    write_csv(rows, args.out)

    best_f1   = pick_best(rows, mode="f1")
    best_sum  = pick_best(rows, mode="sum", alpha=args.alpha)
    print("\n=== Best by F1 ===")
    print(f"threshold={best_f1['threshold']:.4f} | F1={best_f1['F1']:.4f} | Acc={best_f1['Accuracy']:.4f} | FAR={best_f1['FAR']:.4f} | FRR={best_f1['FRR']:.4f} | TP={best_f1['TP']} FP={best_f1['FP']} TN={best_f1['TN']} FN={best_f1['FN']}")

    print("\n=== Best by (alpha*FAR + (1-alpha)*FRR), alpha={:.2f} ===".format(args.alpha))
    s = args.alpha*best_sum["FAR"] + (1-args.alpha)*best_sum["FRR"]
    print(f"threshold={best_sum['threshold']:.4f} | score={s:.4f} | Acc={best_sum['Accuracy']:.4f} | F1={best_sum['F1']:.4f} | FAR={best_sum['FAR']:.4f} | FRR={best_sum['FRR']:.4f}")

    if args.target_frr is not None:
        best_t = pick_best(rows, mode="target_frr", target_frr=args.target_frr)
        print("\n=== Best with FRR <= {:.2%} (min FAR) ===".format(args.target_frr))
        print(f"threshold={best_t['threshold']:.4f} | Acc={best_t['Accuracy']:.4f} | F1={best_t['F1']:.4f} | FAR={best_t['FAR']:.4f} | FRR={best_t['FRR']:.4f} | TP={best_t['TP']} FP={best_t['FP']} TN={best_t['TN']} FN={best_t['FN']}")
        print("\nTip: use that threshold with apply_threshold_to_csv.py to stamp final PASS/FAIL.")

if __name__ == "__main__":
    main()
