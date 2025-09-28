#!/usr/bin/env python3
import os, glob, argparse, joblib, cv2, numpy as np
from sklearn.svm import OneClassSVM
from services.forensics import _prep_binary, _glyph_features

def collect_features(folder, min_side=18, min_fg_ratio=0.02, max_blur_var=8.0):
    feats, kept = [], 0
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(os.path.join(folder, ext))
    paths = sorted(paths)
    for p in paths:
        img = cv2.imread(p)
        if img is None: 
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_side:  # too tiny
            continue

        # binarize & quick quality filters
        th = _prep_binary(img)  # Otsu + open (+ invert if needed)
        fg_ratio = (255 - th).sum() / (th.size * 255.0)
        if fg_ratio < min_fg_ratio:
            continue

        # simple blur check (variance of Laplacian)
        if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < max_blur_var:
            continue

        feat = _glyph_features(th)  # 1x8
        feats.append(feat)
        kept += 1
    if not feats:
        return np.zeros((0, 8), np.float32)
    X = np.vstack(feats).astype(np.float32)
    print(f"[collect] {folder}: kept={kept}/{len(paths)} -> X.shape={X.shape}")
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nid-dir", required=True, help="Folder with cropped NID images")
    ap.add_argument("--serial-dir", required=True, help="Folder with cropped serial images")
    ap.add_argument("--nu", type=float, default=0.05, help="OCSVM nu (expected outlier fraction)")
    ap.add_argument("--out", default="models/font_ocsvm.bin", help="Output model path")
    args = ap.parse_args()

    X_nid = collect_features(args.nid_dir)
    X_ser = collect_features(args.serial_dir)
    X = np.vstack([X_nid, X_ser]) if X_nid.size and X_ser.size else (X_nid if X_nid.size else X_ser)

    if X.size == 0:
        raise SystemExit("No training features collected. Check your crop folders.")

    print(f"[train] total samples: {X.shape[0]}, features: {X.shape[1]}")
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=args.nu)
    oc.fit(X)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(oc, args.out)
    print(f"[save] {args.out}")

    # quick score distribution on training (should be mostly >= 0)
    df = oc.decision_function(X).ravel()
    print(f"[scores] mean={df.mean():.3f}  p05={np.percentile(df,5):.3f}  p50={np.median(df):.3f}  p95={np.percentile(df,95):.3f}")
    print("[hint] Start with a fail threshold around -0.20 (you can tune later).")

if __name__ == "__main__":
    main()
