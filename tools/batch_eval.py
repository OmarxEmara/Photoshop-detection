#!/usr/bin/env python3
import os, sys, json, glob, subprocess, csv, argparse, re
from tqdm import tqdm

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG")

def list_images(folder):
    paths = [p for p in glob.glob(os.path.join(folder, "*")) if p.endswith(IMG_EXTS)]
    return sorted(paths)

def extract_last_json(text: str):
    """
    Extract the last top-level JSON object from noisy stdout/stderr.
    Tries all {...} blobs and returns the last one that parses.
    """
    candidates = re.findall(r"\{.*\}", text, flags=re.S)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser()
    # accept both flags; prefer --img-dir if both given
    ap.add_argument("--img-dir", help="Folder of images to evaluate")
    ap.add_argument("--images",  help="(alias) Folder of images to evaluate")
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--card-weights", required=True)
    ap.add_argument("--obj-weights", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--obj-conf", type=float, default=0.08)
    ap.add_argument("--obj-imgsz", type=int, default=1536)
    ap.add_argument("--invalid-photo-fail-conf", type=float, default=0.80)
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR-dependent checks in run_front")
    args = ap.parse_args()

    img_dir = args.img_dir or args.images
    if not img_dir or not os.path.isdir(img_dir):
        raise SystemExit(f"Image folder not found: {img_dir}")

    imgs = list_images(img_dir)
    if not imgs:
        raise SystemExit(f"No images found in {img_dir}")

    rows = []
    for i, p in enumerate(tqdm(imgs, desc="Evaluating")):
        outdir = f"/tmp/ekyc_eval/{i:06d}"
        os.makedirs(outdir, exist_ok=True)

        cmd = [
            sys.executable, "-u", "run_front.py",
            "--image", p,
            "--card-weights", args.card_weights,
            "--obj-weights",  args.obj_weights,
            "--outdir", outdir,
            "--calibration", args.calibration,
            "--invalid-photo-fail-conf", str(args.invalid_photo_fail_conf),
            "--obj-conf", str(args.obj_conf),
            "--obj-imgsz", str(args.obj_imgsz),
            "--no-review",
        ]
        if args.thresholds:
            cmd += ["--thresholds", args.thresholds]
        if args.no_ocr:
            cmd += ["--no-ocr"]

        res = subprocess.run(cmd, capture_output=True, text=True)
        text = (res.stdout or "") + "\n" + (res.stderr or "")
        j = extract_last_json(text)
        if not j:
            # Keep a copy of raw logs to debug this sample
            with open(os.path.join(outdir, "stdout_stderr.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            continue

        v = j.get("validation",{}) or {}
        m = v.get("metrics",{}) or {}
        fz = m.get("forensic",{}) or {}
        g = m.get("photo_geometry") or {}

        rows.append({
          "image": p,
          "decision": (v.get("verdict",{}) or {}).get("decision"),
          "fraud_score": v.get("fraud_score"),
          "ela_mean": m.get("ela_mean"), "ela_max": m.get("ela_max"),
          "jpeg_std": m.get("jpeg_grid_std"),
          "illum_std": m.get("illum_std"),
          "photo_ar": g.get("ar"), "photo_area": g.get("area"),
          "photo_ela_z": m.get("photo_ela_z"), "photo_jpeg_z": m.get("photo_jpeg_z"),
          "residual": fz.get("residual"), "cfa_std_z": fz.get("cfa_std_z"),
          "lbp_var_z": fz.get("lbp_var_z"), "copy_move": fz.get("copy_move_ratio"),
        })

    if not rows:
        raise SystemExit("No rows collected. Check /tmp/ekyc_eval/*/stdout_stderr.txt for clues.")

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    with open(args.outcsv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.outcsv}")

if __name__ == "__main__":
    main()
