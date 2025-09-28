#!/usr/bin/env python3
import os, cv2, numpy as np, json
from services.detect import load_yolo_model, detect_objects_raw

CANONICAL_W, CANONICAL_H = 864, 544  # same as run_front

def photo_geometry(box, w=CANONICAL_W, h=CANONICAL_H):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    area = (bw * bh) / float(w * h)
    ar = bw / float(bh) if bh > 0 else 0
    return dict(width=bw, height=bh, ar=ar, area=area)

def main(img_dir, obj_weights, out_json="calibration_stats.json"):
    model = load_yolo_model(obj_weights, imgsz=640, conf=0.10)
    results = []

    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        dets = detect_objects_raw(model, img)
        photo_dets = [d for d in dets if d["label"] == "photo"]
        if not photo_dets:
            continue

        best = max(photo_dets, key=lambda d: d["conf"])
        geom = photo_geometry(best["bbox_xyxy"])
        results.append(geom)

    # summarize
    areas = [r["area"] for r in results]
    ars   = [r["ar"] for r in results]
    summary = {
        "n": len(results),
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "ar_min": float(np.min(ars)),
        "ar_max": float(np.max(ars)),
        "ar_mean": float(np.mean(ars)),
        "ar_std": float(np.std(ars)),
        "samples": results[:10]  # preview of first 10
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", dest="img_dir", required=True)
    ap.add_argument("--obj-weights", required=True)
    ap.add_argument("--out", default="calibration_stats.json")
    args = ap.parse_args()
    main(args.img_dir, args.obj_weights, args.out)

