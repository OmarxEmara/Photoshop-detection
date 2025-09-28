#!/usr/bin/env python3
import os, argparse, shutil
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Input folder with raw real ID images")
    ap.add_argument("--card-weights", required=True, help="YOLO weights for card detection")
    ap.add_argument("--out", required=True, help="Output folder for clean images")
    ap.add_argument("--min-area", type=float, default=0.15,
                    help="Min area ratio of detected card (default=0.15)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold for YOLO card detector")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference image size")
    args = ap.parse_args()

    ensure_dir(args.out)

    # Load YOLO card detector
    model = YOLO(args.card_weights)

    img_files = [f for f in os.listdir(args.images)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    kept, dropped = 0, 0

    for fname in tqdm(img_files, desc="Prefiltering"):
        path = os.path.join(args.images, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                dropped += 1
                continue

            h, w = img.shape[:2]
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            if res.boxes is None or len(res.boxes) == 0:
                dropped += 1
                continue

            # Take the largest detected box
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
            max_area = areas.max() / float(w * h)

            if max_area < args.min_area:
                dropped += 1
                continue

            # Passed filter → copy
            shutil.copy2(path, os.path.join(args.out, fname))
            kept += 1

        except Exception as e:
            print(f"[WARN] error processing {fname}: {e}")
            dropped += 1

    print(f"✅ Done. Kept={kept}, Dropped={dropped}, Total={len(img_files)}")

if __name__ == "__main__":
    main()
