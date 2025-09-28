#!/usr/bin/env python3
import os, glob, argparse, cv2, numpy as np
from services.detect import load_yolo_model, detect_card_bbox, detect_objects_raw
from services.geometry import crop_with_bbox, detect_card_quad_from_crop, warp_to_canonical

CANONICAL_W, CANONICAL_H = 864, 544

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _safe_quad_or_rect(card_crop, quad):
    h, w = card_crop.shape[:2]
    rect = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    if quad is None: return rect
    q = np.asarray(quad, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 4 or q.shape[1] != 2: return rect
    if q.shape[0] > 4:
        hull = cv2.convexHull(q)
        if hull is None or hull.shape[0] < 4: return rect
        q = hull.reshape(-1,2).astype(np.float32)
        if q.shape[0] > 4:
            c = q.mean(axis=0); d = np.linalg.norm(q-c, axis=1)
            q = q[np.argsort(-d)[:4]]
    if q.shape != (4,2): return rect
    if cv2.contourArea(q.astype(np.float32)) < 0.75*(w*h): return rect
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, help="Folder with REAL ID images (can have subfolders)")
    ap.add_argument("--card-weights", required=True)
    ap.add_argument("--obj-weights", required=True)
    ap.add_argument("--out-nid", default="train_fonts/nid")
    ap.add_argument("--out-serial", default="train_fonts/serial")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--card-conf", type=float, default=0.10)
    ap.add_argument("--card-imgsz", type=int, default=640)
    ap.add_argument("--obj-conf", type=float, default=0.12)
    ap.add_argument("--obj-imgsz", type=int, default=1280)
    args = ap.parse_args()

    ensure_dir(args.out_nid); ensure_dir(args.out_serial)

    pattern_list = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"]
    imgs = []
    for ext in (["**/"+e for e in pattern_list] if args.recursive else pattern_list):
        imgs += glob.glob(os.path.join(args.images_root, ext), recursive=args.recursive)
    imgs = sorted(imgs)

    print(f"[collect_font_crops] root={args.images_root}")
    print(f"[collect_font_crops] recursive={args.recursive}  total_found={len(imgs)}")
    if len(imgs) > 0:
        print("  sample files:")
        for p in imgs[:5]:
            print("   -", p)

    model_card = load_yolo_model(args.card_weights, imgsz=args.card_imgsz, conf=args.card_conf)
    model_obj  = load_yolo_model(args.obj_weights,  imgsz=args.obj_imgsz,  conf=args.obj_conf)

    processed = saved_nid = saved_serial = 0
    for idx, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None: continue

        card_xyxy = detect_card_bbox(model_card, img)
        if card_xyxy is None: continue
        card_crop = crop_with_bbox(img, card_xyxy)

        try:
            quad = detect_card_quad_from_crop(card_crop)
        except Exception:
            quad = None
        quad = _safe_quad_or_rect(card_crop, quad)

        try:
            warped, _ = warp_to_canonical(card_crop, quad, dst_size=(CANONICAL_W, CANONICAL_H))
        except Exception:
            continue

        try:
            dets = detect_objects_raw(model_obj, warped)
        except Exception:
            continue

        by = {}
        for d in dets:
            lbl = d.get("label"); 
            if not lbl: continue
            if lbl not in by or d["conf"] > by[lbl]["conf"]:
                by[lbl] = d

        if "nid" in by:
            x1,y1,x2,y2 = map(int, by["nid"]["bbox_xyxy"])
            x1=max(0,x1); y1=max(0,y1); x2=min(CANONICAL_W-1,x2); y2=min(CANONICAL_H-1,y2)
            if x2>x1 and y2>y1:
                cv2.imwrite(os.path.join(args.out_nid, f"nid_{idx:06d}.png"), warped[y1:y2, x1:x2])
                saved_nid += 1
        if "serial" in by:
            x1,y1,x2,y2 = map(int, by["serial"]["bbox_xyxy"])
            x1=max(0,x1); y1=max(0,y1); x2=min(CANONICAL_W-1,x2); y2=min(CANONICAL_H-1,y2)
            if x2>x1 and y2>y1:
                cv2.imwrite(os.path.join(args.out_serial, f"serial_{idx:06d}.png"), warped[y1:y2, x1:x2])
                saved_serial += 1

        processed += 1
        if processed % 25 == 0:
            print(f"[{processed}/{len(imgs)}] saved_nid={saved_nid}, saved_serial={saved_serial}")

    print(f"Done. Processed={processed}/{len(imgs)}, saved_nid={saved_nid}, saved_serial={saved_serial}")

if __name__ == "__main__":
    main()
