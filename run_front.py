#!/usr/bin/env python3
import os, json, argparse, logging
import cv2
import numpy as np

from services.detect import (
    load_yolo_model,
    detect_card_bbox,
    detect_objects_raw,
    map_raw_boxes_to_warped,
    detect_objects_direct_on_warped,
)
from services.geometry import (
    crop_with_bbox,
    detect_card_quad_from_crop,
    warp_to_canonical,
    aspect_ratio_ok,
    estimate_skew_deg,
    draw_boxes,
)
# from services.ocr_utils import (
#     set_paddle_paths,
#     get_paddle,
#     extract_arabic_field,
#     extract_address_field,
#     extract_digits_strict,
#     extract_date_digits,
#     decode_egyptian_id,
#     tesseract_text_simple,
# )
# Validation / Forgery checks
from services.forgery import compute_validation

CANONICAL_W, CANONICAL_H = 864, 544


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _draw_overlay(img, dets, path):
    overlay = img.copy()
    draw_boxes(overlay, [{**d, "bbox_xyxy": [int(a) for a in d["bbox_xyxy"]]} for d in dets])
    cv2.imwrite(path, overlay)


def run(image_path, card_weights, obj_weights, outdir,
        card_conf=0.10, card_imgsz=640, obj_conf=0.12, obj_imgsz=1280,
        paddle_det=None, paddle_rec=None,
        calibration=None,
        thresholds_path=None,
        invalid_photo_fail_conf=0.80,
        no_review=False,
        no_ocr=False):

    ensure_dir(outdir)

    # --- OCR init (optional) ---
    if not no_ocr:
        set_paddle_paths(paddle_det, paddle_rec)
        _ = get_paddle()

    # --- YOLO models ---
    model_card = load_yolo_model(card_weights, imgsz=card_imgsz, conf=card_conf)
    model_obj  = load_yolo_model(obj_weights,  imgsz=obj_imgsz,  conf=obj_conf)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # --- Card detect & warp ---
    card_xyxy = detect_card_bbox(model_card, img)
    if card_xyxy is None:
        raise RuntimeError("No card detected.")
    x1, y1, x2, y2 = map(int, card_xyxy)
    card_crop = crop_with_bbox(img, card_xyxy)

    quad = detect_card_quad_from_crop(card_crop)
    ch, cw = card_crop.shape[:2]
    if cv2.contourArea(quad.astype(np.float32)) < 0.75 * (cw * ch):
        quad = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)

    warped, H_crop2canon = warp_to_canonical(card_crop, quad, dst_size=(CANONICAL_W, CANONICAL_H))
    ar, ar_ok = aspect_ratio_ok(warped)
    gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    skew = estimate_skew_deg(gray_warp)
    cv2.imwrite(os.path.join(outdir, "01_warped.png"), warped)

    # --- Field detection ---
    raw_det = detect_objects_raw(model_obj, img)
    mapped_to_warped = map_raw_boxes_to_warped(raw_det, (x1, y1, x2, y2), H_crop2canon, dst_size=(CANONICAL_W, CANONICAL_H))
    direct_det = detect_objects_direct_on_warped(model_obj, warped)

    _draw_overlay(img, raw_det, os.path.join(outdir, "02a_raw_overlay.png"))
    _draw_overlay(warped, direct_det, os.path.join(outdir, "02b_direct_on_warp_overlay.png"))

    # Merge detections by label
    merged = {}
    def consider(det_list, source):
        for d in det_list:
            if d["label"] not in merged or d["conf"] > merged[d["label"]]["conf"]:
                merged[d["label"]] = {**d, "source": source}
    consider(mapped_to_warped, "mapped_from_raw")
    consider(direct_det, "direct_on_warp")

    # Fallback heuristic for photo
    fallback_used = False
    if "photo" not in merged and "invalid_photo" not in merged:
        cand = None; cand_score = -1.0
        for d in direct_det + mapped_to_warped:
            x1b, y1b, x2b, y2b = map(int, d["bbox_xyxy"])
            w = max(1, x2b - x1b); h = max(1, y2b - y1b)
            ar_loc = w / float(h)
            area = (w*h) / float(CANONICAL_W * CANONICAL_H)
            cx = 0.5 * (x1b + x2b) / CANONICAL_W
            score = (0.5 if cx < 0.45 else 0.0) + (min(area, 0.12)*2.0) + (1.0 - min(abs(ar_loc-0.8), 0.8))
            if score > cand_score:
                cand, cand_score = [x1b, y1b, x2b, y2b], score
        if cand is not None:
            merged["photo"] = {"label":"photo","bbox_xyxy":[float(v) for v in cand],"conf":0.25,"source":"fallback"}
            fallback_used = True
            fb = warped.copy()
            bx = [int(x) for x in merged["photo"]["bbox_xyxy"]]
            cv2.rectangle(fb, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
            cv2.putText(fb, "fallback_face", (bx[0], max(12, bx[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(outdir, "02c_fallback_face_overlay.png"), fb)

    _draw_overlay(warped, list(merged.values()), os.path.join(outdir, "02_overlay.png"))

    # --- OCR (optional) ---
    get = lambda k: merged[k]["bbox_xyxy"] if k in merged else None
    ocr = {"first_name":"", "last_name":"", "address":"", "nid":"", "dob":"", "serial":""}
    nid_dec = {}

    if not no_ocr:
        if get("firstName") is not None:
            ocr["first_name"] = extract_arabic_field(warped, get("firstName"))
        if get("lastName") is not None:
            ocr["last_name"] = extract_arabic_field(warped, get("lastName"))
        if get("address") is not None:
            ocr["address"] = extract_address_field(warped, get("address"))
        if get("nid") is not None:
            ocr["nid"] = extract_digits_strict(warped, get("nid"), expected=14)
        if get("dob") is not None:
            ocr["dob"] = extract_date_digits(warped, get("dob"))
        if get("serial") is not None:
            x1s, y1s, x2s, y2s = map(int, get("serial"))
            ocr["serial"] = tesseract_text_simple(warped[y1s:y2s, x1s:x2s])

        if len(ocr["nid"]) == 14 and ocr["nid"].isdigit():
            nid_dec = decode_egyptian_id(ocr["nid"])

    # --- geometry pairs ---
    def norm(pt): return (pt[0]/CANONICAL_W, pt[1]/CANONICAL_H)
    def center(box):
        x1b, y1b, x2b, y2b = map(int, box)
        return ((x1b + x2b) / 2.0, (y1b + y2b) / 2.0)
    pairs = {}
    if get("front_logo") is not None and get("firstName") is not None:
        p, q = norm(center(get("front_logo"))), norm(center(get("firstName")))
        dx, dy = (q[0]-p[0]), (q[1]-p[1]); pairs["logo_to_name"] = dict(dx=dx, dy=dy, dist=float(np.hypot(dx, dy)))
    if get("dob") is not None and get("nid") is not None:
        p, q = norm(center(get("dob"))), norm(center(get("nid")))
        dx, dy = (q[0]-p[0]), (q[1]-p[1]); pairs["dob_to_nid"] = dict(dx=dx, dy=dy, dist=float(np.hypot(dx, dy)))
    if get("dob") is not None and get("serial") is not None:
        p, q = norm(center(get("dob"))), norm(center(get("serial")))
        dx, dy = (q[0]-p[0]), (q[1]-p[1]); pairs["dob_to_serial"] = dict(dx=dx, dy=dy, dist=float(np.hypot(dx, dy)))
    if get("lastName") is not None and get("address") is not None:
        p, q = norm(center(get("lastName"))), norm(center(get("address")))
        dx, dy = (q[0]-p[0]), (q[1]-p[1]); pairs["name_to_address"] = dict(dx=dx, dy=dy, dist=float(np.hypot(dx, dy)))

    # --- save overlays ---
    cli_equiv = img.copy()
    cv2.rectangle(cli_equiv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(outdir, "00_cli_equiv_full.png"), cli_equiv)

    boxes_canonical_first = {k: [int(x) for x in v["bbox_xyxy"]] for k, v in merged.items()}

    # Photo selection
    photo_label, photo_conf = None, None
    if "photo" in merged and "invalid_photo" in merged:
        if merged["invalid_photo"]["conf"] >= merged["photo"]["conf"]:
            photo_label = "invalid_photo"; photo_conf = float(merged["invalid_photo"]["conf"])
        else:
            photo_label = "photo"; photo_conf = float(merged["photo"]["conf"])
    elif "photo" in merged:
        photo_label = "photo"; photo_conf = float(merged["photo"]["conf"])
    elif "invalid_photo" in merged:
        photo_label = "invalid_photo"; photo_conf = float(merged["invalid_photo"]["conf"])

    validation = compute_validation(
        warped_bgr=warped,
        outdir=outdir,
        boxes=boxes_canonical_first,
        ar_ok=ar_ok,
        skew_deg=skew,
        pairs=pairs,
        photo_conf=photo_conf,
        photo_label=photo_label,
        photo_conf_ok=(photo_conf is None or photo_conf >= 0.50),
        calibration_path=calibration,
        thresholds_path=thresholds_path,
        require_photo=True,
        invalid_photo_fail_conf=float(invalid_photo_fail_conf),
        map_review_to_fail=bool(no_review),
        fail_if_geometry_bad=True,
    )

    out = {
        "image": image_path,
        "card_bbox_original": [int(v) for v in (x1, y1, x2, y2)],
        "homography_card_to_canonical": np.array(H_crop2canon).tolist(),
        "geometry": {"aspect_ratio": ar, "aspect_ratio_ok": ar_ok,
                     "skew_deg": skew, "skew_ok": (None if skew is None else abs(skew) < 3.0)},
        "detections_raw": raw_det,
        "detections_canonical_mapped": mapped_to_warped,
        "detections_canonical_direct": direct_det,
        "boxes_canonical_first": boxes_canonical_first,
        "geometry_pairs": pairs,
        "ocr": ocr,
        "nid_decoded": nid_dec,
        "validation": validation,
        "artifacts": {
            "cli_equiv_overlay": os.path.join(outdir, "00_cli_equiv_full.png"),
            "warped_path": os.path.join(outdir, "01_warped.png"),
            "raw_overlay": os.path.join(outdir, "02a_raw_overlay.png"),
            "direct_on_warp_overlay": os.path.join(outdir, "02b_direct_on_warp_overlay.png"),
            "overlay_path": os.path.join(outdir, "02_overlay.png"),
        }
    }
    if fallback_used:
        out["artifacts"]["fallback_face_overlay"] = os.path.join(outdir, "02c_fallback_face_overlay.png")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--card-weights", required=True)
    ap.add_argument("--obj-weights", required=True)
    ap.add_argument("--card-conf", type=float, default=0.10)
    ap.add_argument("--card-imgsz", type=int, default=640)
    ap.add_argument("--obj-conf", type=float, default=0.12)
    ap.add_argument("--obj-imgsz", type=int, default=1280)
    ap.add_argument("--paddle-det", type=str, default=None)
    ap.add_argument("--paddle-rec", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--thresholds", type=str, default=None)
    ap.add_argument("--invalid-photo-fail-conf", type=float, default=0.80)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR and PaddleOCR init")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    run(
        image_path=args.image,
        card_weights=args.card_weights,
        obj_weights=args.obj_weights,
        outdir=args.outdir,
        card_conf=args.card_conf,
        card_imgsz=args.card_imgsz,
        obj_conf=args.obj_conf,
        obj_imgsz=args.obj_imgsz,
        paddle_det=args.paddle_det,
        paddle_rec=args.paddle_rec,
        calibration=args.calibration,
        thresholds_path=args.thresholds,
        invalid_photo_fail_conf=args.invalid_photo_fail_conf,
        no_review=args.no_review,
        no_ocr=args.no_ocr,
    )
