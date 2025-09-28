from __future__ import annotations
import os, json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2

# Threshold loader (data-driven)
from services.thresholds import load_thresholds

# Optional advanced forensics (safe imports; functions should return None/0 if unavailable)
try:
    from services.forensics import (
        residual_anomaly_score,         # portrait/seam residual (cheap splice proxy)
        lbp_texture_zscores,            # (var_z, ent_z) inside portrait vs card
        copy_move_ratio,                # duplicated patches score in card
        cfa_mismatch_zscores,           # (mean_z, std_z) demosaic residual mismatch
        font_distance_scores,           # (nid_df, serial_df, model_ok)
    )
except Exception:
    residual_anomaly_score = None
    lbp_texture_zscores = None
    copy_move_ratio = None
    cfa_mismatch_zscores = None
    font_distance_scores = None


# ---------------- ELA ----------------

def _compute_ela_heatmap(bgr: np.ndarray, quality: int = 95) -> Tuple[np.ndarray, float, float]:
    """
    Error Level Analysis: re-encode as JPEG, diff, summarize.
    """
    tmp_path = "__tmp_ela.jpg"
    cv2.imwrite(tmp_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    comp = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    if comp is None:
        comp = bgr.copy()
    diff = cv2.absdiff(bgr, comp)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vis = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return vis, float(gray.mean()), float(gray.max())


# -------------- JPEG GRID --------------

def _compute_jpeg_grid_strength(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    8x8 JPEG blocking energy (robust, no broadcasting issues).
    Return heatmap and std().
    """
    h, w = gray.shape[:2]
    grayf = gray.astype(np.float32)

    gx = np.zeros_like(grayf)
    for k in range(8, w, 8):
        k0 = max(0, k - 1)
        k1 = min(w - 1, k)
        gx[:, k0] = np.abs(grayf[:, k0] - grayf[:, k1])

    gy = np.zeros_like(grayf)
    for k in range(8, h, 8):
        k0 = max(0, k - 1)
        k1 = min(h - 1, k)
        gy[k0, :] = np.abs(grayf[k0, :] - grayf[k1, :])

    heat = np.clip(gx + gy, 0, 255).astype(np.uint8)
    return heat, float(heat.std())


# -------------- ILLUMINATION ------------

def _illumination_map(gray: np.ndarray, sigma: int = 21) -> Tuple[np.ndarray, float, float]:
    """
    Smooth/low-freq illumination map to quantify uneven lighting.
    Returns map, mean, std.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blur, float(blur.mean()), float(blur.std())


# -------------- PHOTO GEOMETRY (CALIBRATED) ------------

def _load_calibration(calib_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not calib_path:
        return None
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def _photo_geometry_checks(
    photo_xyxy: Optional[Tuple[int, int, int, int]],
    canvas_w: int,
    canvas_h: int,
    calib: Optional[Dict[str, Any]],
    thr_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if photo_xyxy is None:
        return None

    x1, y1, x2, y2 = map(int, photo_xyxy)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = w / float(h)
    area_ratio = (w * h) / float(canvas_w * canvas_h)

    # Fallback bands:
    ar_lo, ar_hi = 0.68, 0.78
    area_lo, area_hi = 0.045, 0.10

    if calib:
        try:
            ar_p05 = float(calib["photo_ar"]["p05"])
            ar_p95 = float(calib["photo_ar"]["p95"])
            area_p05 = float(calib["photo_area"]["p05"])
            area_p95 = float(calib["photo_area"]["p95"])
            pad_ar = 0.02
            pad_area = 0.01
            ar_lo, ar_hi = ar_p05 - pad_ar, ar_p95 + pad_ar
            area_lo, area_hi = max(0.0, area_p05 - pad_area), area_p95 + pad_area
        except Exception:
            pass
    elif thr_cfg:
        g = thr_cfg.get("geometry", {})
        pad_ar = float(g.get("pad_ar", 0.02))
        pad_area = float(g.get("pad_area", 0.01))
        ar_lo = float(g.get("ar_lo", ar_lo)) - pad_ar
        ar_hi = float(g.get("ar_hi", ar_hi)) + pad_ar
        area_lo = max(0.0, float(g.get("area_lo", area_lo)) - pad_area)
        area_hi = float(g.get("area_hi", area_hi)) + pad_area

    return {
        "width": w,
        "height": h,
        "ar": ar,
        "area": area_ratio,
        "ar_ok": (ar_lo <= ar <= ar_hi),
        "area_ok": (area_lo <= area_ratio <= area_hi),
        "bands": {
            "ar_lo": ar_lo, "ar_hi": ar_hi,
            "area_lo": area_lo, "area_hi": area_hi,
        }
    }


# -------------- MAIN ENTRY --------------

def compute_validation(
    warped_bgr: np.ndarray,
    outdir: str,
    boxes: Dict[str, Any],
    ar_ok: bool,
    skew_deg: Optional[float],
    pairs: Dict[str, Any],
    # photo detector metadata (optional)
    photo_conf: Optional[float] = None,
    photo_label: Optional[str] = None,
    photo_conf_ok: bool = True,
    # calibration file for geometry ranges
    calibration_path: Optional[str] = None,
    # thresholds config (data-driven)
    thresholds_path: Optional[str] = None,
    # --- PRODUCTION POLICY KNOBS ---
    require_photo: bool = True,             # if True and no photo detected -> FAIL
    invalid_photo_fail_conf: float = 0.80,  # high-confidence invalid_photo => FAIL
    map_review_to_fail: bool = True,        # never return "review"; map to FAIL
    fail_if_geometry_bad: bool = True,      # outside geometry bands => FAIL (can relax to soft)
) -> Dict[str, Any]:

    os.makedirs(outdir, exist_ok=True)
    H, W = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # Load thresholds / weights / hard-fail from JSON (or defaults)
    thr_cfg = load_thresholds(thresholds_path)
    weights = thr_cfg.get("weights", {"photo": 0.55, "ela_global": 0.15, "jpeg_global": 0.15, "illum": 0.05, "adv_forensics": 0.10})
    score_threshold = float(thr_cfg.get("score_threshold", 0.60))
    hf = thr_cfg.get("hard_fail", {})
    soft = thr_cfg.get("soft_starts", {})

    # --- ELA
    ela_heat, ela_mean, ela_max = _compute_ela_heatmap(warped_bgr)
    ela_path = os.path.join(outdir, "03_ela.png")
    cv2.imwrite(ela_path, ela_heat)

    # --- JPEG grid
    jpeg_heat, jpeg_std = _compute_jpeg_grid_strength(gray)
    jpeg_path = os.path.join(outdir, "04_jpeggrid.png")
    cv2.imwrite(jpeg_path, jpeg_heat)

    # --- Illumination
    illum_map, illum_mean, illum_std = _illumination_map(gray)
    illum_path = os.path.join(outdir, "05_illum.png")
    cv2.imwrite(illum_path, cv2.normalize(illum_map, None, 0, 255, cv2.NORM_MINMAX))

    # --- global geometry sanity (from warp stage)
    geometry = {
        "aspect_ratio_ok": bool(ar_ok),
        "skew_ok": (skew_deg is None or abs(float(skew_deg)) < 3.0),
        "relative_layout_ok": True,  # reserved for future template checks
    }

    # --- Photo region / geometry
    photo_xyxy = boxes.get("photo") or boxes.get("invalid_photo")
    calib = _load_calibration(calibration_path)
    photo_geom = _photo_geometry_checks(photo_xyxy, W, H, calib, thr_cfg=thr_cfg)

    # --- Local anomalies inside the photo region
    photo_ela_z = None
    photo_jpeg_z = None
    if photo_xyxy is not None:
        x1, y1, x2, y2 = map(int, photo_xyxy)
        crop = warped_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            _, p_ela_mean, _ = _compute_ela_heatmap(crop)
            _, p_jpeg_std = _compute_jpeg_grid_strength(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            # Normalize vs. global
            photo_ela_z = float(max(0.0, (p_ela_mean - ela_mean)))
            photo_jpeg_z = float(max(0.0, (p_jpeg_std - jpeg_std)))

    # --------- Advanced Forensics (optional; safe fallbacks) ----------
    forensic: Dict[str, Any] = {
        "mantra": None, "noiseprint": None,
        "residual": None,
        "lbp_var_z": None, "lbp_entropy_z": None,
        "copy_move_ratio": 0.0,
        "cfa_mean_z": None, "cfa_std_z": None,
        "font_nid_df": None, "font_serial_df": None, "font_model_ok": False
    }
    # residual seam / portrait-only
    if residual_anomaly_score is not None and photo_xyxy is not None:
        try:
            forensic["residual"] = float(residual_anomaly_score(warped_bgr, photo_xyxy) or 0.0)
        except Exception:
            forensic["residual"] = None

    # LBP texture Zs
    if lbp_texture_zscores is not None and photo_xyxy is not None:
        try:
            v, e = lbp_texture_zscores(warped_bgr, photo_xyxy)
            forensic["lbp_var_z"] = float(v) if v is not None else None
            forensic["lbp_entropy_z"] = float(e) if e is not None else None
        except Exception:
            pass

    # Copy-move (inside card)
    if copy_move_ratio is not None:
        try:
            forensic["copy_move_ratio"] = float(copy_move_ratio(warped_bgr) or 0.0)
        except Exception:
            forensic["copy_move_ratio"] = 0.0

    # CFA mismatch z-scores
    if cfa_mismatch_zscores is not None and photo_xyxy is not None:
        try:
            mz, sz = cfa_mismatch_zscores(warped_bgr, photo_xyxy)
            forensic["cfa_mean_z"] = float(mz) if mz is not None else None
            forensic["cfa_std_z"]  = float(sz) if sz is not None else None
        except Exception:
            pass

    # Font distances (OCSVM)
    if font_distance_scores is not None:
        try:
            nd, sd, ok = font_distance_scores(warped_bgr, boxes)
            forensic["font_nid_df"] = None if nd is None else float(nd)
            forensic["font_serial_df"] = None if sd is None else float(sd)
            forensic["font_model_ok"] = bool(ok)
        except Exception:
            forensic["font_model_ok"] = False

    # --------- Flags ----------
    ela_suspicious = (ela_mean > (hf.get("ela_mean", 15.0)*0.8)) or (ela_max > (hf.get("ela_max", 120.0)*0.8))
    jpeg_suspicious = (jpeg_std > (hf.get("jpeg_std", 30.0)*0.7))
    illum_suspicious = (illum_std > (soft.get("illum_std", 9999))) if soft.get("illum_std") else False

    # Photo suspicious?
    photo_suspicious = False
    if photo_xyxy is not None:
        label_is_invalid = (photo_label == "invalid_photo")
        geom_bad = (photo_geom is not None and (not photo_geom["ar_ok"] or not photo_geom["area_ok"]))
        ela_anom = (photo_ela_z is not None and photo_ela_z > (hf.get("photo_ela_z", 0.50)*0.6))
        jpeg_anom = (photo_jpeg_z is not None and photo_jpeg_z > (hf.get("photo_jpeg_z", 0.70)*0.5))
        photo_suspicious = label_is_invalid or geom_bad or ela_anom or jpeg_anom

    metrics = {
        "ela_mean": ela_mean,
        "ela_max": ela_max,
        "ela_path": ela_path,
        "jpeg_grid_std": jpeg_std,
        "jpeg_grid_path": jpeg_path,
        "illum_mean": illum_mean,
        "illum_std": illum_std,
        "illum_path": illum_path,
        "name_line_angle_mean": pairs.get("name_line_angle_mean", 0.0) if isinstance(pairs, dict) else 0.0,
        "name_line_angle_std": pairs.get("name_line_angle_std", 0.0) if isinstance(pairs, dict) else 0.0,
        "addr_line_angle_mean": pairs.get("addr_line_angle_mean", 0.0) if isinstance(pairs, dict) else 0.0,
        "addr_line_angle_std": pairs.get("addr_line_angle_std", 0.0) if isinstance(pairs, dict) else 0.0,
        "geometry": {
            "aspect_ratio_ok": bool(ar_ok),
            "skew_ok": (skew_deg is None or abs(float(skew_deg)) < 3.0),
            "relative_layout_ok": True
        },
        "photo_box": list(map(int, photo_xyxy)) if photo_xyxy is not None else None,
        "photo_label": photo_label,
        "photo_conf_ok": bool(photo_conf_ok),
        "photo_ela_z": photo_ela_z,
        "photo_jpeg_z": photo_jpeg_z,
        "photo_geometry": photo_geom,
        "forensic": forensic,
    }

    flags = {
        "ela_suspicious": bool(ela_suspicious),
        "jpeg_grid_suspicious": bool(jpeg_suspicious),
        "illum_suspicious": bool(illum_suspicious),
        "photo_suspicious": bool(photo_suspicious),
        "geometry_ok": bool(metrics["geometry"]["aspect_ratio_ok"] and metrics["geometry"]["skew_ok"] and metrics["geometry"]["relative_layout_ok"]),
    }

    # ------- Scoring (kept for telemetry), with adv_forensics bucket -------
    comp = {
        "photo_norm": 1.0 if photo_suspicious else 0.0,
        "ela_norm_global": min(1.0, max(0.0, (ela_mean - 3.0) / 12.0)),
        "jpeg_norm_global": min(1.0, jpeg_std / 30.0),
        "illum_norm": min(1.0, (illum_std if illum_mean <= 1 else illum_std / 60.0)),
        "adv_forensics_norm": 0.0,
    }

    # include residual / cfa / lbp / copy-move into adv_forensics_norm via soft starts
    def ramp(x, start, end):
        if x is None: return 0.0
        if x <= start: return 0.0
        if x >= end: return 1.0
        return (x - start) / max(1e-6, (end - start))

    # choose start from p95 (soft) and end from hard-fail p99 (hf)
    af = 0.0; denom = 0.0
    for key, soft_name, hard_name in [
        ("residual", "residual", "residual"),
        ("cfa_std_z", "cfa_std_z", "cfa_std_z"),
        ("lbp_var_z", "lbp_var_z", "lbp_var_z"),
        ("copy_move_ratio", "copy_move", "copy_move")
    ]:
        x = forensic.get(key)
        s = soft.get(soft_name, None)
        h = hf.get(hard_name, None)
        if (x is not None) and (s is not None) and (h is not None) and (h > s):
            af += ramp(float(x), float(s), float(h))
            denom += 1.0
    if denom > 0:
        comp["adv_forensics_norm"] = min(1.0, af / denom)

    fraud_score = (
        weights["photo"] * comp["photo_norm"]
        + weights["ela_global"] * comp["ela_norm_global"]
        + weights["jpeg_global"] * comp["jpeg_norm_global"]
        + weights["illum"] * comp["illum_norm"]
        + weights["adv_forensics"] * comp["adv_forensics_norm"]
    )

    # ----------------- PRODUCTION, NO-REVIEW POLICY -----------------
    hard_reasons = []
    decision = None

    # 1) Require a detected portrait box on the card
    if require_photo and (metrics["photo_box"] is None):
        decision = "fail"
        hard_reasons.append("Photo region not detected")

    # 2) High-confidence invalid_photo -> FAIL
    if decision is None and photo_label == "invalid_photo" and (photo_conf or 0.0) >= invalid_photo_fail_conf:
        decision = "fail"
        hard_reasons.append("Photo region flagged invalid with high confidence")

    # 3) Geometry outside calibrated bands -> FAIL (keep hard by default)
    if decision is None and fail_if_geometry_bad and (photo_geom is not None) and (not photo_geom["ar_ok"] or not photo_geom["area_ok"]):
        decision = "fail"
        if not photo_geom["area_ok"]:
            hard_reasons.append("Photo geometry out of expected range (area)")
        if not photo_geom["ar_ok"]:
            hard_reasons.append("Photo geometry out of expected range (aspect ratio)")

    # 4) Strong global forensic anomalies -> FAIL
    if decision is None:
        if ela_mean > hf.get("ela_mean", 15.0) or ela_max > hf.get("ela_max", 120.0):
            decision = "fail"; hard_reasons.append("Global ELA suspicious")
        elif jpeg_std > hf.get("jpeg_std", 30.0):
            decision = "fail"; hard_reasons.append("Inconsistent JPEG grid")

    # 5) Strong local anomalies in the photo box -> FAIL
    if decision is None and metrics["photo_box"] is not None:
        if (photo_ela_z or 0.0) > hf.get("photo_ela_z", 0.50):
            decision = "fail"; hard_reasons.append("Photo ELA anomaly vs. card")
        elif (photo_jpeg_z or 0.0) > hf.get("photo_jpeg_z", 0.70):
            decision = "fail"; hard_reasons.append("Photo JPEG-grid anomaly vs. card")

    # 6) Advanced forensics decisive thresholds (hard)
    if decision is None and metrics["photo_box"] is not None:
        if (forensic.get("residual") or 0.0) > hf.get("residual", 18.0):
            decision = "fail"; hard_reasons.append("Portrait/seam residual anomaly")
        elif (forensic.get("cfa_std_z") or 0.0) > hf.get("cfa_std_z", 1.25):
            decision = "fail"; hard_reasons.append("CFA residual mismatch in portrait region")
        elif (forensic.get("copy_move_ratio") or 0.0) > hf.get("copy_move", 0.15):
            decision = "fail"; hard_reasons.append("Copy-move duplication detected")

    # 7) If none of the hard rules fired, use fraud_score with one threshold
    if decision is None:
        decision = "fail" if fraud_score >= score_threshold else "pass"

    # Build human-readable reasons if no hard reasons added but we failed by score
    reasons: list[str] = []
    reasons.extend(hard_reasons)
    if not reasons:
        if decision == "fail":
            if photo_suspicious:
                if photo_label == "invalid_photo":
                    reasons.append("Photo region flagged as invalid by detector")
                if (photo_ela_z or 0) > hf.get("photo_ela_z", 0.50):
                    reasons.append("Photo ELA anomaly vs. card")
                if (photo_jpeg_z or 0) > hf.get("photo_jpeg_z", 0.70):
                    reasons.append("Photo JPEG-grid anomaly vs. card")
                if photo_geom is not None and not photo_geom["area_ok"]:
                    reasons.append("Photo geometry out of expected range (area)")
                if photo_geom is not None and not photo_geom["ar_ok"]:
                    reasons.append("Photo geometry out of expected range (aspect ratio)")
            if ela_suspicious: reasons.append("Global ELA suspicious")
            if jpeg_suspicious: reasons.append("Inconsistent JPEG grid")
            if illum_suspicious: reasons.append("Uneven illumination")
            if not metrics["geometry"]["skew_ok"] or not metrics["geometry"]["aspect_ratio_ok"]:
                reasons.append("Geometry not ideal (angle/AR)")
            if (forensic.get("residual") or 0.0) > (soft.get("residual", 9e9)):
                reasons.append("Portrait/seam residual elevated")
            if (forensic.get("copy_move_ratio") or 0.0) > (soft.get("copy_move", 9e9)):
                reasons.append("Copy-move similarity elevated")
        else:
            if illum_suspicious:
                reasons.append("Uneven illumination (quality issue)")

    return {
        "metrics": metrics,
        "flags": flags,
        "artifacts": {
            "ela_heatmap": ela_path,
            "jpeg_grid_heatmap": jpeg_path,
            "illum_heatmap": illum_path,
            "photo_seam_overlay": None,
        },
        "fraud_score": float(fraud_score),
        "verdict": {
            "decision": decision,
            "fraud_score": float(fraud_score),
            "thresholds": {"auto_fail": score_threshold},
            "weights": weights,
            "components": comp,
            "reasons": reasons,
        }
    }
