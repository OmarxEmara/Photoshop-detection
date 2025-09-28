
# services/forensics.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

# Optional heavy deps (graceful fallback if missing)
try:
    import torch  # noqa: F401
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from sklearn.svm import OneClassSVM  # type: ignore
    import joblib  # type: ignore
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------- helpers ----------------------

def safe_crop(img: np.ndarray, xyxy, pad: int = 0) -> Optional[np.ndarray]:
    if img is None or xyxy is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def seam_margin_mask(shape_hw: Tuple[int, int] | np.ndarray, box, margin=8) -> np.ndarray:
    """Binary mask for a ring around the portrait box (the seam)."""
    if isinstance(shape_hw, np.ndarray):
        h, w = shape_hw[:2]
    else:
        h, w = shape_hw
    x1, y1, x2, y2 = map(int, box)
    m = np.zeros((h, w), np.uint8)
    x1o = max(0, x1 - margin); y1o = max(0, y1 - margin)
    x2o = min(w - 1, x2 + margin); y2o = min(h - 1, y2 + margin)
    m[y1o:y2o, x1o:x2o] = 1
    # inner hole
    m[y1:y2, x1:x2] = 0
    return m


# ----------------------------------------------------
# 1) Noiseprint / ManTra-Net (splice detectors)
#    - If Torch models are absent, we use a residual heuristic.
# ----------------------------------------------------

def _highfreq_residual_score(img_gray: np.ndarray) -> float:
    """Cheap fallback: high-frequency residual magnitude."""
    k = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]], dtype=np.float32)
    res = cv2.filter2D(img_gray.astype(np.float32), -1, k)
    return float(np.mean(np.abs(res)))

def noiseprint_or_mantra_score(roi_bgr: np.ndarray, seam_gray: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Returns {'mantra': s1, 'noiseprint': s2, 'residual': s3}
    Higher => more suspicious. Uses whatever is available.
    """
    out = {"mantra": 0.0, "noiseprint": 0.0, "residual": 0.0}
    if roi_bgr is None or roi_bgr.size == 0:
        return out

    g = to_gray(roi_bgr)

    # Cheap residual baseline (always available)
    out["residual"] = _highfreq_residual_score(g)

    # Seam boost (if provided): take residual on seam, use max of seam vs ROI
    if seam_gray is not None and seam_gray.size > 0:
        seam_res = _highfreq_residual_score(seam_gray)
        out["residual"] = max(out["residual"], seam_res)

    # TODO: plug real pretrained Torch models here; map logits to [0..1]
    if TORCH_OK:
        # Placeholders until models are wired
        out["mantra"] = 0.0
        out["noiseprint"] = 0.0

    return out


# ----------------------------------------------------
# 2) Local CLBP/LBP variance (portrait vs card)
# ----------------------------------------------------

def lbp_texture_zscores(card_bgr: np.ndarray, photo_xyxy) -> Dict[str, float]:
    """
    Compute texture complexity (LBP histogram variance & entropy) for
    portrait vs whole card. Returns z-scores: higher => more suspicious.
    Robust to tiny crops and dtype issues.
    """
    out = {"lbp_var_z": 0.0, "lbp_entropy_z": 0.0}
    if card_bgr is None or photo_xyxy is None:
        return out

    gray = to_gray(card_bgr)
    roi = safe_crop(gray, photo_xyxy)
    if roi is None or roi.size == 0:
        return out

    # Must have at least 3x3 to compute 8-neighborhood LBP
    if min(gray.shape[:2]) < 3 or min(roi.shape[:2]) < 3:
        return out

    def lbp_u8(img: np.ndarray) -> np.ndarray:
        """Classic 8-neighbor LBP, returns uint8 image (H-2, W-2)."""
        img = img.astype(np.uint8, copy=False)
        c = img[1:-1, 1:-1].astype(np.uint8)

        def b(mask: np.ndarray) -> np.uint8:
            return mask.astype(np.uint8)

        code = (
            (b(img[0:-2, 1:-1] > c) << 7) |
            (b(img[0:-2, 2:]   > c) << 6) |
            (b(img[1:-1, 2:]   > c) << 5) |
            (b(img[2:,   2:]   > c) << 4) |
            (b(img[2:,   1:-1] > c) << 3) |
            (b(img[2:,   0:-2] > c) << 2) |
            (b(img[1:-1, 0:-2] > c) << 1) |
            (b(img[0:-2, 0:-2] > c) << 0)
        ).astype(np.uint8)
        return code

    card_lbp = lbp_u8(gray)
    roi_lbp  = lbp_u8(roi)

    def stats_u8(x: np.ndarray) -> tuple[float, float]:
        """Histogram-based variance & entropy from uint8 LBP image."""
        if x.size == 0:
            return 0.0, 0.0
        # Use numpy (works regardless of OpenCV build)
        hist = np.bincount(x.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return 0.0, 0.0
        p = hist / total
        var = float(np.var(p))
        nz = p > 0
        ent = float(-(p[nz] * np.log2(p[nz])).sum())
        return var, ent

    var_card, ent_card = stats_u8(card_lbp)
    var_roi,  ent_roi  = stats_u8(roi_lbp)

    # Guard against degenerate denominators
    denom_var = np.sqrt(max(var_card, 1e-9))
    denom_ent = np.sqrt(max(abs(ent_card), 1e-9))

    lbp_var_z = float((var_roi - var_card) / denom_var)
    lbp_entropy_z = float((ent_roi - ent_card) / denom_ent)

    out["lbp_var_z"] = lbp_var_z
    out["lbp_entropy_z"] = lbp_entropy_z
    return out



# ----------------------------------------------------
# 3) Copy-move detection (self-matching via ORB)
# ----------------------------------------------------

def copy_move_score(card_gray: np.ndarray, min_matches=30, eps_px=3) -> float:
    """
    Returns a scalar score; > ~0.6 suggests duplicated content.
    - ORB features, match to themselves (cross-check),
    - filter near neighbors (within eps_px),
    - cluster displacement vectors by angle x magnitude; dominant ratio => score.
    """
    if card_gray is None or card_gray.size == 0:
        return 0.0

    gray = card_gray if card_gray.ndim == 2 else to_gray(card_gray)

    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
    kpts, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(kpts) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc, desc)

    disp = []
    for m in matches:
        if m.queryIdx == m.trainIdx:
            continue
        p = np.array(kpts[m.queryIdx].pt)
        q = np.array(kpts[m.trainIdx].pt)
        d = q - p
        if np.linalg.norm(d) < eps_px:  # ignore tiny locality
            continue
        disp.append(d)

    if len(disp) < min_matches:
        return 0.0

    disp = np.array(disp, dtype=np.float32)
    ang = (np.degrees(np.arctan2(disp[:, 1], disp[:, 0])) + 360.0) % 360.0
    mag = np.linalg.norm(disp, axis=1)

    # 2D histogram over (angle, magnitude)
    H, _, _ = np.histogram2d(ang, mag, bins=(36, 30),
                             range=[[0, 360], [0, max(50, float(mag.max()))]])
    peak = H.max()
    total = H.sum() + 1e-9
    ratio = float(peak / total)  # ~[0..1]
    return ratio


# ----------------------------------------------------
# 4) CFA / PRNU inconsistency (cheap demosaic residuals)
# ----------------------------------------------------

def cfa_inconsistency(card_bgr: np.ndarray, photo_xyxy) -> Dict[str, float]:
    """
    Compute demosaicing residual map and compare inside vs outside portrait.
    Returns z-score of residual mean and std; higher => more suspicious.
    """
    if card_bgr is None or photo_xyxy is None:
        return {"cfa_mean_z": 0.0, "cfa_std_z": 0.0}

    g = to_gray(card_bgr).astype(np.float32)

    # Predict green by a simple cross bilinear; residual = |G - G_pred|
    kernel = np.array([[0.0, 0.25, 0.0],
                       [0.25, 0.0, 0.25],
                       [0.0, 0.25, 0.0]], np.float32)
    g_pred = cv2.filter2D(g, -1, kernel)
    res = np.abs(g - g_pred)

    # Split inside vs outside portrait
    mask = np.zeros_like(res, np.uint8)
    x1, y1, x2, y2 = map(int, photo_xyxy)
    mask[y1:y2, x1:x2] = 1
    inside = res[mask == 1]
    outside = res[mask == 0]

    mi, si = float(inside.mean() + 1e-9), float(inside.std() + 1e-9)
    mo, so = float(outside.mean() + 1e-9), float(outside.std() + 1e-9)

    mean_z = (mi - mo) / (so + 1e-9)
    std_z  = (si - so) / (so + 1e-9)
    return {"cfa_mean_z": float(mean_z), "cfa_std_z": float(std_z)}


# ----------------------------------------------------
# 5) Font forensics (one-class SVM on glyph metrics)
#    - Runtime scorer (training is external/offline).
# ----------------------------------------------------

def _prep_binary(img_bgr: np.ndarray) -> np.ndarray:
    g = to_gray(img_bgr)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return th

def _glyph_features(th_bin: np.ndarray) -> np.ndarray:
    """
    Extract glyph metrics from a binarized region:
      - mean/std glyph width/height
      - mean/std glyph area (log1p)
      - stroke density proxy (skeleton)
      - mean inter-glyph gap (log1p)
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th_bin, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    if areas.size == 0:
        return np.zeros((1, 8), dtype=np.float32)

    # Skeleton-based stroke density
    skel = th_bin.copy()
    size = float(np.size(skel))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel_count = 0
    while True:
        opened = cv2.morphologyEx(skel, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(skel, opened)
        eroded = cv2.erode(skel, element)
        skel = eroded.copy()
        skel_count += int(np.count_nonzero(temp))
        if cv2.countNonZero(skel) == 0:
            break
    stroke_density = skel_count / (size + 1e-9)

    # Inter-glyph spacing (project to X)
    proj = (255 - th_bin).sum(axis=0)
    edges = (proj > 0).astype(np.uint8)
    gaps = []
    in_run = False; last_end = 0
    for i, v in enumerate(edges):
        if v and not in_run:
            in_run = True
            if i - last_end > 0:
                gaps.append(i - last_end)
        if not v and in_run:
            in_run = False
            last_end = i
    gaps = np.array(gaps, dtype=np.float32) if gaps else np.array([0.0], dtype=np.float32)

    feat = np.array([
        float(np.mean(widths)), float(np.std(widths) + 1e-9),
        float(np.mean(heights)), float(np.std(heights) + 1e-9),
        float(np.mean(areas)), float(np.std(areas) + 1e-9),
        stroke_density,
        float(np.mean(gaps)),
    ], dtype=np.float32).reshape(1, -1)

    # log-scale heavy-tailed stats
    feat[:, 4:6] = np.log1p(feat[:, 4:6])
    feat[:, 7:8] = np.log1p(feat[:, 7:8])
    return feat

def font_od_score(card_bgr: np.ndarray,
                  nid_xyxy,
                  serial_xyxy,
                  ocsvm_path: Optional[str]) -> Dict[str, float]:
    """
    Load a One-Class SVM (trained offline on genuine glyph metrics).
    Return negative decision_function => out-of-distribution (suspicious).
    """
    scores = {"font_nid_df": 0.0, "font_serial_df": 0.0, "ok": False}
    if not SKLEARN_OK or not ocsvm_path or not os.path.exists(ocsvm_path):
        return scores

    try:
        svm: OneClassSVM = joblib.load(ocsvm_path)  # type: ignore
    except Exception:
        return scores

    scores["ok"] = True

    # NID block
    if nid_xyxy is not None:
        crop = safe_crop(card_bgr, nid_xyxy, pad=2)
        if crop is not None and crop.size > 0:
            th = _prep_binary(crop)
            feat = _glyph_features(th)
            scores["font_nid_df"] = float(svm.decision_function(feat)[0])

    # Serial block
    if serial_xyxy is not None:
        crop = safe_crop(card_bgr, serial_xyxy, pad=2)
        if crop is not None and crop.size > 0:
            th = _prep_binary(crop)
            feat = _glyph_features(th)
            scores["font_serial_df"] = float(svm.decision_function(feat)[0])

    return scores
