"""
Image Forensics Module for ID Card Tampering Detection

Provides multiple forensic analysis techniques:
1. Splice detection (Noiseprint/ManTra-Net with fallback)
2. Texture analysis (LBP variance comparison)
3. Copy-move detection (ORB feature matching)
4. CFA inconsistency detection
5. Font forensics (One-Class SVM on glyph metrics)
"""
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

# Optional dependencies with graceful degradation
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sklearn.svm import OneClassSVM  # type: ignore
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Constants
DEFAULT_SEAM_MARGIN_PIXELS = 8
MIN_COPY_MOVE_MATCHES = 30
COPY_MOVE_EPSILON_PIXELS = 3
ORB_MAX_FEATURES = 4000
ORB_SCALE_FACTOR = 1.2
ORB_PYRAMID_LEVELS = 8
ANGLE_HISTOGRAM_BINS = 36
MAGNITUDE_HISTOGRAM_BINS = 30
MAGNITUDE_MAX_DEFAULT = 50
BINARIZATION_UPSCALE_FACTOR = 2.0
MORPHOLOGY_KERNEL_SIZE = (2, 2)
SKELETON_KERNEL_SIZE = (3, 3)
MIN_IMAGE_SIZE_FOR_LBP = 3
LBP_HISTOGRAM_SIZE = 256
EPSILON_SMALL = 1e-9


# ============================================================================
# Image Utilities
# ============================================================================

def safe_crop_region(
    image: np.ndarray, 
    bbox_xyxy: Tuple[int, int, int, int], 
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Safely crop region from image with optional padding.
    Returns None if crop is invalid or empty.
    """
    if image is None or bbox_xyxy is None:
        return None
    
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    
    # Apply padding with boundary clipping
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width - 1, x2 + padding)
    y2 = min(height - 1, y2 + padding)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]


def convert_to_grayscale(bgr_image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale, passthrough if already gray."""
    if bgr_image.ndim == 3:
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return bgr_image


def create_seam_margin_mask(
    image_shape: Tuple[int, int] | np.ndarray,
    inner_box: Tuple[int, int, int, int],
    margin_pixels: int = DEFAULT_SEAM_MARGIN_PIXELS
) -> np.ndarray:
    """
    Create binary mask for ring around a bounding box (the seam region).
    The mask is 1 in the margin ring, 0 everywhere else.
    """
    if isinstance(image_shape, np.ndarray):
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    
    x1, y1, x2, y2 = map(int, inner_box)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Outer rectangle (with margin)
    x1_outer = max(0, x1 - margin_pixels)
    y1_outer = max(0, y1 - margin_pixels)
    x2_outer = min(width - 1, x2 + margin_pixels)
    y2_outer = min(height - 1, y2 + margin_pixels)
    
    mask[y1_outer:y2_outer, x1_outer:x2_outer] = 1
    
    # Cut out inner rectangle (creating ring)
    mask[y1:y2, x1:x2] = 0
    
    return mask


# ============================================================================
# 1. Splice Detection (Noiseprint/ManTra-Net)
# ============================================================================

def compute_highfreq_residual_score(grayscale_image: np.ndarray) -> float:
    """
    Compute high-frequency residual magnitude using Laplacian kernel.
    This serves as a cheap fallback when deep learning models are unavailable.
    """
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    residual = cv2.filter2D(
        grayscale_image.astype(np.float32), 
        -1, 
        laplacian_kernel
    )
    
    return float(np.mean(np.abs(residual)))


def compute_splice_detection_scores(
    roi_bgr: np.ndarray,
    seam_grayscale: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute splice detection scores using available methods.
    Returns dictionary with 'mantra', 'noiseprint', and 'residual' scores.
    Higher scores indicate more suspicious regions.
    
    Args:
        roi_bgr: Region of interest in BGR format
        seam_grayscale: Optional seam region in grayscale for enhanced detection
    
    Returns:
        Dictionary of splice detection scores
    """
    scores = {
        "mantra": 0.0,
        "noiseprint": 0.0,
        "residual": 0.0
    }
    
    if roi_bgr is None or roi_bgr.size == 0:
        return scores
    
    grayscale = convert_to_grayscale(roi_bgr)
    
    # Baseline residual score (always available)
    scores["residual"] = compute_highfreq_residual_score(grayscale)
    
    # Boost score using seam region if provided
    if seam_grayscale is not None and seam_grayscale.size > 0:
        seam_residual_score = compute_highfreq_residual_score(seam_grayscale)
        scores["residual"] = max(scores["residual"], seam_residual_score)
    
    # Deep learning models (placeholder for future implementation)
    if TORCH_AVAILABLE:
        # TODO: Integrate pretrained ManTra-Net and Noiseprint models
        # Map logits to [0, 1] range for consistency
        scores["mantra"] = 0.0
        scores["noiseprint"] = 0.0
    
    return scores


# ============================================================================
# 2. Texture Analysis (LBP)
# ============================================================================

def compute_lbp_image(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Compute classic 8-neighbor Local Binary Pattern (LBP).
    Returns LBP-encoded image with dimensions (H-2, W-2).
    """
    image_uint8 = grayscale_image.astype(np.uint8, copy=False)
    center = image_uint8[1:-1, 1:-1].astype(np.uint8)
    
    def to_binary(comparison_mask: np.ndarray) -> np.ndarray:
        return comparison_mask.astype(np.uint8)
    
    # Build 8-bit LBP code by comparing neighbors to center
    lbp_code = (
        (to_binary(image_uint8[0:-2, 1:-1] > center) << 7) |  # Top
        (to_binary(image_uint8[0:-2, 2:] > center) << 6) |    # Top-right
        (to_binary(image_uint8[1:-1, 2:] > center) << 5) |    # Right
        (to_binary(image_uint8[2:, 2:] > center) << 4) |      # Bottom-right
        (to_binary(image_uint8[2:, 1:-1] > center) << 3) |    # Bottom
        (to_binary(image_uint8[2:, 0:-2] > center) << 2) |    # Bottom-left
        (to_binary(image_uint8[1:-1, 0:-2] > center) << 1) |  # Left
        (to_binary(image_uint8[0:-2, 0:-2] > center) << 0)    # Top-left
    ).astype(np.uint8)
    
    return lbp_code


def compute_lbp_statistics(lbp_image: np.ndarray) -> Tuple[float, float]:
    """
    Compute histogram-based variance and entropy from LBP-encoded image.
    
    Returns:
        Tuple of (variance, entropy)
    """
    if lbp_image.size == 0:
        return 0.0, 0.0
    
    # Compute histogram of LBP codes
    histogram = np.bincount(
        lbp_image.ravel(), 
        minlength=LBP_HISTOGRAM_SIZE
    ).astype(np.float64)
    
    total_pixels = histogram.sum()
    if total_pixels <= 0:
        return 0.0, 0.0
    
    # Normalize to probability distribution
    probabilities = histogram / total_pixels
    
    # Variance of probability distribution
    variance = float(np.var(probabilities))
    
    # Entropy calculation (skip zero probabilities)
    nonzero_mask = probabilities > 0
    entropy = float(-(probabilities[nonzero_mask] * np.log2(probabilities[nonzero_mask])).sum())
    
    return variance, entropy


def compute_lbp_texture_zscores(
    card_bgr: np.ndarray,
    photo_bbox_xyxy: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Compare texture complexity between portrait region and full card.
    Uses LBP histogram variance and entropy as texture descriptors.
    
    Returns z-scores where higher values indicate more suspicious differences.
    Robust to small crops and type conversion issues.
    """
    result = {
        "lbp_var_z": 0.0,
        "lbp_entropy_z": 0.0
    }
    
    if card_bgr is None or photo_bbox_xyxy is None:
        return result
    
    grayscale = convert_to_grayscale(card_bgr)
    photo_roi = safe_crop_region(grayscale, photo_bbox_xyxy)
    
    if photo_roi is None or photo_roi.size == 0:
        return result
    
    # Require minimum image size for 8-neighbor LBP
    if (min(grayscale.shape[:2]) < MIN_IMAGE_SIZE_FOR_LBP or 
        min(photo_roi.shape[:2]) < MIN_IMAGE_SIZE_FOR_LBP):
        return result
    
    # Compute LBP for full card and portrait region
    card_lbp = compute_lbp_image(grayscale)
    roi_lbp = compute_lbp_image(photo_roi)
    
    # Extract statistics
    card_variance, card_entropy = compute_lbp_statistics(card_lbp)
    roi_variance, roi_entropy = compute_lbp_statistics(roi_lbp)
    
    # Compute z-scores with safe denominators
    variance_denominator = np.sqrt(max(card_variance, EPSILON_SMALL))
    entropy_denominator = np.sqrt(max(abs(card_entropy), EPSILON_SMALL))
    
    variance_zscore = float((roi_variance - card_variance) / variance_denominator)
    entropy_zscore = float((roi_entropy - card_entropy) / entropy_denominator)
    
    result["lbp_var_z"] = variance_zscore
    result["lbp_entropy_z"] = entropy_zscore
    
    return result


# ============================================================================
# 3. Copy-Move Detection
# ============================================================================

def detect_copy_move_manipulation(
    card_grayscale: np.ndarray,
    min_matches: int = MIN_COPY_MOVE_MATCHES,
    epsilon_pixels: int = COPY_MOVE_EPSILON_PIXELS
) -> float:
    """
    Detect copy-move forgery by finding duplicated content within the image.
    
    Method:
    1. Extract ORB features and match them to themselves (cross-check)
    2. Filter out near-neighbor matches (within epsilon_pixels)
    3. Cluster displacement vectors by angle and magnitude
    4. Return ratio of dominant cluster to total matches
    
    Returns:
        Score in [0, 1] where > ~0.6 suggests duplicated content
    """
    if card_grayscale is None or card_grayscale.size == 0:
        return 0.0
    
    grayscale = (card_grayscale if card_grayscale.ndim == 2 
                 else convert_to_grayscale(card_grayscale))
    
    # Detect ORB features
    orb = cv2.ORB_create(
        nfeatures=ORB_MAX_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_PYRAMID_LEVELS
    )
    keypoints, descriptors = orb.detectAndCompute(grayscale, None)
    
    if descriptors is None or len(keypoints) < 2:
        return 0.0
    
    # Match features to themselves with cross-checking
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors, descriptors)
    
    # Compute displacement vectors, filtering self-matches and near-neighbors
    displacement_vectors = []
    for match in matches:
        # Skip self-matches
        if match.queryIdx == match.trainIdx:
            continue
        
        point_query = np.array(keypoints[match.queryIdx].pt)
        point_train = np.array(keypoints[match.trainIdx].pt)
        displacement = point_train - point_query
        
        # Ignore tiny displacements (local noise)
        if np.linalg.norm(displacement) < epsilon_pixels:
            continue
        
        displacement_vectors.append(displacement)
    
    if len(displacement_vectors) < min_matches:
        return 0.0
    
    # Analyze displacement vector distribution
    displacements = np.array(displacement_vectors, dtype=np.float32)
    
    angles = (np.degrees(np.arctan2(displacements[:, 1], displacements[:, 0])) + 360.0) % 360.0
    magnitudes = np.linalg.norm(displacements, axis=1)
    
    # Create 2D histogram of angle vs magnitude
    magnitude_max = max(MAGNITUDE_MAX_DEFAULT, float(magnitudes.max()))
    histogram_2d, _, _ = np.histogram2d(
        angles,
        magnitudes,
        bins=(ANGLE_HISTOGRAM_BINS, MAGNITUDE_HISTOGRAM_BINS),
        range=[[0, 360], [0, magnitude_max]]
    )
    
    # Dominant cluster ratio indicates copy-move likelihood
    peak_count = histogram_2d.max()
    total_count = histogram_2d.sum() + EPSILON_SMALL
    cluster_ratio = float(peak_count / total_count)
    
    return cluster_ratio


# ============================================================================
# 4. CFA/PRNU Inconsistency Detection
# ============================================================================

def compute_cfa_inconsistency_scores(
    card_bgr: np.ndarray,
    photo_bbox_xyxy: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Detect Color Filter Array (CFA) inconsistencies by comparing demosaicing
    residuals inside vs outside the portrait region.
    
    Uses simple bilinear interpolation to predict green channel values,
    then computes z-scores of residual statistics.
    
    Returns:
        Dictionary with 'cfa_mean_z' and 'cfa_std_z' scores
        Higher values indicate more suspicious differences
    """
    if card_bgr is None or photo_bbox_xyxy is None:
        return {"cfa_mean_z": 0.0, "cfa_std_z": 0.0}
    
    grayscale = convert_to_grayscale(card_bgr).astype(np.float32)
    
    # Predict green channel using cross-shaped bilinear kernel
    demosaic_kernel = np.array([
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0]
    ], dtype=np.float32)
    
    predicted_green = cv2.filter2D(grayscale, -1, demosaic_kernel)
    residual_map = np.abs(grayscale - predicted_green)
    
    # Create mask separating inside/outside portrait
    mask = np.zeros_like(residual_map, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, photo_bbox_xyxy)
    mask[y1:y2, x1:x2] = 1
    
    # Split residuals by region
    residuals_inside = residual_map[mask == 1]
    residuals_outside = residual_map[mask == 0]
    
    # Compute statistics with epsilon for numerical stability
    mean_inside = float(residuals_inside.mean() + EPSILON_SMALL)
    std_inside = float(residuals_inside.std() + EPSILON_SMALL)
    mean_outside = float(residuals_outside.mean() + EPSILON_SMALL)
    std_outside = float(residuals_outside.std() + EPSILON_SMALL)
    
    # Z-scores comparing inside vs outside
    mean_zscore = (mean_inside - mean_outside) / (std_outside + EPSILON_SMALL)
    std_zscore = (std_inside - std_outside) / (std_outside + EPSILON_SMALL)
    
    return {
        "cfa_mean_z": float(mean_zscore),
        "cfa_std_z": float(std_zscore)
    }


# ============================================================================
# 5. Font Forensics (Glyph Metrics Analysis)
# ============================================================================

def prepare_binary_text_image(bgr_image: np.ndarray) -> np.ndarray:
    """
    Prepare text region for glyph analysis.
    Upscales, binarizes with Otsu, inverts if needed, and applies morphology.
    """
    grayscale = convert_to_grayscale(bgr_image)
    
    # Upscale for better feature extraction
    upscaled = cv2.resize(
        grayscale,
        None,
        fx=BINARIZATION_UPSCALE_FACTOR,
        fy=BINARIZATION_UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )
    
    # Otsu binarization
    _, binary = cv2.threshold(
        upscaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Ensure white text on black background
    if binary.mean() < 127:
        binary = 255 - binary
    
    # Clean up noise
    morphology_kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, dtype=np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        morphology_kernel,
        iterations=1
    )
    
    return binary


def compute_stroke_density(binary_image: np.ndarray) -> float:
    """
    Compute stroke density using morphological skeleton.
    Returns ratio of skeleton pixels to total pixels.
    """
    skeleton = binary_image.copy()
    total_pixels = float(np.size(skeleton))
    skeleton_pixel_count = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, SKELETON_KERNEL_SIZE)
    
    while True:
        opened = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(skeleton, opened)
        eroded = cv2.erode(skeleton, kernel)
        skeleton = eroded.copy()
        skeleton_pixel_count += int(np.count_nonzero(temp))
        
        if cv2.countNonZero(skeleton) == 0:
            break
    
    return skeleton_pixel_count / (total_pixels + EPSILON_SMALL)


def compute_interglyph_gaps(binary_image: np.ndarray) -> np.ndarray:
    """
    Compute inter-glyph spacing by projecting to X-axis and finding gaps.
    Returns array of gap widths in pixels.
    """
    # Project foreground pixels to X-axis
    projection = (255 - binary_image).sum(axis=0)
    has_content = (projection > 0).astype(np.uint8)
    
    gaps = []
    in_content_run = False
    last_content_end = 0
    
    for i, has_pixel in enumerate(has_content):
        if has_pixel and not in_content_run:
            # Start of content run
            in_content_run = True
            gap_width = i - last_content_end
            if gap_width > 0:
                gaps.append(gap_width)
        
        if not has_pixel and in_content_run:
            # End of content run
            in_content_run = False
            last_content_end = i
    
    return np.array(gaps, dtype=np.float32) if gaps else np.array([0.0], dtype=np.float32)


def extract_glyph_feature_vector(binary_image: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive glyph metrics from binarized text region.
    
    Features (8 dimensions):
    - Mean and std of glyph widths
    - Mean and std of glyph heights
    - Mean and std of glyph areas (log-scaled)
    - Stroke density (skeleton-based)
    - Mean inter-glyph gap (log-scaled)
    
    Returns:
        Feature vector of shape (1, 8)
    """
    # Analyze connected components (glyphs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8
    )
    
    # Extract component statistics (excluding background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    
    if areas.size == 0:
        return np.zeros((1, 8), dtype=np.float32)
    
    # Compute auxiliary features
    stroke_density = compute_stroke_density(binary_image)
    interglyph_gaps = compute_interglyph_gaps(binary_image)
    
    # Assemble feature vector
    features = np.array([
        float(np.mean(widths)),
        float(np.std(widths) + EPSILON_SMALL),
        float(np.mean(heights)),
        float(np.std(heights) + EPSILON_SMALL),
        float(np.mean(areas)),
        float(np.std(areas) + EPSILON_SMALL),
        stroke_density,
        float(np.mean(interglyph_gaps)),
    ], dtype=np.float32).reshape(1, -1)
    
    # Log-transform heavy-tailed distributions
    features[:, 4:6] = np.log1p(features[:, 4:6])  # Area statistics
    features[:, 7:8] = np.log1p(features[:, 7:8])  # Gap statistics
    
    return features


def compute_font_anomaly_scores(
    card_bgr: np.ndarray,
    nid_bbox_xyxy: Optional[Tuple[int, int, int, int]],
    serial_bbox_xyxy: Optional[Tuple[int, int, int, int]],
    ocsvm_model_path: Optional[str]
) -> Dict[str, float]:
    """
    Detect font anomalies using One-Class SVM trained on genuine glyph metrics.
    
    Returns decision function values where negative scores indicate
    out-of-distribution (suspicious) fonts.
    
    Args:
        card_bgr: Full card image in BGR
        nid_bbox_xyxy: Bounding box for National ID number field
        serial_bbox_xyxy: Bounding box for serial number field
        ocsvm_model_path: Path to trained One-Class SVM model
    
    Returns:
        Dictionary with 'font_nid_df', 'font_serial_df' scores and 'ok' flag
    """
    scores = {
        "font_nid_df": 0.0,
        "font_serial_df": 0.0,
        "ok": False
    }
    
    # Check prerequisites
    if not SKLEARN_AVAILABLE:
        return scores
    
    if not ocsvm_model_path or not os.path.exists(ocsvm_model_path):
        return scores
    
    # Load trained model
    try:
        svm_model: OneClassSVM = joblib.load(ocsvm_model_path)  # type: ignore
    except Exception:
        return scores
    
    scores["ok"] = True
    
    # Analyze NID field
    if nid_bbox_xyxy is not None:
        nid_crop = safe_crop_region(card_bgr, nid_bbox_xyxy, padding=2)
        if nid_crop is not None and nid_crop.size > 0:
            binary = prepare_binary_text_image(nid_crop)
            features = extract_glyph_feature_vector(binary)
            scores["font_nid_df"] = float(svm_model.decision_function(features)[0])
    
    # Analyze serial field
    if serial_bbox_xyxy is not None:
        serial_crop = safe_crop_region(card_bgr, serial_bbox_xyxy, padding=2)
        if serial_crop is not None and serial_crop.size > 0:
            binary = prepare_binary_text_image(serial_crop)
            features = extract_glyph_feature_vector(binary)
            scores["font_serial_df"] = float(svm_model.decision_function(features)[0])
    
    return scores


# ============================================================================
# Public API (maintaining original function names)
# ============================================================================

# Alias original function names for backward compatibility
safe_crop = safe_crop_region
to_gray = convert_to_grayscale
seam_margin_mask = create_seam_margin_mask
noiseprint_or_mantra_score = compute_splice_detection_scores
lbp_texture_zscores = compute_lbp_texture_zscores
copy_move_score = detect_copy_move_manipulation
cfa_inconsistency = compute_cfa_inconsistency_scores
font_od_score = compute_font_anomaly_scores