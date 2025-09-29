#!/usr/bin/env python3
"""
Egyptian ID Card Processing Pipeline
Detects, warps, and validates ID cards with optional OCR extraction.
"""
import os
import json
import argparse
import logging
from typing import Optional, Dict, List, Tuple, Any

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
from services.forgery import compute_validation

# Constants
CANONICAL_WIDTH = 864
CANONICAL_HEIGHT = 544
CANONICAL_SIZE = (CANONICAL_WIDTH, CANONICAL_HEIGHT)

# Thresholds
MIN_CARD_AREA_RATIO = 0.75
SKEW_TOLERANCE_DEGREES = 3.0
PHOTO_CONFIDENCE_THRESHOLD = 0.50
FALLBACK_PHOTO_CONFIDENCE = 0.25

# Scoring weights for photo fallback heuristic
FALLBACK_SCORE_CENTER_X_THRESHOLD = 0.45
FALLBACK_SCORE_CENTER_WEIGHT = 0.5
FALLBACK_SCORE_AREA_TARGET = 0.12
FALLBACK_SCORE_AREA_WEIGHT = 2.0
FALLBACK_SCORE_ASPECT_RATIO_TARGET = 0.8
FALLBACK_SCORE_ASPECT_RATIO_TOLERANCE = 0.8

# Expected field lengths
EXPECTED_NID_LENGTH = 14


def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_detection_overlay(image: np.ndarray, detections: List[Dict], output_path: str) -> None:
    """Draw detection boxes on image and save to file."""
    overlay = image.copy()
    formatted_detections = [
        {**detection, "bbox_xyxy": [int(coord) for coord in detection["bbox_xyxy"]]} 
        for detection in detections
    ]
    draw_boxes(overlay, formatted_detections)
    cv2.imwrite(output_path, overlay)


def load_image(image_path: str) -> np.ndarray:
    """Load and validate image from path."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image


def detect_and_crop_card(
    model, 
    image: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect ID card in image and return crop with bbox coordinates."""
    card_bbox = detect_card_bbox(model, image)
    if card_bbox is None:
        raise RuntimeError("No card detected.")
    
    x1, y1, x2, y2 = map(int, card_bbox)
    card_crop = crop_with_bbox(image, card_bbox)
    
    return card_crop, (x1, y1, x2, y2)


def create_card_quadrilateral(card_crop: np.ndarray) -> np.ndarray:
    """
    Detect card corners or create default rectangle.
    Falls back to full crop rectangle if detected area is too small.
    """
    quad = detect_card_quad_from_crop(card_crop)
    crop_height, crop_width = card_crop.shape[:2]
    
    quad_area = cv2.contourArea(quad.astype(np.float32))
    min_area = MIN_CARD_AREA_RATIO * (crop_width * crop_height)
    
    if quad_area < min_area:
        quad = np.array([
            [0, 0], 
            [crop_width - 1, 0], 
            [crop_width - 1, crop_height - 1], 
            [0, crop_height - 1]
        ], dtype=np.float32)
    
    return quad


def warp_card_to_canonical(
    card_crop: np.ndarray, 
    quad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp card crop to canonical rectangular view."""
    warped_image, homography_matrix = warp_to_canonical(
        card_crop, 
        quad, 
        dst_size=CANONICAL_SIZE
    )
    return warped_image, homography_matrix


def analyze_card_geometry(warped_image: np.ndarray) -> Dict[str, Any]:
    """Compute aspect ratio and skew angle of warped card."""
    aspect_ratio, is_aspect_ratio_valid = aspect_ratio_ok(warped_image)
    
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    skew_degrees = estimate_skew_deg(gray_warped)
    is_skew_valid = None if skew_degrees is None else abs(skew_degrees) < SKEW_TOLERANCE_DEGREES
    
    return {
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_ok": is_aspect_ratio_valid,
        "skew_deg": skew_degrees,
        "skew_ok": is_skew_valid
    }


def detect_fields_on_card(
    model,
    original_image: np.ndarray,
    warped_image: np.ndarray,
    card_bbox: Tuple[int, int, int, int],
    homography_matrix: np.ndarray
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Detect text fields and photo using two approaches:
    1. Detect on original image and map to warped
    2. Detect directly on warped image
    """
    x1, y1, x2, y2 = card_bbox
    
    raw_detections = detect_objects_raw(model, original_image)
    
    mapped_detections = map_raw_boxes_to_warped(
        raw_detections, 
        (x1, y1, x2, y2), 
        homography_matrix, 
        dst_size=CANONICAL_SIZE
    )
    
    direct_detections = detect_objects_direct_on_warped(model, warped_image)
    
    return raw_detections, mapped_detections, direct_detections


def merge_detections_by_label(
    mapped_detections: List[Dict],
    direct_detections: List[Dict]
) -> Dict[str, Dict]:
    """
    Merge detections from both sources, keeping highest confidence per label.
    Mapped detections are considered first, then direct detections.
    """
    best_detections = {}
    
    def consider_detections(detection_list: List[Dict], source_name: str) -> None:
        for detection in detection_list:
            label = detection["label"]
            if label not in best_detections or detection["conf"] > best_detections[label]["conf"]:
                best_detections[label] = {**detection, "source": source_name}
    
    consider_detections(mapped_detections, "mapped_from_raw")
    consider_detections(direct_detections, "direct_on_warp")
    
    return best_detections


def compute_fallback_photo_score(
    detection: Dict,
    canonical_width: int,
    canonical_height: int
) -> float:
    """
    Compute heuristic score for potential photo region.
    Favors left-side regions with appropriate size and aspect ratio.
    """
    x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
    
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / float(height)
    
    area_normalized = (width * height) / float(canonical_width * canonical_height)
    center_x_normalized = 0.5 * (x1 + x2) / canonical_width
    
    # Score components
    score_center = FALLBACK_SCORE_CENTER_WEIGHT if center_x_normalized < FALLBACK_SCORE_CENTER_X_THRESHOLD else 0.0
    score_area = min(area_normalized, FALLBACK_SCORE_AREA_TARGET) * FALLBACK_SCORE_AREA_WEIGHT
    score_aspect_ratio = 1.0 - min(abs(aspect_ratio - FALLBACK_SCORE_ASPECT_RATIO_TARGET), FALLBACK_SCORE_ASPECT_RATIO_TOLERANCE)
    
    return score_center + score_area + score_aspect_ratio


def apply_photo_fallback_heuristic(
    best_detections: Dict[str, Dict],
    mapped_detections: List[Dict],
    direct_detections: List[Dict]
) -> Tuple[Dict[str, Dict], bool]:
    """
    Apply fallback heuristic to detect photo if neither photo nor invalid_photo detected.
    Returns updated detections dict and whether fallback was used.
    """
    if "photo" in best_detections or "invalid_photo" in best_detections:
        return best_detections, False
    
    all_detections = direct_detections + mapped_detections
    
    best_candidate_bbox = None
    best_candidate_score = -1.0
    
    for detection in all_detections:
        score = compute_fallback_photo_score(detection, CANONICAL_WIDTH, CANONICAL_HEIGHT)
        if score > best_candidate_score:
            best_candidate_bbox = [int(coord) for coord in detection["bbox_xyxy"]]
            best_candidate_score = score
    
    if best_candidate_bbox is not None:
        best_detections["photo"] = {
            "label": "photo",
            "bbox_xyxy": [float(coord) for coord in best_candidate_bbox],
            "conf": FALLBACK_PHOTO_CONFIDENCE,
            "source": "fallback"
        }
        return best_detections, True
    
    return best_detections, False


def draw_fallback_photo_overlay(warped_image: np.ndarray, photo_bbox: List[float], output_path: str) -> None:
    """Draw fallback photo detection on warped image."""
    overlay = warped_image.copy()
    x1, y1, x2, y2 = [int(coord) for coord in photo_bbox]
    
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        overlay, 
        "fallback_face", 
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 0, 255), 
        1, 
        cv2.LINE_AA
    )
    
    cv2.imwrite(output_path, overlay)


def extract_ocr_fields(
    warped_image: np.ndarray,
    best_detections: Dict[str, Dict],
    no_ocr: bool
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Extract text from detected fields using OCR.
    Returns OCR results and decoded NID information.
    """
    ocr_results = {
        "first_name": "",
        "last_name": "",
        "address": "",
        "nid": "",
        "dob": "",
        "serial": ""
    }
    nid_decoded = {}
    
    if no_ocr:
        return ocr_results, nid_decoded
    
    # Import OCR utilities only when needed
    from services.ocr_utils import (
        set_paddle_paths,
        get_paddle,
        extract_arabic_field,
        extract_address_field,
        extract_digits_strict,
        extract_date_digits,
        decode_egyptian_id,
        tesseract_text_simple,
    )
    
    def get_bbox_for_field(field_name: str) -> Optional[List[float]]:
        return best_detections[field_name]["bbox_xyxy"] if field_name in best_detections else None
    
    if get_bbox_for_field("firstName") is not None:
        ocr_results["first_name"] = extract_arabic_field(warped_image, get_bbox_for_field("firstName"))
    
    if get_bbox_for_field("lastName") is not None:
        ocr_results["last_name"] = extract_arabic_field(warped_image, get_bbox_for_field("lastName"))
    
    if get_bbox_for_field("address") is not None:
        ocr_results["address"] = extract_address_field(warped_image, get_bbox_for_field("address"))
    
    if get_bbox_for_field("nid") is not None:
        ocr_results["nid"] = extract_digits_strict(
            warped_image, 
            get_bbox_for_field("nid"), 
            expected=EXPECTED_NID_LENGTH
        )
    
    if get_bbox_for_field("dob") is not None:
        ocr_results["dob"] = extract_date_digits(warped_image, get_bbox_for_field("dob"))
    
    if get_bbox_for_field("serial") is not None:
        x1, y1, x2, y2 = map(int, get_bbox_for_field("serial"))
        serial_crop = warped_image[y1:y2, x1:x2]
        ocr_results["serial"] = tesseract_text_simple(serial_crop)
    
    if len(ocr_results["nid"]) == EXPECTED_NID_LENGTH and ocr_results["nid"].isdigit():
        nid_decoded = decode_egyptian_id(ocr_results["nid"])
    
    return ocr_results, nid_decoded


def normalize_point(point: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize point coordinates to [0, 1] range."""
    return (point[0] / CANONICAL_WIDTH, point[1] / CANONICAL_HEIGHT)


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Compute center point of bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_geometry_pairs(best_detections: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized geometric relationships between key field pairs.
    Used for validating card layout consistency.
    """
    def get_bbox_for_field(field_name: str) -> Optional[List[float]]:
        return best_detections[field_name]["bbox_xyxy"] if field_name in best_detections else None
    
    pairs = {}
    
    # Logo to first name
    if get_bbox_for_field("front_logo") is not None and get_bbox_for_field("firstName") is not None:
        point_a = normalize_point(compute_bbox_center(get_bbox_for_field("front_logo")))
        point_b = normalize_point(compute_bbox_center(get_bbox_for_field("firstName")))
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        pairs["logo_to_name"] = {
            "dx": dx,
            "dy": dy,
            "dist": float(np.hypot(dx, dy))
        }
    
    # Date of birth to NID
    if get_bbox_for_field("dob") is not None and get_bbox_for_field("nid") is not None:
        point_a = normalize_point(compute_bbox_center(get_bbox_for_field("dob")))
        point_b = normalize_point(compute_bbox_center(get_bbox_for_field("nid")))
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        pairs["dob_to_nid"] = {
            "dx": dx,
            "dy": dy,
            "dist": float(np.hypot(dx, dy))
        }
    
    # Date of birth to serial
    if get_bbox_for_field("dob") is not None and get_bbox_for_field("serial") is not None:
        point_a = normalize_point(compute_bbox_center(get_bbox_for_field("dob")))
        point_b = normalize_point(compute_bbox_center(get_bbox_for_field("serial")))
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        pairs["dob_to_serial"] = {
            "dx": dx,
            "dy": dy,
            "dist": float(np.hypot(dx, dy))
        }
    
    # Last name to address
    if get_bbox_for_field("lastName") is not None and get_bbox_for_field("address") is not None:
        point_a = normalize_point(compute_bbox_center(get_bbox_for_field("lastName")))
        point_b = normalize_point(compute_bbox_center(get_bbox_for_field("address")))
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        pairs["name_to_address"] = {
            "dx": dx,
            "dy": dy,
            "dist": float(np.hypot(dx, dy))
        }
    
    return pairs


def determine_photo_label_and_confidence(
    best_detections: Dict[str, Dict]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Determine which photo detection to use (photo vs invalid_photo).
    Returns label and confidence of chosen detection.
    """
    has_photo = "photo" in best_detections
    has_invalid_photo = "invalid_photo" in best_detections
    
    if has_photo and has_invalid_photo:
        if best_detections["invalid_photo"]["conf"] >= best_detections["photo"]["conf"]:
            return "invalid_photo", float(best_detections["invalid_photo"]["conf"])
        else:
            return "photo", float(best_detections["photo"]["conf"])
    elif has_photo:
        return "photo", float(best_detections["photo"]["conf"])
    elif has_invalid_photo:
        return "invalid_photo", float(best_detections["invalid_photo"]["conf"])
    else:
        return None, None


def save_card_detection_overlay(
    original_image: np.ndarray,
    card_bbox: Tuple[int, int, int, int],
    output_path: str
) -> None:
    """Draw card bounding box on original image."""
    overlay = original_image.copy()
    x1, y1, x2, y2 = card_bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_path, overlay)


def format_boxes_for_validation(best_detections: Dict[str, Dict]) -> Dict[str, List[int]]:
    """Convert detection bboxes to integer format for validation."""
    return {
        label: [int(coord) for coord in detection["bbox_xyxy"]]
        for label, detection in best_detections.items()
    }


def run_validation(
    warped_image: np.ndarray,
    output_directory: str,
    boxes_canonical: Dict[str, List[int]],
    geometry_info: Dict[str, Any],
    geometry_pairs: Dict[str, Dict[str, float]],
    photo_label: Optional[str],
    photo_conf: Optional[float],
    calibration_path: Optional[str],
    thresholds_path: Optional[str],
    invalid_photo_fail_conf: float,
    no_review: bool
) -> Dict[str, Any]:
    """Run forgery detection and validation checks."""
    photo_conf_ok = photo_conf is None or photo_conf >= PHOTO_CONFIDENCE_THRESHOLD
    
    validation_results = compute_validation(
        warped_bgr=warped_image,
        outdir=output_directory,
        boxes=boxes_canonical,
        ar_ok=geometry_info["aspect_ratio_ok"],
        skew_deg=geometry_info["skew_deg"],
        pairs=geometry_pairs,
        photo_conf=photo_conf,
        photo_label=photo_label,
        photo_conf_ok=photo_conf_ok,
        calibration_path=calibration_path,
        thresholds_path=thresholds_path,
        require_photo=True,
        invalid_photo_fail_conf=float(invalid_photo_fail_conf),
        map_review_to_fail=bool(no_review),
        fail_if_geometry_bad=True,
    )
    
    return validation_results


def build_output_json(
    image_path: str,
    card_bbox: Tuple[int, int, int, int],
    homography_matrix: np.ndarray,
    geometry_info: Dict[str, Any],
    raw_detections: List[Dict],
    mapped_detections: List[Dict],
    direct_detections: List[Dict],
    boxes_canonical: Dict[str, List[int]],
    geometry_pairs: Dict[str, Dict[str, float]],
    ocr_results: Dict[str, str],
    nid_decoded: Dict[str, Any],
    validation_results: Dict[str, Any],
    output_directory: str,
    fallback_used: bool
) -> Dict[str, Any]:
    """Construct final output JSON structure."""
    x1, y1, x2, y2 = card_bbox
    
    output = {
        "image": image_path,
        "card_bbox_original": [int(v) for v in (x1, y1, x2, y2)],
        "homography_card_to_canonical": np.array(homography_matrix).tolist(),
        "geometry": geometry_info,
        "detections_raw": raw_detections,
        "detections_canonical_mapped": mapped_detections,
        "detections_canonical_direct": direct_detections,
        "boxes_canonical_first": boxes_canonical,
        "geometry_pairs": geometry_pairs,
        "ocr": ocr_results,
        "nid_decoded": nid_decoded,
        "validation": validation_results,
        "artifacts": {
            "cli_equiv_overlay": os.path.join(output_directory, "00_cli_equiv_full.png"),
            "warped_path": os.path.join(output_directory, "01_warped.png"),
            "raw_overlay": os.path.join(output_directory, "02a_raw_overlay.png"),
            "direct_on_warp_overlay": os.path.join(output_directory, "02b_direct_on_warp_overlay.png"),
            "overlay_path": os.path.join(output_directory, "02_overlay.png"),
        }
    }
    
    if fallback_used:
        output["artifacts"]["fallback_face_overlay"] = os.path.join(
            output_directory, 
            "02c_fallback_face_overlay.png"
        )
    
    return output


def run(
    image_path: str,
    card_weights: str,
    obj_weights: str,
    outdir: str,
    card_conf: float = 0.10,
    card_imgsz: int = 640,
    obj_conf: float = 0.12,
    obj_imgsz: int = 1280,
    paddle_det: Optional[str] = None,
    paddle_rec: Optional[str] = None,
    calibration: Optional[str] = None,
    thresholds_path: Optional[str] = None,
    invalid_photo_fail_conf: float = 0.80,
    no_review: bool = False,
    no_ocr: bool = False
) -> None:
    """
    Main pipeline for Egyptian ID card processing.
    
    Args:
        image_path: Path to input image
        card_weights: Path to card detection YOLO weights
        obj_weights: Path to field detection YOLO weights
        outdir: Output directory for artifacts
        card_conf: Confidence threshold for card detection
        card_imgsz: Image size for card detection
        obj_conf: Confidence threshold for field detection
        obj_imgsz: Image size for field detection
        paddle_det: Path to PaddleOCR detection model
        paddle_rec: Path to PaddleOCR recognition model
        calibration: Path to calibration file for validation
        thresholds_path: Path to validation thresholds file
        invalid_photo_fail_conf: Confidence threshold for invalid photo
        no_review: Map review status to fail in validation
        no_ocr: Skip OCR processing
    """
    ensure_directory_exists(outdir)
    
    # Initialize OCR if needed
    if not no_ocr:
        from services.ocr_utils import set_paddle_paths, get_paddle
        set_paddle_paths(paddle_det, paddle_rec)
        _ = get_paddle()
    
    # Load YOLO models
    model_card = load_yolo_model(card_weights, imgsz=card_imgsz, conf=card_conf)
    model_obj = load_yolo_model(obj_weights, imgsz=obj_imgsz, conf=obj_conf)
    
    # Load and process image
    original_image = load_image(image_path)
    
    # Detect and crop card
    card_crop, card_bbox = detect_and_crop_card(model_card, original_image)
    
    # Warp to canonical view
    quad = create_card_quadrilateral(card_crop)
    warped_image, homography_matrix = warp_card_to_canonical(card_crop, quad)
    geometry_info = analyze_card_geometry(warped_image)
    
    # Save warped image
    cv2.imwrite(os.path.join(outdir, "01_warped.png"), warped_image)
    
    # Detect fields
    raw_detections, mapped_detections, direct_detections = detect_fields_on_card(
        model_obj,
        original_image,
        warped_image,
        card_bbox,
        homography_matrix
    )
    
    # Save detection overlays
    save_detection_overlay(original_image, raw_detections, os.path.join(outdir, "02a_raw_overlay.png"))
    save_detection_overlay(warped_image, direct_detections, os.path.join(outdir, "02b_direct_on_warp_overlay.png"))
    
    # Merge detections
    best_detections = merge_detections_by_label(mapped_detections, direct_detections)
    
    # Apply photo fallback if needed
    best_detections, fallback_used = apply_photo_fallback_heuristic(
        best_detections,
        mapped_detections,
        direct_detections
    )
    
    if fallback_used:
        draw_fallback_photo_overlay(
            warped_image,
            best_detections["photo"]["bbox_xyxy"],
            os.path.join(outdir, "02c_fallback_face_overlay.png")
        )
    
    # Save merged detection overlay
    save_detection_overlay(warped_image, list(best_detections.values()), os.path.join(outdir, "02_overlay.png"))
    
    # Extract OCR
    ocr_results, nid_decoded = extract_ocr_fields(warped_image, best_detections, no_ocr)
    
    # Compute geometry pairs
    geometry_pairs = compute_geometry_pairs(best_detections)
    
    # Save card detection overlay
    save_card_detection_overlay(original_image, card_bbox, os.path.join(outdir, "00_cli_equiv_full.png"))
    
    # Format boxes for validation
    boxes_canonical = format_boxes_for_validation(best_detections)
    
    # Determine photo detection
    photo_label, photo_conf = determine_photo_label_and_confidence(best_detections)
    
    # Run validation
    validation_results = run_validation(
        warped_image,
        outdir,
        boxes_canonical,
        geometry_info,
        geometry_pairs,
        photo_label,
        photo_conf,
        calibration,
        thresholds_path,
        invalid_photo_fail_conf,
        no_review
    )
    
    # Build and print output
    output = build_output_json(
        image_path,
        card_bbox,
        homography_matrix,
        geometry_info,
        raw_detections,
        mapped_detections,
        direct_detections,
        boxes_canonical,
        geometry_pairs,
        ocr_results,
        nid_decoded,
        validation_results,
        outdir,
        fallback_used
    )
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    """Parse command-line arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Process Egyptian ID card images with detection, warping, and validation"
    )
    
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--card-weights", required=True, help="Path to card detection YOLO weights")
    parser.add_argument("--obj-weights", required=True, help="Path to field detection YOLO weights")
    parser.add_argument("--card-conf", type=float, default=0.10, help="Card detection confidence threshold")
    parser.add_argument("--card-imgsz", type=int, default=640, help="Card detection image size")
    parser.add_argument("--obj-conf", type=float, default=0.12, help="Field detection confidence threshold")
    parser.add_argument("--obj-imgsz", type=int, default=1280, help="Field detection image size")
    parser.add_argument("--paddle-det", type=str, default=None, help="Path to PaddleOCR detection model")
    parser.add_argument("--paddle-rec", type=str, default=None, help="Path to PaddleOCR recognition model")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for artifacts")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration file")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to validation thresholds file")
    parser.add_argument("--invalid-photo-fail-conf", type=float, default=0.80, 
                        help="Confidence threshold for invalid photo to trigger failure")
    parser.add_argument("--no-review", action="store_true", help="Map review status to fail in validation")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR and PaddleOCR initialization")
    
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()