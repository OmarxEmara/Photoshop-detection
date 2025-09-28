# services/ocr_utils.py
"""
Thin wrapper around PaddleOCR that lets the rest of the pipeline import OCR
utilities safely even when PaddleOCR or its models are missing.

- set_paddle_paths(det_dir, rec_dir): optionally point to local model dirs
- get_paddle(): returns a ready PaddleOCR instance or None if unavailable
"""

from __future__ import annotations
import os
from typing import Optional

# Globals to cache paths and the OCR instance
_DET_DIR: Optional[str] = None
_REC_DIR: Optional[str] = None
_OCR = None
_OCR_INIT_TRIED = False

# Default config — keep it conservative and English-only to avoid missing model errors
# (Change to 'ar' later only after installing Arabic PP-OCRv4 models.)
_OCR_KW = dict(
    use_angle_cls=True,
    lang="en",           # IMPORTANT: avoid ValueError for unavailable 'ar' models
    show_log=False,
    det_limit_side_len=1536,
)

def set_paddle_paths(paddle_det: Optional[str], paddle_rec: Optional[str]) -> None:
    """
    Set optional local directories for detector/recognizer models.
    If None, PaddleOCR will try to download as needed.
    """
    global _DET_DIR, _REC_DIR
    _DET_DIR = paddle_det if paddle_det else None
    _REC_DIR = paddle_rec if paddle_rec else None

def _init_paddle(det_dir: Optional[str], rec_dir: Optional[str]):
    """
    Try to build a PaddleOCR instance. Return the instance, or None on failure.
    We swallow import / model errors so the pipeline can continue without OCR.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None  # PaddleOCR not installed — run without OCR

    kwargs = dict(_OCR_KW)
    if det_dir and os.path.isdir(det_dir):
        kwargs["det_model_dir"] = det_dir
    if rec_dir and os.path.isdir(rec_dir):
        kwargs["rec_model_dir"] = rec_dir

    try:
        return PaddleOCR(**kwargs)
    except Exception:
        # Most common failure here is: models not available for selected lang
        # or bad local model paths. Run without OCR instead of crashing.
        return None

def get_paddle():
    """
    Public accessor expected by run_front.py.
    Returns:
      - PaddleOCR instance if available, else None.
    """
    global _OCR, _OCR_INIT_TRIED
    if _OCR is not None:
        return _OCR
    if not _OCR_INIT_TRIED:
        _OCR_INIT_TRIED = True
        _ocr = _init_paddle(_DET_DIR, _REC_DIR)
        # Cache even if None so we don't retry on every call
        _OCR = _ocr
    return _OCR

# (Optional) convenience for callers that want a boolean
def ocr_enabled() -> bool:
    return get_paddle() is not None
