from __future__ import annotations
import json, os
from typing import Dict, Any, Optional

DEFAULTS = {
    "weights": {"photo": 0.55, "ela_global": 0.15, "jpeg_global": 0.15, "illum": 0.05, "adv_forensics": 0.10},
    "score_threshold": 0.60,
    "hard_fail": {
        "ela_mean": 15.0, "ela_max": 120.0, "jpeg_std": 30.0,
        "photo_ela_z": 0.50, "photo_jpeg_z": 0.70,
        "residual": 18.0, "cfa_std_z": 1.25, "copy_move": 0.15
    },
    "soft_starts": {},  # optional p95 per metric
    "geometry": { "ar_lo": 0.68, "ar_hi": 0.78, "area_lo": 0.045, "area_hi": 0.10, "pad_ar": 0.02, "pad_area": 0.01 }
}

def load_thresholds(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # shallow-merge with defaults to keep missing keys sane
        out = DEFAULTS.copy()
        for k, v in cfg.items():
            if isinstance(v, dict):
                out[k] = {**DEFAULTS.get(k, {}), **v}
            else:
                out[k] = v
        return out
    except Exception:
        return DEFAULTS
