# api.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import io, json, os, tempfile
from contextlib import redirect_stdout
from typing import Optional
from pathlib import Path  # <-- needed!

import run_front  # uses your run_front.run(...)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

CARD_WEIGHTS = os.path.join(APP_ROOT, "models", "detect_id_card.pt")
OBJ_WEIGHTS  = os.path.join(APP_ROOT, "models", "detect_objects.pt")
CALIBRATION  = os.path.join(APP_ROOT, "calibration_stats.json")
THRESHOLDS_JSON = os.path.join(APP_ROOT, "thresholds.json")
OUTDIR = os.path.join(APP_ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)


def load_default_threshold() -> Optional[float]:
    try:
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "fraud_threshold" in data:
            return float(data["fraud_threshold"])
    except Exception:
        pass
    return None


DEFAULT_THRESHOLD = load_default_threshold() or 0.5912  # from your sweep


def run_front_capture(image_path: str,
                      threshold_path: Optional[str] = THRESHOLDS_JSON) -> dict:
    """
    Call run_front.run(...) and capture its printed JSON from stdout.
    We run with no OCR (photo-only pipeline).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_front.run(
            image_path=image_path,
            card_weights=CARD_WEIGHTS,
            obj_weights=OBJ_WEIGHTS,
            outdir=OUTDIR,
            card_conf=0.10,
            card_imgsz=640,
            obj_conf=0.12,
            obj_imgsz=1280,
            paddle_det=None,
            paddle_rec=None,
            calibration=CALIBRATION if os.path.exists(CALIBRATION) else None,
            thresholds_path=threshold_path if (threshold_path and os.path.exists(threshold_path)) else None,
            invalid_photo_fail_conf=0.80,
            no_review=True,   # map_review_to_fail=True inside run_front
            no_ocr=True,      # photo-only
        )
    text = buf.getvalue().strip()
    if not text:
        raise RuntimeError("run_front produced no output")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse run_front output: {e}\nRaw head:\n{text[:600]}")


app = FastAPI(title="eKYC ID Front – Photo Forgery API", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    ok = {
        "card_weights": os.path.exists(CARD_WEIGHTS),
        "obj_weights": os.path.exists(OBJ_WEIGHTS),
        "calibration": os.path.exists(CALIBRATION),
        "thresholds_json": os.path.exists(THRESHOLDS_JSON),
        "outdir": os.path.isdir(OUTDIR),
    }
    return {"status": ("ok" if all(ok.values()) else "degraded"), **ok}


@app.post("/verify")
async def verify_id(
    file: UploadFile = File(...),
    threshold: float | None = Query(
        None,
        description="Override fraud threshold (uses thresholds.json or 0.5912 if not provided)."
    ),
):
    img_path: str | None = None
    try:
        # 1) Save upload
        suffix = Path(file.filename or "id").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            img_path = tmp.name

        # 2) Run pipeline (photo-only)
        result = run_front_capture(img_path)

        v = (result or {}).get("validation") or {}
        # fraud_score is inside validation already (run_front.compute_validation)
        fraud_score = v.get("fraud_score", None)

        used_threshold = float(threshold) if threshold is not None else float(DEFAULT_THRESHOLD)

        decision = None
        reason = None
        if isinstance(fraud_score, (int, float)):
            if fraud_score >= used_threshold:
                decision = "FAIL"
                reason = f"fraud_score={fraud_score:.4f} >= threshold={used_threshold:.4f}"
            else:
                decision = "PASS"
                reason = f"fraud_score={fraud_score:.4f} < threshold={used_threshold:.4f}"

        # collect photo-only metrics if present
        photo_metrics = {
            "fraud_score": fraud_score,
            "ela_mean": v.get("ela_mean"),
            "ela_max": v.get("ela_max"),
            "jpeg_std": v.get("jpeg_std"),
            "illum_std": v.get("illum_std"),
            "photo_ar": v.get("photo_ar"),
            "photo_area": v.get("photo_area"),
            "copy_move": v.get("copy_move"),
        }

        return {
            "decision": decision,
            "reason": reason,
            "threshold_used": used_threshold,
            "photo_only_scores": {k: photo_metrics[k] for k in photo_metrics if photo_metrics[k] is not None},
            "artifacts": result.get("artifacts", {}),
        }

    finally:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
