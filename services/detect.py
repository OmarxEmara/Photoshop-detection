from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------------
# Labels we keep from YOLO
# IMPORTANT: include all portrait-related labels
# -------------------------------------------------------------------
KEEP_LABELS = None

def _norm_label_map(model: YOLO) -> Dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}

def load_yolo_model(weights_path: str, imgsz: int = 640, conf: float = 0.25) -> YOLO:
    model = YOLO(weights_path)
    model.overrides["imgsz"] = imgsz
    model.overrides["conf"] = conf
    model.overrides["verbose"] = False
    return model

def _yolo_predict(model: YOLO, img) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = model.predict(img, verbose=False)[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy.numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy().astype(float)
    return boxes_xyxy, cls_ids, confs

def detect_card_bbox(model_card: YOLO, img) -> List[float] | None:
    res = model_card.predict(img, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    i = int(np.argmax(res.boxes.conf.cpu().numpy()))
    return res.boxes.xyxy.cpu().numpy()[i].tolist()

def detect_objects_raw(model_obj: YOLO, img) -> List[Dict]:
    boxes, cls_ids, confs = _yolo_predict(model_obj, img)
    id2name = _norm_label_map(model_obj)
    out: List[Dict] = []
    for (x1, y1, x2, y2), cid, cf in zip(boxes, cls_ids, confs):
        label = id2name.get(int(cid), str(cid))
        if (KEEP_LABELS is None) or (label in KEEP_LABELS):
            out.append({
                "label": label,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(cf),
            })
    return out

def map_raw_boxes_to_warped(raw_det, crop_xyxy, H_crop2canon, dst_size):
    import cv2
    x1c, y1c, x2c, y2c = map(float, crop_xyxy)
    W, H = dst_size
    mapped: List[Dict] = []
    for d in raw_det:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        xx1, yy1 = max(x1, x1c), max(y1, y1c)
        xx2, yy2 = min(x2, x2c), min(y2, y2c)
        if xx2 <= xx1 or yy2 <= yy1:
            continue
        src = np.array(
            [[xx1 - x1c, yy1 - y1c],
             [xx2 - x1c, yy1 - y1c],
             [xx2 - x1c, yy2 - y1c],
             [xx1 - x1c, yy2 - y1c]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, H_crop2canon).reshape(-1, 2)
        xw1, yw1 = float(dst[:, 0].min()), float(dst[:, 1].min())
        xw2, yw2 = float(dst[:, 0].max()), float(dst[:, 1].max())
        xw1 = float(np.clip(xw1, 0, W - 1)); xw2 = float(np.clip(xw2, 0, W - 1))
        yw1 = float(np.clip(yw1, 0, H - 1)); yw2 = float(np.clip(yw2, 0, H - 1))
        mapped.append({**d, "bbox_xyxy": [xw1, yw1, xw2, yw2]})
    return mapped

def detect_objects_direct_on_warped(model_obj: YOLO, warped_bgr) -> List[Dict]:
    boxes, cls_ids, confs = _yolo_predict(model_obj, warped_bgr)
    id2name = _norm_label_map(model_obj)
    out: List[Dict] = []
    for (x1, y1, x2, y2), cid, cf in zip(boxes, cls_ids, confs):
        label = id2name.get(int(cid), str(cid))
        if (KEEP_LABELS is None) or (label in KEEP_LABELS):
            out.append({
                "label": label,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(cf),
            })
    return out
