# services/yolo_infer.py
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

def load_yolo_models(card_weights: str, obj_weights: str):
    model_card = YOLO(card_weights)
    model_obj  = YOLO(obj_weights)
    return model_card, model_obj

def detect_card_bbox(model_card, img_bgr, conf: float = 0.15, iou: float = 0.5, imgsz: int = 1280) -> Optional[list]:
    res = model_card.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    i = int(np.argmax(areas))
    return list(map(int, boxes[i]))

def detect_fields(model_obj, img_bgr, conf: float = 0.15, iou: float = 0.5, imgsz: int = 1280) -> List[Dict[str, Any]]:
    res = model_obj.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz)[0]
    names = res.names
    dets = []
    for b in res.boxes:
        cls = int(b.cls.item()); cf = float(b.conf.item())
        x1,y1,x2,y2 = map(float, b.xyxy.squeeze().tolist())
        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        dets.append({"label": label, "conf": cf, "bbox_xyxy": [x1,y1,x2,y2]})
    return dets
