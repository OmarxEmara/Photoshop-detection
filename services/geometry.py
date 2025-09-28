import cv2
import numpy as np

def crop_with_bbox(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    return img[y1:y2, x1:x2]

def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    # start near TL
    idx = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -idx, axis=0)

def detect_card_quad_from_crop(card_crop) -> np.ndarray:
    g = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    v = np.median(g); lo = int(max(0, 0.66 * v)); hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(g, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = card_crop.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 4:
        approx = cv2.boxPoints(cv2.minAreaRect(cnt))
    quad = approx.reshape(-1, 2).astype(np.float32)
    if quad.shape[0] > 4:
        hull = cv2.convexHull(quad)
        if hull.shape[0] >= 4:
            quad = hull.reshape(-1, 2)
    return order_corners_clockwise(quad).astype(np.float32)

def warp_to_canonical(card_crop, quad, dst_size=(864, 544)):
    W, H = dst_size
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card_crop, Hm, (W, H), flags=cv2.INTER_CUBIC)
    return warped, Hm

def aspect_ratio_ok(warped, target=85.6/53.98, tol=0.03):
    h, w = warped.shape[:2]
    ar = w / float(h)
    return ar, abs(ar - target) <= tol

def estimate_skew_deg(gray):
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None:
        return None
    angles = []
    for rho, theta in lines[:, 0, :]:
        angle = (theta * 180.0/np.pi) - 90.0
        angle = ((angle + 90) % 180) - 90
        if -60 < angle < 60:
            angles.append(angle)
    if not angles:
        return None
    return float(np.median(angles))

def draw_boxes(img, dets, color=(0, 255, 0)):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = d.get("label", ""); conf = d.get("conf", 0.0)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
