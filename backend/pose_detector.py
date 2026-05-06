import os

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


class PoseDetector:
    def __init__(self, model_path, device="cpu", conf=0.25, iou=0.45, imgsz=640):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = None
        self.available = False

        if YOLO is None:
            return

        try:
            self.model = YOLO(model_path)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def detect(self, frame, conf=None, iou=None, imgsz=None):
        if not self.available:
            return []

        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            return []

        try:
            kpts_xy = keypoints.xy.cpu().numpy()
            kpts_conf = (
                keypoints.conf.cpu().numpy()
                if getattr(keypoints, "conf", None) is not None
                else None
            )
        except Exception:
            return []

        detections = []
        total = min(len(boxes), len(kpts_xy))
        for idx in range(total):
            xyxy = boxes[idx].xyxy[0].tolist()
            score = float(boxes[idx].conf[0].item())
            clamped = _clamp_box(*xyxy, width, height)
            if clamped is None:
                continue
            detections.append(
                {
                    "box": clamped,
                    "score": score,
                    "keypoints": kpts_xy[idx],
                    "keypoint_scores": (
                        kpts_conf[idx] if kpts_conf is not None else None
                    ),
                }
            )

        return detections
