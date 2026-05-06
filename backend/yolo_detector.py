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
    return x1, y1, x2, y2


class YoloDetector:
    def __init__(self, model_path, device="cpu", conf=0.25, iou=0.45, imgsz=640):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = None
        self.available = False
        self.names = {}

        if YOLO is None:
            return
        if not model_path or not os.path.exists(model_path):
            return

        try:
            self.model = YOLO(model_path)
            self.names = getattr(self.model, "names", {}) or {}
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def detect(self, frame, conf=None, iou=None, imgsz=None, classes=None):
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
            classes=classes,
        )

        if not results:
            return []

        detections = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0].item())
            cls_idx = int(box.cls[0].item())
            label = self.names.get(cls_idx, str(cls_idx))
            x1, y1, x2, y2 = _clamp_box(*xyxy, width, height)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            detections.append(
                {
                    "box": (x1, y1, w, h),
                    "score": score,
                    "label": label,
                }
            )

        return detections
