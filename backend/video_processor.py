import os

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_YOLO_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "results", "runs", "spixgro_face_detector", "weights", "best.pt"
)


class VIDEO:
    def __init__(self, gpu=False, use_yolo=True):
        print(
            f"[DEBUG-OCR] Initializing OCR Processor with gpu={gpu}, use_yolo={use_yolo}"
        )
        self.reader = None
        self.gpu_enabled = False
        self.smart_model = None
        self.use_yolo = use_yolo

        if easyocr is not None:
            try:
                print(f"[DEBUG-OCR] Attempting to load EasyOCR with GPU={gpu}")
                self.reader = easyocr.Reader(["en"], gpu=gpu)
                self.gpu_enabled = gpu
                print(f"[DEBUG-OCR] Successfully loaded EasyOCR with GPU={gpu}")
            except Exception as e:
                print(
                    f"[DEBUG-OCR] Failed to load EasyOCR with GPU, falling back to CPU: {e}"
                )
                try:
                    self.reader = easyocr.Reader(["en"], gpu=False)
                    self.gpu_enabled = False
                    print("[DEBUG-OCR] Successfully loaded EasyOCR with CPU")
                except Exception as e2:
                    print(f"[DEBUG-OCR] EasyOCR CPU fallback also failed: {e2}")
                    self.reader = None
        else:
            print("[DEBUG-OCR] easyocr not installed, OCR features disabled")

        if use_yolo and YOLO is not None:
            print("[DEBUG-OCR] Attempting to load YOLO models")
            if os.path.exists(_YOLO_MODEL_PATH):
                try:
                    self.smart_model = YOLO(_YOLO_MODEL_PATH)
                    if self.gpu_enabled:
                        self.smart_model.to("cuda")
                    print("[DEBUG-OCR] Successfully loaded primary YOLO model")
                except Exception as e:
                    print(f"[DEBUG-OCR] Primary YOLO model failed: {e}")
                    self.smart_model = None
            else:
                print(
                    f"[DEBUG-OCR] YOLO model not found at {_YOLO_MODEL_PATH}, skipping"
                )
        elif use_yolo and YOLO is None:
            print("[DEBUG-OCR] ultralytics not installed, YOLO features disabled")
        else:
            print("[DEBUG-OCR] YOLO disabled, using EasyOCR only")

    def frame_extract(self, video_path):
        """Extract frames from video at the given path."""
        print(f"Extracting frames from video at {video_path}")

    def yolo_detect(self, frame):
        """Perform YOLO detection on a frame."""
        print(f"Performing YOLO detection on frame {frame}")

    def blur_faces(self, frame):
        """Blur faces detected in the frame."""
        print(f"Blurring faces in frame {frame}")

    def blur_text(self, frame):
        """Blur text detected in the frame."""
        print(f"Blurring text in frame {frame}")

    def save_frame(self, frame, output_path):
        """Save a processed frame to disk."""
        print(f"Saving frame {frame} to {output_path}")

    def compile_video(self, frame, output_video_path):
        """Compile processed frames into an output video."""
        print(f"Compiling frames into video at {output_video_path}")