from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
class VIDEO:
    def __init__(self, gpu=False, use_yolo=True):
        print(f"[DEBUG-OCR] Initializing OCR Processor with gpu={gpu}, use_yolo={use_yolo}")
        # Try GPU first, fallback to CPU if GPU is not available
        try:
            print(f"[DEBUG-OCR] Attempting to load EasyOCR with GPU={gpu}")
            self.reader = easyocr.Reader(
                ['en'],
                gpu=gpu
            )
            self.gpu_enabled = gpu
            print(f"[DEBUG-OCR] Successfully loaded EasyOCR with GPU={gpu}")
        except Exception as e:
            print(f"[DEBUG-OCR] Failed to load EasyOCR with GPU, falling back to CPU: {e}")
            self.reader = easyocr.Reader(
                ['en'],
                gpu=False
            )
            self.gpu_enabled = False
            print(f"[DEBUG-OCR] Successfully loaded EasyOCR with CPU")
        self.use_yolo = use_yolo
        if use_yolo:
            print(f"[DEBUG-OCR] Attempting to load YOLO models")
            try:
                self.smart_model = YOLO(r"yolov8n.pt")
                if self.gpu_enabled:
                    self.smart_model.to('cuda')
                print(f"[DEBUG-OCR] Successfully loaded primary YOLO model")
            except Exception as e:
                print(f"[DEBUG-OCR] Primary YOLO model failed: {e}")
        else:
            print(f"[DEBUG-OCR] YOLO disabled, using EasyOCR only")        
    
    def frame_extract(self, video_path):
        print(f"Extracting frames from video at {video_path}")
    
    def yolo_detect(self, frame):
        print(f"Performing YOLO detection on frame {frame}")
    
    def blur_faces(self, frame):
        print(f"Blurring faces in frame {frame}")
    
    def blur_text(self, frame):
        print(f"Blurring text in frame {frame}")
    
    def save_frame(self, frame, output_path):
        print(f"Saving frame {frame} to {output_path}")
    
    def compile_video(self, frame, output_video_path):
        print(f"Compiling frames into video at {output_video_path}")
    