import os
import shutil
from datetime import datetime, timedelta

import cv2

from settings import ALLOWED_EXTENSIONS


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_info(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "duration": round(duration, 2),
            "frame_count": int(frame_count),
        }
    except Exception:
        return None


def cleanup_expired_files(folders, expiry_hours):
    now = datetime.now()
    for folder in folders:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            modified = datetime.fromtimestamp(os.path.getmtime(path))
            if now - modified <= timedelta(hours=expiry_hours):
                continue

            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
