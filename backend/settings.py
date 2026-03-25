import os

UPLOAD_FOLDER = "temp_storage"
PROCESSED_FOLDER = "processed_videos"
TRUSTED_FOLDER = os.path.join(UPLOAD_FOLDER, "trusted_faces")

MAX_FILE_SIZE = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "wmv", "flv", "webm", "mkv"}
TEMP_FILE_EXPIRY_HOURS = 24

DEFAULT_OPTIONS = {
    "blur_faces": True,
    "blur_background_faces": True,
    "preserve_primary_subjects": True,
    "primary_subject_count": 1,
    "blur_sensitive_text": False,
    "detect_nudity": True,
    "nudity_policy_mode": "streaming_strict",
    "censor_sensitive_audio": True,
    "keep_audio": True,
    "trusted_face_threshold": 0.82,
    "nudity_threshold": 0.82,
    "nudity_sample_stride": 5,
    "nudity_min_relative_area": 0.01,
    "nudity_consecutive_hits": 2,
    "nudity_strict_labels": True,
    "blur_strength_face": 1.2,
    "blur_strength_nudity": 1.55,
    "temporal_smoothing_alpha": 0.7,
    "temporal_blur_threshold": 0.5,
    "face_detect_stride": 0,
}


def ensure_directories():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(TRUSTED_FOLDER, exist_ok=True)
