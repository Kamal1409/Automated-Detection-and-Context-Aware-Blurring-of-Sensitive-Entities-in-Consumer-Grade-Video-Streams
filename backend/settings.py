import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH")
if not DEFAULT_FACE_MODEL_PATH:
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "results", "runs", "spixgro_face_detector", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "results", "yolo26n.pt"),
        os.path.join(MODEL_DIR, "faces", "best.pt"),
    ]
    DEFAULT_FACE_MODEL_PATH = next(
        (path for path in candidate_paths if os.path.exists(path)),
        candidate_paths[-1],
    )
DEFAULT_EXPLICIT_MODEL_PATH = os.getenv(
    "EXPLICIT_MODEL_PATH", os.path.join(MODEL_DIR, "explicit", "best.pt")
)
DEFAULT_EXPLICIT_CLASSIFIER_MODEL = os.getenv(
    "EXPLICIT_CLASSIFIER_MODEL", "Falconsai/nsfw_image_detection"
)
DEFAULT_EXPLICIT_POSE_MODEL_PATH = os.getenv(
    "EXPLICIT_POSE_MODEL_PATH", "yolov8n-pose.pt"
)

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
    "subject_selection_mode": "heuristic",
    "llm_decision_delay_seconds": 6.0,
    "llm_min_track_seconds": 1.5,
    "llm_max_tracks": 10,
    "llm_context_hint": "",
    "blur_sensitive_text": False,
    "detect_nudity": True,
    "explicit_backend": "nudenet",
    "explicit_model_path": "",
    "explicit_iou": 0.45,
    "explicit_imgsz": 640,
    "explicit_classifier_model": DEFAULT_EXPLICIT_CLASSIFIER_MODEL,
    "explicit_classifier_threshold": 0.45,
    "explicit_sample_stride": 5,
    "explicit_consecutive_hits": 2,
    "explicit_hold_frames": 3,
    "explicit_pose_model_path": DEFAULT_EXPLICIT_POSE_MODEL_PATH,
    "explicit_pose_confidence": 0.25,
    "explicit_pose_iou": 0.45,
    "explicit_pose_imgsz": 640,
    "explicit_keypoint_confidence": 0.3,
    "nudity_policy_mode": "streaming_strict",
    "censor_sensitive_audio": True,
    "keep_audio": True,
    "trusted_face_threshold": 0.82,
    "nudity_threshold": 0.82,
    "nudity_sample_stride": 5,
    "nudity_min_relative_area": 0.01,
    "nudity_consecutive_hits": 2,
    "nudity_strict_labels": True,
    "face_backend": "auto",
    "face_model_path": "",
    "face_confidence": 0.25,
    "face_iou": 0.5,
    "face_imgsz": 640,
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
