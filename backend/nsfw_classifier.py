import os
from functools import lru_cache

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from PIL import Image
except Exception:
    Image = None


DEFAULT_MODEL = os.getenv("EXPLICIT_CLASSIFIER_MODEL", "Falconsai/nsfw_image_detection")


def _as_device_index(device):
    if isinstance(device, int):
        return device
    if device is None:
        return -1
    raw = str(device).strip().lower()
    if raw in {"cpu", "-1"}:
        return -1
    if raw in {"cuda", "gpu", "0"}:
        return 0
    return -1


@lru_cache(maxsize=4)
def get_nsfw_classifier(model_name=DEFAULT_MODEL, device=-1):
    if pipeline is None:
        return None
    if Image is None:
        return None

    device_index = _as_device_index(device)
    try:
        return pipeline(
            "image-classification",
            model=model_name or DEFAULT_MODEL,
            device=device_index,
        )
    except Exception:
        return None


def predict_nsfw_score(frame_bgr, classifier):
    if classifier is None or Image is None:
        return 0.0, "unknown"

    try:
        image = Image.fromarray(frame_bgr[:, :, ::-1])
    except Exception:
        return 0.0, "unknown"

    try:
        outputs = classifier(image)
    except Exception:
        return 0.0, "unknown"

    nsfw_score = 0.0
    nsfw_label = "unknown"
    for item in outputs:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "nsfw" in label:
            nsfw_score = score
            nsfw_label = label
            break

    return nsfw_score, nsfw_label
