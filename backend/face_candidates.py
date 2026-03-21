import base64
import math

import cv2
import numpy as np


def _crop_with_padding(frame, box, pad_ratio=0.18):
    x, y, w, h = box
    height, width = frame.shape[:2]
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)
    crop = frame[y1:y2, x1:x2]
    return crop


def _face_signature(crop):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (40, 40))
    vec = gray.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def _cosine_similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _encode_as_data_url(image):
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_face_candidates(
    video_path,
    max_candidates=12,
    sample_stride=6,
    max_seconds=45,
):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for face analysis.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    max_frames = int(max_seconds * fps)

    tracks = {}
    next_id = 1
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index > max_frames:
            break
        if frame_index % max(1, sample_stride) != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(28, 28)
        )
        if len(detections) == 0:
            continue

        for box in detections:
            box = tuple(map(int, box))
            crop = _crop_with_padding(frame, box)
            signature = _face_signature(crop)
            if signature is None:
                continue

            best_track_id = None
            best_score = 0.0
            for track_id, track in tracks.items():
                score = _cosine_similarity(signature, track["signature"])
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            x, y, w, h = box
            area = w * h
            center_dist = math.hypot(
                (x + w * 0.5) - (frame.shape[1] * 0.5),
                (y + h * 0.5) - (frame.shape[0] * 0.5),
            )
            center_score = 1.0 - min(
                1.0,
                center_dist
                / (math.hypot(frame.shape[1] * 0.5, frame.shape[0] * 0.5) + 1e-6),
            )

            if best_track_id is not None and best_score >= 0.88:
                track = tracks[best_track_id]
                track["seen"] += 1
                track["score"] += center_score
                if area > track["best_area"]:
                    track["best_area"] = area
                    track["crop"] = crop
                track["signature"] = signature
            else:
                tracks[next_id] = {
                    "id": next_id,
                    "seen": 1,
                    "score": center_score,
                    "best_area": area,
                    "crop": crop,
                    "signature": signature,
                }
                next_id += 1

    cap.release()

    ranked = sorted(
        tracks.values(),
        key=lambda t: (t["seen"] * 1.7) + t["score"] + (t["best_area"] / 1000.0),
        reverse=True,
    )

    payload = []
    for track in ranked[: max(1, int(max_candidates))]:
        image_data_url = _encode_as_data_url(track["crop"])
        if not image_data_url:
            continue
        payload.append(
            {
                "candidate_id": f"face_{track['id']}",
                "image_data_url": image_data_url,
                "frames_seen": int(track["seen"]),
            }
        )

    return payload
