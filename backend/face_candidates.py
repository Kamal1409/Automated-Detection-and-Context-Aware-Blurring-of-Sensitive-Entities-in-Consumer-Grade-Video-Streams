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
    if a is None or b is None:
        return 0.0
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _signature_similarity(sig_a, sig_b):
    dct_sim = _cosine_similarity(sig_a["dct"], sig_b["dct"])
    hist_sim = _cosine_similarity(sig_a["hist"], sig_b["hist"])
    return (0.82 * dct_sim) + (0.18 * hist_sim)


def _face_quality(crop, box, frame_shape):
    if crop is None or crop.size == 0:
        return 0.0
    x, y, w, h = box
    area_ratio = (w * h) / float(frame_shape[0] * frame_shape[1] + 1e-6)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_score = min(1.0, sharpness / 180.0)

    center_dist = math.hypot(
        (x + w * 0.5) - (frame_shape[1] * 0.5), (y + h * 0.5) - (frame_shape[0] * 0.5)
    )
    center_score = 1.0 - min(
        1.0,
        center_dist / (math.hypot(frame_shape[1] * 0.5, frame_shape[0] * 0.5) + 1e-6),
    )
    center_dist = math.hypot(
        (x + w * 0.5) - (frame_shape[1] * 0.5), (y + h * 0.5) - (frame_shape[0] * 0.5)
    )
    center_score = 1.0 - min(
        1.0,
        center_dist / (math.hypot(frame_shape[1] * 0.5, frame_shape[0] * 0.5) + 1e-6),
    )
    size_score = min(1.0, area_ratio / 0.06)

    return (0.55 * size_score) + (0.3 * sharp_score) + (0.15 * center_score)


def _merge_boxes_nms(boxes, iou_threshold=0.35):
    if not boxes:
        return []

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        union = (aw * ah) + (bw * bh) - inter
        if union <= 0:
            return 0.0
        return inter / union

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _detect_faces_multi(detector_frontal, detector_profile, gray):
    frontal = detector_frontal.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(28, 28)
    )
    profile_left = detector_profile.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28)
    )
    frontal = detector_frontal.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(28, 28)
    )
    profile_left = detector_profile.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28)
    )

    flipped = cv2.flip(gray, 1)
    profile_right_raw = detector_profile.detectMultiScale(
        flipped, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28)
    )
    profile_right_raw = detector_profile.detectMultiScale(
        flipped, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28)
    )
    width = gray.shape[1]
    profile_right = []
    for x, y, w, h in profile_right_raw:
        profile_right.append((width - x - w, y, w, h))

    merged = [tuple(map(int, b)) for b in frontal]
    merged.extend(tuple(map(int, b)) for b in profile_left)
    merged.extend(tuple(map(int, b)) for b in profile_right)
    return _merge_boxes_nms(merged)


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
                track_sig = track["signature"]
                if isinstance(signature, dict) and isinstance(track_sig, dict):
                    score = _signature_similarity(signature, track_sig)
                elif isinstance(signature, np.ndarray) and isinstance(
                    track_sig, np.ndarray
                ):
                    score = _cosine_similarity(signature, track_sig)
                else:
                    score = 0.0
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
                if isinstance(signature, dict) and isinstance(track["signature"], dict):
                    track["signature"]["dct"] = (
                        0.8 * track["signature"]["dct"] + 0.2 * signature["dct"]
                    )
                    track["signature"]["hist"] = (
                        0.8 * track["signature"]["hist"] + 0.2 * signature["hist"]
                    )
                elif isinstance(signature, np.ndarray) and isinstance(
                    track["signature"], np.ndarray
                ):
                    track["signature"] = 0.85 * track["signature"] + 0.15 * signature
                else:
                    track["signature"] = 0.85 * track["signature"] + 0.15 * signature
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
