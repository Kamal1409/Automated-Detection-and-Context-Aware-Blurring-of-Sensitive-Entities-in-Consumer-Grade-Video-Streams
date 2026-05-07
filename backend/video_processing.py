import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np

from face_engine import get_face_engine
from llm_manager import LLM
from nsfw_classifier import get_nsfw_classifier, predict_nsfw_score
from pose_detector import PoseDetector
from settings import (
    DEFAULT_EXPLICIT_CLASSIFIER_MODEL,
    DEFAULT_EXPLICIT_MODEL_PATH,
    DEFAULT_EXPLICIT_POSE_MODEL_PATH,
    DEFAULT_FACE_MODEL_PATH,
)
from yolo_detector import YoloDetector

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

try:
    from nudenet import NudeDetector
except Exception:
    NudeDetector = None


def _face_signature(frame, box):
    x, y, w, h = box
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))
    vec = gray.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def _cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    if not isinstance(vec_a, np.ndarray) or not isinstance(vec_b, np.ndarray):
        return 0.0
    if vec_a.shape != vec_b.shape:
        return 0.0
    return float(
        np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8)
    )


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _blur_region(frame, box, strength=1.0):
    x, y, w, h = box
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return

    base_kx = max(15, (w // 3) | 1)
    base_ky = max(15, (h // 3) | 1)
    kx = max(15, int(base_kx * strength) | 1)
    ky = max(15, int(base_ky * strength) | 1)
    frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (kx, ky), 0)


def _clip_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _pose_boxes_from_detection(
    person_box, keypoints, keypoint_scores, frame_shape, keypoint_conf=0.3
):
    height, width = frame_shape[:2]
    x, y, w, h = person_box

    def _valid_point(idx):
        if keypoints is None:
            return None
        if keypoint_scores is not None and keypoint_scores[idx] < keypoint_conf:
            return None
        px, py = keypoints[idx]
        if px <= 0 or py <= 0:
            return None
        return px, py

    shoulders = list(filter(None, [_valid_point(5), _valid_point(6)]))
    hips = list(filter(None, [_valid_point(11), _valid_point(12)]))
    knees = list(filter(None, [_valid_point(13), _valid_point(14)]))

    boxes = []

    if shoulders and hips:
        shoulder_x = [p[0] for p in shoulders]
        shoulder_y = [p[1] for p in shoulders]
        hip_y = [p[1] for p in hips]
        hip_x = [p[0] for p in hips]

        top_y = min(shoulder_y) - 0.08 * h
        mid_y = min(hip_y)
        chest_bottom = top_y + 0.6 * max(10.0, mid_y - top_y)
        chest_x1 = min(shoulder_x + hip_x) - 0.12 * w
        chest_x2 = max(shoulder_x + hip_x) + 0.12 * w
        chest_box = _clip_box(chest_x1, top_y, chest_x2, chest_bottom, width, height)
        if chest_box is not None:
            boxes.append(chest_box)

        knee_y = min([p[1] for p in knees]) if knees else y + 0.85 * h
        pelvis_x1 = min(hip_x) - 0.12 * w
        pelvis_x2 = max(hip_x) + 0.12 * w
        pelvis_y1 = min(hip_y) - 0.05 * h
        pelvis_y2 = knee_y
        pelvis_box = _clip_box(
            pelvis_x1, pelvis_y1, pelvis_x2, pelvis_y2, width, height
        )
        if pelvis_box is not None:
            boxes.append(pelvis_box)

    if not boxes:
        chest_box = _clip_box(x, y, x + w, y + 0.35 * h, width, height)
        pelvis_box = _clip_box(x, y + 0.45 * h, x + w, y + 0.85 * h, width, height)
        if chest_box is not None:
            boxes.append(chest_box)
        if pelvis_box is not None:
            boxes.append(pelvis_box)

    return boxes


STRICT_NUDITY_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def _is_allowed_nudity_label(hit, strict_labels):
    raw_label = str(hit.get("label") or hit.get("class") or "").upper()
    if not raw_label:
        return False
    if not strict_labels:
        return raw_label.endswith("_EXPOSED")
    return raw_label in STRICT_NUDITY_LABELS


def _detect_text_like_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        12,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    height, width = gray.shape[:2]
    min_area = int(0.0004 * width * height)
    max_area = int(0.05 * width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if area < min_area or area > max_area:
            continue
        if h < 12 or h > int(height * 0.2):
            continue
        if aspect < 1.5:
            continue
        candidates.append((x, y, w, h))
    return candidates


def _merge_intervals(intervals, max_gap=0.4):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: item[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(s, 2), round(e, 2)) for s, e in merged]


def _build_llm_track_summaries(tracks, fps, min_seconds, max_tracks):
    summaries = []
    if fps <= 0:
        fps = 1.0

    for track_id, track in tracks.items():
        visible_frames = int(track.get("visible_frames", 0))
        if visible_frames <= 0:
            visible_frames = int(track.get("seen", 0))
        visible_seconds = visible_frames / fps
        if visible_seconds < min_seconds:
            continue

        seen = max(int(track.get("seen", 0)), 1)
        summaries.append(
            {
                "track_id": int(track_id),
                "visible_seconds": round(visible_seconds, 2),
                "avg_center_score": round(
                    float(track.get("center_score", 0.0)) / seen, 4
                ),
                "avg_area_ratio": round(float(track.get("area_score", 0.0)) / seen, 4),
                "first_seen_s": round(float(track.get("first_seen_frame", 0)) / fps, 2),
                "last_seen_s": round(float(track.get("last_seen_frame", 0)) / fps, 2),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["visible_seconds"],
            item["avg_area_ratio"],
            item["avg_center_score"],
        ),
        reverse=True,
    )
    return summaries[: max(1, int(max_tracks))]


def _detect_nudity_scaled(nudity_detector, frame, max_side=768):
    height, width = frame.shape[:2]
    longest = max(width, height)
    scale = 1.0
    infer_frame = frame

    if longest > max_side:
        scale = float(max_side) / float(longest)
        infer_frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    try:
        hits = nudity_detector.detect(infer_frame)
    except Exception:
        return []

    if scale == 1.0:
        return hits

    inv = 1.0 / scale
    scaled_hits = []
    for hit in hits:
        box = hit.get("box", [0, 0, 0, 0])
        x, y, w, h = box
        adjusted = dict(hit)
        adjusted["box"] = [
            int(x * inv),
            int(y * inv),
            int(w * inv),
            int(h * inv),
        ]
        scaled_hits.append(adjusted)
    return scaled_hits


def _resolve_ffmpeg_command():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return "ffmpeg"
    except Exception:
        pass

    if imageio_ffmpeg is None:
        return None

    try:
        embedded_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [embedded_ffmpeg, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return embedded_ffmpeg
    except Exception:
        return None

    return None


def _mux_audio(
    visual_path, original_path, output_path, mute_intervals, keep_audio=True
):
    ffmpeg_cmd = _resolve_ffmpeg_command()

    if ffmpeg_cmd is None:
        shutil.copy2(visual_path, output_path)
        return (
            False,
            "FFmpeg not available, output generated without web-compatible mux step.",
        )

    if not keep_audio:
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            visual_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True
        )
        if proc.returncode != 0:
            shutil.copy2(visual_path, output_path)
            return False, "FFmpeg video encode failed; fell back to raw visual output."
        return True, "Web-compatible video generated without audio by user option."

    mute_intervals = _merge_intervals(mute_intervals)

    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i",
        visual_path,
        "-i",
        original_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    if mute_intervals:
        filters = [
            f"volume=enable='between(t,{s},{e})':volume=0" for s, e in mute_intervals
        ]
        cmd.extend(["-af", ",".join(filters), "-c:a", "aac"])
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.extend(["-shortest", output_path])

    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True
    )
    if proc.returncode != 0:
        shutil.copy2(visual_path, output_path)
        return False, "FFmpeg mux failed; fell back to raw visual output."
    return True, "Web-compatible video generated with audio policy applied."


def _load_reference_signatures(
    face_engine, trusted_face_paths, signature_mode="embedding"
):
    signatures = []
    for path in trusted_face_paths:
        image = cv2.imread(path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            continue

        faces = sorted(
            faces,
            key=lambda face: face["box"][2] * face["box"][3],
            reverse=True,
        )
        if signature_mode == "fallback":
            sig = _face_signature(image, faces[0]["box"])
        else:
            sig = faces[0].get("embedding")
        if sig is not None:
            signatures.append(sig)
    return signatures


def process_video(
    input_path, output_path, options, trusted_face_paths, progress_callback=None
):
    """Run context-aware censoring and return a processing report."""
    if progress_callback:
        progress_callback(0.0, "Initializing detectors", "initializing")

    use_gpu = bool(options.get("use_gpu", True))
    device = 0 if use_gpu else "cpu"
    face_engine = get_face_engine(prefer_gpu=use_gpu)

    face_backend = str(options.get("face_backend", "auto")).strip().lower()
    face_model_path = options.get("face_model_path") or DEFAULT_FACE_MODEL_PATH
    face_detector = None
    use_face_yolo = False
    if face_backend in {"auto", "yolo"}:
        face_detector = YoloDetector(
            face_model_path,
            device=device,
            conf=float(options.get("face_confidence", 0.25)),
            iou=float(options.get("face_iou", 0.5)),
            imgsz=int(options.get("face_imgsz", 640)),
        )
        use_face_yolo = bool(face_detector.available)
    if face_backend == "yolo" and not use_face_yolo:
        face_backend = "insightface"

    explicit_backend = str(options.get("explicit_backend", "auto")).strip().lower()
    explicit_model_path = (
        options.get("explicit_model_path") or DEFAULT_EXPLICIT_MODEL_PATH
    )
    explicit_classifier_model = (
        options.get("explicit_classifier_model") or DEFAULT_EXPLICIT_CLASSIFIER_MODEL
    )
    explicit_pose_model_path = (
        options.get("explicit_pose_model_path") or DEFAULT_EXPLICIT_POSE_MODEL_PATH
    )
    explicit_classifier_threshold = float(
        options.get(
            "explicit_classifier_threshold", options.get("nudity_threshold", 0.55)
        )
    )
    explicit_sample_stride = int(
        options.get("explicit_sample_stride", options.get("nudity_sample_stride", 5))
    )
    explicit_consecutive_hits = int(
        options.get(
            "explicit_consecutive_hits", options.get("nudity_consecutive_hits", 2)
        )
    )
    explicit_hold_frames = int(options.get("explicit_hold_frames", 3))
    explicit_keypoint_conf = float(options.get("explicit_keypoint_confidence", 0.3))
    explicit_classifier = None
    use_falconai = False
    if options.get("detect_nudity", False) and explicit_backend in {"auto", "falconai"}:
        explicit_classifier = get_nsfw_classifier(
            explicit_classifier_model,
            device=0 if use_gpu else -1,
        )
        use_falconai = explicit_classifier is not None
    if use_falconai:
        explicit_backend = "falconai"
    if explicit_backend == "falconai" and not use_falconai:
        explicit_backend = "yolo"

    explicit_detector = None
    use_explicit_yolo = False
    if options.get("detect_nudity", False) and explicit_backend in {"auto", "yolo"}:
        explicit_detector = YoloDetector(
            explicit_model_path,
            device=device,
            conf=float(options.get("nudity_threshold", 0.55)),
            iou=float(options.get("explicit_iou", 0.45)),
            imgsz=int(options.get("explicit_imgsz", 640)),
        )
        use_explicit_yolo = bool(explicit_detector.available)
    if explicit_backend == "yolo" and not use_explicit_yolo:
        explicit_backend = "nudenet"

    pose_detector = None
    use_pose = False
    if options.get("detect_nudity", False) and explicit_backend == "falconai":
        pose_detector = PoseDetector(
            explicit_pose_model_path,
            device=device,
            conf=float(options.get("explicit_pose_confidence", 0.25)),
            iou=float(options.get("explicit_pose_iou", 0.45)),
            imgsz=int(options.get("explicit_pose_imgsz", 640)),
        )
        use_pose = bool(pose_detector.available)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    primary_subject_count = max(1, int(options.get("primary_subject_count", 1)))
    subject_selection_mode = (
        str(options.get("subject_selection_mode", "heuristic")).strip().lower()
    )
    llm_decision_delay = float(options.get("llm_decision_delay_seconds", 6.0))
    llm_min_track_seconds = float(options.get("llm_min_track_seconds", 1.5))
    llm_max_tracks = int(options.get("llm_max_tracks", 10))
    llm_context_hint = str(options.get("llm_context_hint", "")).strip()
    llm_client = LLM() if subject_selection_mode == "llm" else None
    llm_selected_ids = None
    llm_attempted = False
    llm_meta = {
        "enabled": bool(llm_client and llm_client.enabled),
        "attempted": False,
        "selected_ids": [],
        "reason": None,
        "confidence": None,
        "error": None,
    }
    video_context = {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": {"width": width, "height": height},
        "context_hint": llm_context_hint,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="visual_censor_")
    temp_visual_path = os.path.join(temp_dir, "visual.mp4")

    writer = cv2.VideoWriter(
        temp_visual_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    signature_mode = "fallback" if use_face_yolo else "embedding"
    reference_signatures = _load_reference_signatures(
        face_engine, trusted_face_paths, signature_mode=signature_mode
    )

    nudity_detector = None
    if (
        options.get("detect_nudity", False)
        and not use_falconai
        and not use_explicit_yolo
        and NudeDetector is not None
    ):
        try:
            nudity_detector = NudeDetector()
        except Exception:
            nudity_detector = None

    tracks = {}
    next_track_id = 1
    frame_index = 0
    mute_intervals = []
    blurred_face_count = 0
    preserved_face_count = 0
    blurred_text_regions = 0
    blurred_nudity_regions = 0
    nudity_raw_hits = 0
    nudity_filtered_hits = 0
    nudity_positive_streak = 0
    explicit_positive_streak = 0
    explicit_hold_remaining = 0
    explicit_boxes_cache = []
    explicit_frames_flagged = 0
    explicit_scores = []
    smoothing_alpha = float(options.get("temporal_smoothing_alpha", 0.7))
    smoothing_threshold = float(options.get("temporal_blur_threshold", 0.5))
    face_blur_strength = float(options.get("blur_strength_face", 1.2))
    nudity_blur_strength = float(options.get("blur_strength_nudity", 1.55))
    track_max_missed = int(options.get("track_max_missed_frames", 12))
    ghost_blur_frames = int(options.get("ghost_blur_frames", 4))
    has_reference_faces = len(reference_signatures) > 0
    prefer_trusted_only = bool(options.get("prefer_trusted_faces_only", True))
    trusted_threshold = float(options.get("trusted_face_threshold", 0.82))
    configured_stride = int(options.get("face_detect_stride", 0))
    if configured_stride <= 0:
        if face_engine.device == "cuda":
            face_detect_stride = 1
        elif face_engine.device == "directml":
            # DirectML is GPU-backed but usually slower than CUDA for this workload.
            face_detect_stride = 2
        else:
            face_detect_stride = 3
    else:
        face_detect_stride = max(1, min(6, configured_stride))

    if progress_callback:
        progress_callback(3.0, "Running frame analysis", "processing")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track in tracks.values():
            track["missed"] = track.get("missed", 0) + 1

        current = []
        should_detect = (frame_index % face_detect_stride == 0) or (len(tracks) == 0)
        if should_detect:
            if use_face_yolo and face_detector is not None:
                yolo_faces = face_detector.detect(frame)
                detections = [
                    {
                        "box": item["box"],
                        "score": float(item.get("score", 0.0)),
                        "embedding": None,
                    }
                    for item in yolo_faces
                ]
            else:
                detections = face_engine.detect_faces(frame)

            for detection in detections:
                box = tuple(map(int, detection["box"]))
                best_track = None
                best_score = -1.0
                for track_id, track in tracks.items():
                    if track.get("missed", 0) > track_max_missed:
                        continue
                    iou = _iou(box, track["box"])
                    center_ratio = _center_distance_ratio(
                        box, track["box"], width, height
                    )
                    score = iou + (0.35 * max(0.0, 1.0 - (center_ratio * 4.0)))
                    if score > best_score:
                        best_score = score
                        best_track = track_id

                if best_track is not None and best_score >= 0.22:
                    track_id = best_track
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {
                        "box": box,
                        "seen": 0,
                        "center_score": 0.0,
                        "area_score": 0.0,
                        "signature": None,
                        "blur_score": 1.0,
                        "missed": 0,
                        "trust_score": 0.0,
                        "trusted_locked": False,
                        "last_similarity": 0.0,
                        "visible_frames": 0,
                        "first_seen_frame": frame_index,
                        "last_seen_frame": frame_index,
                    }

                x, y, w, h = box
                center_x = x + w * 0.5
                center_y = y + h * 0.5
                center_dist = math.hypot(
                    center_x - width * 0.5, center_y - height * 0.5
                )
                normalized_center = 1.0 - min(
                    1.0,
                    center_dist / (math.hypot(width * 0.5, height * 0.5) + 1e-6),
                )

                track = tracks[track_id]
                track["box"] = box
                track["seen"] += 1
                track["center_score"] += normalized_center
                track["area_score"] += (w * h) / float(width * height)
                track["missed"] = 0
                track["last_seen_frame"] = frame_index

                sig = _face_signature(frame, box)
                if detection.get("embedding") is not None:
                    sig = detection.get("embedding")
                if sig is not None:
                    track["signature"] = sig

                current.append((track_id, box))
        else:
            # Reuse recent boxes between heavyweight detection frames to keep blur stable and reduce latency.
            for track_id, track in tracks.items():
                if int(track.get("missed", 0)) <= 1:
                    track["missed"] = 0
                    current.append((track_id, track["box"]))

        for track_id, _box in current:
            track = tracks.get(track_id)
            if not track:
                continue
            track["visible_frames"] = int(track.get("visible_frames", 0)) + 1
            track["last_seen_frame"] = frame_index

        ranked_tracks = sorted(
            tracks.items(),
            key=lambda item: (item[1]["seen"] * 1.5)
            + item[1]["center_score"]
            + item[1]["area_score"] * 25,
            reverse=True,
        )

        if subject_selection_mode == "llm" and not llm_attempted:
            if fps > 0 and (frame_index / fps) >= llm_decision_delay:
                summaries = _build_llm_track_summaries(
                    tracks, fps, llm_min_track_seconds, llm_max_tracks
                )
                if summaries:
                    llm_attempted = True
                    llm_meta["attempted"] = True
                    if llm_client:
                        keep_ids, meta = llm_client.select_primary_tracks(
                            video_context,
                            summaries,
                            primary_subject_count=primary_subject_count,
                        )
                        llm_meta["error"] = meta.get("error")
                        llm_meta["reason"] = meta.get("reason")
                        llm_meta["confidence"] = meta.get("confidence")
                        if keep_ids:
                            llm_selected_ids = set(keep_ids)
                            llm_meta["selected_ids"] = list(keep_ids)
                    else:
                        llm_meta["error"] = "disabled"

        heuristic_primary_ids = {
            track_id for track_id, _ in ranked_tracks[:primary_subject_count]
        }
        primary_ids = heuristic_primary_ids
        if llm_selected_ids:
            active_llm_ids = {
                track_id for track_id in llm_selected_ids if track_id in tracks
            }
            if active_llm_ids:
                primary_ids = active_llm_ids
        allow_primary_preserve = bool(
            options.get("preserve_primary_subjects", True)
        ) and not (has_reference_faces and prefer_trusted_only)

        for track_id, box in current:
            blur_this_face = options.get("blur_faces", True)
            trusted = False
            signature = tracks[track_id].get("signature")

            if signature is not None and reference_signatures:
                similarities = [
                    _cosine_similarity(signature, ref) for ref in reference_signatures
                ]
                best_similarity = max(similarities)
                track["last_similarity"] = best_similarity
                track["trust_score"] = (0.88 * float(track.get("trust_score", 0.0))) + (
                    0.12 * best_similarity
                )
                if (
                    best_similarity >= trusted_threshold + 0.04
                    or track["trust_score"] >= trusted_threshold + 0.03
                ):
                    track["trusted_locked"] = True
            else:
                track["trust_score"] = 0.995 * float(track.get("trust_score", 0.0))

            if track.get("trusted_locked", False) and track.get("seen", 0) > 20:
                if float(track.get("trust_score", 0.0)) < trusted_threshold - 0.1:
                    track["trusted_locked"] = False

            trusted = bool(track.get("trusted_locked", False)) or float(
                track.get("trust_score", 0.0)
            ) >= (trusted_threshold + 0.02)

            if trusted:
                blur_this_face = False
            elif allow_primary_preserve and track_id in primary_ids:
                blur_this_face = False
            elif options.get("blur_background_faces", True):
                blur_this_face = True
            else:
                blur_this_face = options.get("blur_faces", True)

            if trusted:
                smoothed = 0.0
                track["blur_score"] = 0.0
            else:
                previous = float(
                    track.get("blur_score", 1.0 if blur_this_face else 0.0)
                )
                current_vote = 1.0 if blur_this_face else 0.0
                smoothed = (smoothing_alpha * previous) + (
                    (1.0 - smoothing_alpha) * current_vote
                )
                track["blur_score"] = smoothed

            if smoothed >= smoothing_threshold:
                _blur_region(frame, box, strength=face_blur_strength)
                blurred_face_count += 1
            else:
                preserved_face_count += 1

        if options.get("blur_sensitive_text", False):
            text_boxes = _detect_text_like_regions(frame)
            for box in text_boxes:
                _blur_region(frame, box)
                blurred_text_regions += 1

        frame_has_sensitive = False
        if use_falconai and explicit_classifier is not None:
            if frame_index % max(1, explicit_sample_stride) == 0:
                nsfw_score, _label = predict_nsfw_score(frame, explicit_classifier)
                explicit_scores.append(float(nsfw_score))

                if nsfw_score >= explicit_classifier_threshold:
                    explicit_positive_streak += 1
                else:
                    explicit_positive_streak = 0

                if explicit_positive_streak >= max(1, explicit_consecutive_hits):
                    explicit_hold_remaining = max(0, explicit_hold_frames)
                    if use_pose and pose_detector is not None:
                        pose_detections = pose_detector.detect(frame)
                        explicit_boxes = []
                        for person in pose_detections:
                            explicit_boxes.extend(
                                _pose_boxes_from_detection(
                                    person["box"],
                                    person.get("keypoints"),
                                    person.get("keypoint_scores"),
                                    frame.shape,
                                    keypoint_conf=explicit_keypoint_conf,
                                )
                            )
                        explicit_boxes_cache = explicit_boxes
                    else:
                        explicit_boxes_cache = []

            if explicit_hold_remaining > 0:
                if explicit_boxes_cache:
                    for region in explicit_boxes_cache:
                        _blur_region(frame, region, strength=nudity_blur_strength)
                        blurred_nudity_regions += 1
                explicit_hold_remaining -= 1
                explicit_frames_flagged += 1
                frame_has_sensitive = True

        elif use_explicit_yolo and explicit_detector is not None:
            if frame_index % int(options.get("nudity_sample_stride", 5)) == 0:
                threshold = float(options.get("nudity_threshold", 0.55))
                min_relative_area = float(options.get("nudity_min_relative_area", 0.01))
                min_consecutive_hits = int(options.get("nudity_consecutive_hits", 2))

                detections = explicit_detector.detect(
                    frame,
                    conf=threshold,
                    iou=float(options.get("explicit_iou", 0.45)),
                    imgsz=int(options.get("explicit_imgsz", 640)),
                )

                filtered_boxes = []
                for hit in detections:
                    nudity_raw_hits += 1
                    if float(hit.get("score", 0.0)) < threshold:
                        continue
                    x, y, w, h = hit.get("box", (0, 0, 0, 0))
                    area_ratio = (w * h) / float(width * height)
                    if area_ratio < min_relative_area:
                        continue
                    filtered_boxes.append((x, y, w, h))

                if filtered_boxes:
                    nudity_positive_streak += 1
                else:
                    nudity_positive_streak = 0

                if nudity_positive_streak >= min_consecutive_hits:
                    for nudity_box in filtered_boxes:
                        _blur_region(frame, nudity_box, strength=nudity_blur_strength)
                        blurred_nudity_regions += 1
                        nudity_filtered_hits += 1
                        frame_has_sensitive = True

        elif nudity_detector is not None:
            if frame_index % int(options.get("nudity_sample_stride", 5)) == 0:
                try:
                    nudity_hits = _detect_nudity_scaled(nudity_detector, frame)
                except Exception:
                    nudity_hits = []

                threshold = float(options.get("nudity_threshold", 0.55))
                min_relative_area = float(options.get("nudity_min_relative_area", 0.01))
                strict_labels = bool(options.get("nudity_strict_labels", True))
                min_consecutive_hits = int(options.get("nudity_consecutive_hits", 2))

                filtered_boxes = []
                for hit in nudity_hits:
                    nudity_raw_hits += 1
                    if hit.get("score", 0) < threshold:
                        continue
                    if not _is_allowed_nudity_label(hit, strict_labels):
                        continue
                    x, y, w, h = hit.get("box", [0, 0, 0, 0])
                    x = max(0, int(x))
                    y = max(0, int(y))
                    w = max(1, int(w))
                    h = max(1, int(h))
                    area_ratio = (w * h) / float(width * height)
                    if area_ratio < min_relative_area:
                        continue

                    filtered_boxes.append((x, y, w, h))

                if filtered_boxes:
                    nudity_positive_streak += 1
                else:
                    nudity_positive_streak = 0

                if nudity_positive_streak >= min_consecutive_hits:
                    for nudity_box in filtered_boxes:
                        _blur_region(frame, nudity_box, strength=nudity_blur_strength)
                        blurred_nudity_regions += 1
                        nudity_filtered_hits += 1
                        frame_has_sensitive = True

        if frame_has_sensitive and options.get("censor_sensitive_audio", False):
            t = frame_index / fps
            mute_intervals.append((max(0.0, t - 0.2), min(duration, t + 0.5)))

        writer.write(frame)
        frame_index += 1

        if (
            progress_callback
            and frame_count > 0
            and frame_index % max(1, frame_count // 120) == 0
        ):
            progress = 3.0 + (90.0 * min(1.0, frame_index / frame_count))
            progress_callback(
                progress, f"Processed frame {frame_index}/{frame_count}", "processing"
            )

    cap.release()
    writer.release()

    mute_intervals = _merge_intervals(mute_intervals)
    if progress_callback:
        progress_callback(94.0, "Merging audio track", "finalizing")

    audio_ok, audio_message = _mux_audio(
        temp_visual_path,
        input_path,
        output_path,
        mute_intervals if options.get("censor_sensitive_audio", False) else [],
        keep_audio=options.get("keep_audio", True),
    )

    shutil.rmtree(temp_dir, ignore_errors=True)

    if progress_callback:
        progress_callback(100.0, "Completed", "completed")

    return {
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": {"width": width, "height": height},
        "trusted_reference_faces": len(reference_signatures),
        "faces_blurred": int(blurred_face_count),
        "faces_preserved": int(preserved_face_count),
        "text_regions_blurred": int(blurred_text_regions),
        "nudity_regions_blurred": int(blurred_nudity_regions),
        "nudity_raw_hits": int(nudity_raw_hits),
        "nudity_filtered_hits": int(nudity_filtered_hits),
        "explicit_frames_flagged": int(explicit_frames_flagged),
        "explicit_classifier_scores": explicit_scores[-50:],
        "audio_censored_intervals": mute_intervals,
        "audio_processing": {
            "success": bool(audio_ok),
            "message": audio_message,
        },
        "subject_selection": {
            "mode": subject_selection_mode,
            "primary_subject_count": int(primary_subject_count),
            "llm": llm_meta,
        },
        "engine": {
            "name": "Context Graph Censor v1",
            "description": "Identity-aware face trust + scene-priority tracks + temporal smoothing + selective modality censoring",
            "face_backend": "yolo" if use_face_yolo else face_engine.backend,
            "face_device": face_engine.device,
            "face_providers": list(getattr(face_engine, "providers", [])),
            "face_detect_stride": int(face_detect_stride),
            "gpu_requested": bool(options.get("use_gpu", True)),
            "gpu_active": bool(face_engine.device in {"cuda", "directml"}),
            "explicit_backend": (
                "falconai"
                if use_falconai
                else ("yolo" if use_explicit_yolo else "nudenet")
            ),
            "explicit_model_path": explicit_model_path if use_explicit_yolo else None,
            "explicit_classifier_model": (
                explicit_classifier_model if use_falconai else None
            ),
        },
    }
