import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np

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


def _load_reference_signatures(face_detector, trusted_face_paths):
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

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        sig = _face_signature(image, tuple(map(int, faces[0])))
        if sig is not None:
            signatures.append(sig)
    return signatures


def process_video(
    input_path, output_path, options, trusted_face_paths, progress_callback=None
):
    """Run context-aware censoring and return a processing report."""
    if progress_callback:
        progress_callback(0.0, "Initializing detectors", "initializing")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="visual_censor_")
    temp_visual_path = os.path.join(temp_dir, "visual.mp4")

    writer = cv2.VideoWriter(
        temp_visual_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    reference_signatures = _load_reference_signatures(face_detector, trusted_face_paths)

    nudity_detector = None
    if options.get("detect_nudity", False) and NudeDetector is not None:
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
    smoothing_alpha = float(options.get("temporal_smoothing_alpha", 0.7))
    smoothing_threshold = float(options.get("temporal_blur_threshold", 0.5))
    face_blur_strength = float(options.get("blur_strength_face", 1.2))
    nudity_blur_strength = float(options.get("blur_strength_nudity", 1.55))

    if progress_callback:
        progress_callback(3.0, "Running frame analysis", "processing")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(30, 30)
        )

        current = []
        for box in detections:
            box = tuple(map(int, box))
            best_track = None
            best_iou = 0.0
            for track_id, track in tracks.items():
                score = _iou(box, track["box"])
                if score > best_iou:
                    best_iou = score
                    best_track = track_id

            if best_track is not None and best_iou > 0.3:
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
                }

            x, y, w, h = box
            center_x = x + w * 0.5
            center_y = y + h * 0.5
            center_dist = math.hypot(center_x - width * 0.5, center_y - height * 0.5)
            normalized_center = 1.0 - min(
                1.0, center_dist / (math.hypot(width * 0.5, height * 0.5) + 1e-6)
            )

            track = tracks[track_id]
            track["box"] = box
            track["seen"] += 1
            track["center_score"] += normalized_center
            track["area_score"] += (w * h) / float(width * height)

            sig = _face_signature(frame, box)
            if sig is not None:
                track["signature"] = sig

            current.append((track_id, box))

        ranked_tracks = sorted(
            tracks.items(),
            key=lambda item: (item[1]["seen"] * 1.5)
            + item[1]["center_score"]
            + item[1]["area_score"] * 25,
            reverse=True,
        )
        top_k = max(1, int(options.get("primary_subject_count", 1)))
        primary_ids = {track_id for track_id, _ in ranked_tracks[:top_k]}

        for track_id, box in current:
            blur_this_face = options.get("blur_faces", True)
            trusted = False
            signature = tracks[track_id].get("signature")

            if signature is not None and reference_signatures:
                similarities = [
                    _cosine_similarity(signature, ref) for ref in reference_signatures
                ]
                trusted = max(similarities) >= float(
                    options.get("trusted_face_threshold", 0.82)
                )

            if trusted:
                blur_this_face = False
            elif (
                options.get("preserve_primary_subjects", True)
                and track_id in primary_ids
            ):
                blur_this_face = False
            elif options.get("blur_background_faces", True):
                blur_this_face = True
            else:
                blur_this_face = options.get("blur_faces", True)

            previous = float(
                tracks[track_id].get("blur_score", 1.0 if blur_this_face else 0.0)
            )
            current_vote = 1.0 if blur_this_face else 0.0
            smoothed = (smoothing_alpha * previous) + (
                (1.0 - smoothing_alpha) * current_vote
            )
            tracks[track_id]["blur_score"] = smoothed

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
        if nudity_detector is not None:
            if frame_index % int(options.get("nudity_sample_stride", 5)) == 0:
                try:
                    nudity_hits = nudity_detector.detect(frame)
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
        "audio_censored_intervals": mute_intervals,
        "audio_processing": {
            "success": bool(audio_ok),
            "message": audio_message,
        },
        "engine": {
            "name": "Context Graph Censor v1",
            "description": "Identity-aware face trust + scene-priority tracks + temporal smoothing + selective modality censoring",
        },
    }
