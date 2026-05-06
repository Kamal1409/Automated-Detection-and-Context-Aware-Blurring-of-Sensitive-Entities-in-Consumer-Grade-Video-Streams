from settings import DEFAULT_OPTIONS


def normalize_options(raw_options):
    options = DEFAULT_OPTIONS.copy()
    options.update(raw_options or {})

    options["blur_faces"] = bool(options.get("blur_faces", True))
    options["blur_background_faces"] = bool(options.get("blur_background_faces", True))
    options["preserve_primary_subjects"] = bool(
        options.get("preserve_primary_subjects", True)
    )
    options["blur_sensitive_text"] = bool(options.get("blur_sensitive_text", False))
    options["detect_nudity"] = bool(options.get("detect_nudity", True))
    explicit_backend = str(options.get("explicit_backend", "auto")).strip().lower()
    if explicit_backend not in {"auto", "falconai", "yolo", "nudenet"}:
        explicit_backend = "auto"
    options["explicit_backend"] = explicit_backend
    options["explicit_model_path"] = str(options.get("explicit_model_path", "")).strip()
    options["explicit_classifier_model"] = str(
        options.get("explicit_classifier_model", "")
    ).strip()
    mode = str(options.get("nudity_policy_mode", "streaming_strict")).strip().lower()
    if mode not in {"porn_only", "balanced", "streaming_strict"}:
        mode = "streaming_strict"
    options["nudity_policy_mode"] = mode
    options["censor_sensitive_audio"] = bool(
        options.get("censor_sensitive_audio", True)
    )
    options["keep_audio"] = bool(options.get("keep_audio", True))

    options["primary_subject_count"] = max(
        1, min(5, int(options.get("primary_subject_count", 1)))
    )
    options["trusted_face_threshold"] = max(
        0.4, min(0.98, float(options.get("trusted_face_threshold", 0.82)))
    )
    options["nudity_threshold"] = max(
        0.2, min(0.99, float(options.get("nudity_threshold", 0.55)))
    )
    options["nudity_sample_stride"] = max(
        1, min(24, int(options.get("nudity_sample_stride", 5)))
    )
    options["nudity_min_relative_area"] = max(
        0.001, min(0.2, float(options.get("nudity_min_relative_area", 0.01)))
    )
    options["nudity_consecutive_hits"] = max(
        1, min(6, int(options.get("nudity_consecutive_hits", 2)))
    )
    options["explicit_classifier_threshold"] = max(
        0.05,
        min(
            0.99,
            float(
                options.get(
                    "explicit_classifier_threshold",
                    options.get("nudity_threshold", 0.55),
                )
            ),
        ),
    )
    options["explicit_sample_stride"] = max(
        1,
        min(
            24,
            int(
                options.get(
                    "explicit_sample_stride", options.get("nudity_sample_stride", 5)
                )
            ),
        ),
    )
    options["explicit_consecutive_hits"] = max(
        1,
        min(
            6,
            int(
                options.get(
                    "explicit_consecutive_hits",
                    options.get("nudity_consecutive_hits", 2),
                )
            ),
        ),
    )
    options["explicit_hold_frames"] = max(
        0, min(12, int(options.get("explicit_hold_frames", 3)))
    )
    options["explicit_pose_model_path"] = str(
        options.get("explicit_pose_model_path", "")
    ).strip()
    options["explicit_pose_confidence"] = max(
        0.1, min(0.95, float(options.get("explicit_pose_confidence", 0.25)))
    )
    options["explicit_pose_iou"] = max(
        0.1, min(0.95, float(options.get("explicit_pose_iou", 0.45)))
    )
    options["explicit_pose_imgsz"] = max(
        320, min(1280, int(options.get("explicit_pose_imgsz", 640)))
    )
    options["explicit_keypoint_confidence"] = max(
        0.1, min(0.95, float(options.get("explicit_keypoint_confidence", 0.3)))
    )
    options["nudity_strict_labels"] = bool(options.get("nudity_strict_labels", True))
    face_backend = str(options.get("face_backend", "auto")).strip().lower()
    if face_backend not in {"auto", "yolo", "insightface"}:
        face_backend = "auto"
    options["face_backend"] = face_backend
    options["face_model_path"] = str(options.get("face_model_path", "")).strip()
    options["explicit_iou"] = max(
        0.1, min(0.95, float(options.get("explicit_iou", 0.45)))
    )
    options["explicit_imgsz"] = max(
        320, min(1280, int(options.get("explicit_imgsz", 640)))
    )
    options["face_confidence"] = max(
        0.1, min(0.95, float(options.get("face_confidence", 0.25)))
    )
    options["face_iou"] = max(0.1, min(0.95, float(options.get("face_iou", 0.5))))
    options["face_imgsz"] = max(320, min(1280, int(options.get("face_imgsz", 640))))
    options["blur_strength_face"] = max(
        0.6, min(2.5, float(options.get("blur_strength_face", 1.2)))
    )
    options["blur_strength_nudity"] = max(
        0.8, min(3.0, float(options.get("blur_strength_nudity", 1.55)))
    )
    options["temporal_smoothing_alpha"] = max(
        0.05, min(0.95, float(options.get("temporal_smoothing_alpha", 0.7)))
    )
    options["temporal_blur_threshold"] = max(
        0.2, min(0.95, float(options.get("temporal_blur_threshold", 0.5)))
    )
    options["face_detect_stride"] = max(
        0, min(6, int(options.get("face_detect_stride", 0)))
    )

    return options
