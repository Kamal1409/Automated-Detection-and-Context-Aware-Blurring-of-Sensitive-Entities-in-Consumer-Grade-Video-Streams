import argparse
import csv
import json
import os
import shutil
from datetime import datetime

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency check
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    import yaml
except Exception as exc:  # pragma: no cover - runtime dependency check
    yaml = None
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None


def _read_metrics(csv_path):
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {}

    def _as_float(value):
        try:
            return float(value)
        except Exception:
            return 0.0

    last = rows[-1]
    best = max(rows, key=lambda row: _as_float(row.get("metrics/mAP50(B)", 0.0)))
    return {"last": last, "best": best}


def _copy_run_artifacts(run_dir, output_root, params_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = os.path.basename(os.path.abspath(run_dir))
    artifact_dir = os.path.join(output_root, f"{timestamp}_{run_name}")
    os.makedirs(artifact_dir, exist_ok=True)

    candidates = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]

    for rel_path in candidates:
        src = os.path.join(run_dir, rel_path)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, os.path.basename(src)))

    if params_path and os.path.exists(params_path):
        shutil.copy2(params_path, os.path.join(artifact_dir, "params.yaml"))

    metrics = _read_metrics(os.path.join(run_dir, "results.csv"))
    if metrics:
        with open(
            os.path.join(artifact_dir, "metrics_summary.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2)

    return artifact_dir


def _load_params(path):
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise SystemExit(f"pyyaml is not installed: {_YAML_ERROR}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _get_section(params, *keys):
    data = params
    for key in keys:
        if not isinstance(data, dict):
            return {}
        data = data.get(key, {})
    return data if isinstance(data, dict) else {}


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--params", default=os.path.join(repo_root, "params.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    params = _load_params(pre_args.params)
    cfg = _get_section(params, "training", "explicit")
    artifacts_cfg = _get_section(params, "artifacts")

    parser = argparse.ArgumentParser(
        description="Train YOLO explicit-content detector", parents=[pre_parser]
    )
    parser.add_argument("--model", default=cfg.get("model", "yolov8n.pt"))
    parser.add_argument(
        "--data",
        default=cfg.get("data", os.path.join(base_dir, "explicit", "data.yaml")),
    )
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 120))
    parser.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 8))
    parser.add_argument("--workers", type=int, default=cfg.get("workers", 6))
    parser.add_argument("--device", default=cfg.get("device", 0))
    parser.add_argument("--cache", default=cfg.get("cache", "disk"))
    parser.add_argument("--patience", type=int, default=cfg.get("patience", 25))
    parser.add_argument(
        "--project",
        default=cfg.get("project", os.path.join(base_dir, "runs", "explicit")),
    )
    parser.add_argument("--name", default=cfg.get("name", "explicit_model"))
    parser.add_argument(
        "--artifact-root",
        default=artifacts_cfg.get(
            "explicit", os.path.join(base_dir, "artifacts", "explicit")
        ),
    )
    parser.add_argument("--save-period", type=int, default=cfg.get("save_period", 10))
    parser.add_argument("--plots", type=_coerce_bool, default=cfg.get("plots", True))
    parser.add_argument("--val", type=_coerce_bool, default=cfg.get("val", True))
    parser.add_argument("--lr0", type=float, default=cfg.get("lr0", 0.008))
    parser.add_argument("--lrf", type=float, default=cfg.get("lrf", 0.01))
    parser.add_argument("--momentum", type=float, default=cfg.get("momentum", 0.937))
    parser.add_argument(
        "--weight-decay", type=float, default=cfg.get("weight_decay", 0.0005)
    )
    parser.add_argument("--hsv-h", type=float, default=cfg.get("hsv_h", 0.02))
    parser.add_argument("--hsv-s", type=float, default=cfg.get("hsv_s", 0.7))
    parser.add_argument("--hsv-v", type=float, default=cfg.get("hsv_v", 0.4))
    parser.add_argument("--degrees", type=float, default=cfg.get("degrees", 8))
    parser.add_argument("--translate", type=float, default=cfg.get("translate", 0.1))
    parser.add_argument("--scale", type=float, default=cfg.get("scale", 0.5))
    parser.add_argument("--fliplr", type=float, default=cfg.get("fliplr", 0.5))
    args = parser.parse_args()

    if YOLO is None:
        raise SystemExit(f"ultralytics is not installed: {_IMPORT_ERROR}")

    os.makedirs(args.project, exist_ok=True)
    os.makedirs(args.artifact_root, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        project=args.project,
        name=args.name,
        save=True,
        save_period=args.save_period,
        plots=args.plots,
        val=args.val,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        patience=args.patience,
    )

    run_dir = getattr(results, "save_dir", None) or os.path.join(
        args.project, args.name
    )
    artifact_dir = _copy_run_artifacts(run_dir, args.artifact_root, args.params)
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
