# Training Workflow (Two-Model YOLO)

This folder contains the scaffolding for training two separate YOLO models:

- **Face detector** in `training/faces/`
- **Explicit-content detector** in `training/explicit/`

## Dataset layout (YOLO format)

```
training/
  faces/
    data.yaml
    dataset/
      images/train
      images/val
      labels/train
      labels/val
  explicit/
    data.yaml
    dataset/
      images/train
      images/val
      labels/train
      labels/val
```

Each label file uses standard YOLO format: `class x_center y_center width height` (normalized).

### Class lists

- Faces: `face`
- Explicit: `genitals`, `breast`, `nipple`, `buttocks`, `cleavage`

## Training

From repo root:

```
python training/train_faces.py
python training/train_explicit.py
```

Both scripts load defaults from `params.yaml`. Override any parameter via CLI flags if needed.

## Face dataset (WIDER FACE)

WIDER FACE can be used for the custom face detector. Download via Kaggle and convert:

```
python training/download.py
python training/convert.py
```

After conversion, the dataset should be placed in:

```
training/faces/dataset/
```

Artifacts are copied into:

```
training/artifacts/faces/<timestamp_run>/
training/artifacts/explicit/<timestamp_run>/
```

Each artifact directory includes `best.pt`, `last.pt`, `results.csv`, and the standard Ultralytics plots.

## Metrics and plots

Ultralytics outputs:

- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`

Additional plots:

```
python training/metrics_report.py --csv <path-to-results.csv> --out <output-dir>
```

This generates:

- `map_curve.png`
- `precision_recall.png`
- `train_loss.png`
- `val_loss.png`
- `metrics_summary.txt`

## DVC tracking (recommended)

Install DVC, then:

```
dvc init
# Optional: track datasets with DVC if you want reproducibility
# dvc add training/faces/dataset
# dvc add training/explicit/dataset

# Track model artifacts
# dvc add training/artifacts/faces
# dvc add training/artifacts/explicit

# Commit the DVC metadata
# git add dvc.yaml .dvc .gitignore
```

## Deployment

Copy the chosen weights into:

```
backend/models/faces/best.pt
backend/models/explicit/best.pt
```

The backend will load these paths by default once YOLO integration is enabled.
