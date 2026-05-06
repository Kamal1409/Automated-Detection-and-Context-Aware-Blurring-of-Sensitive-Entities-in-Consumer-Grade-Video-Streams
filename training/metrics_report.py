import argparse
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency check
    pd = None
    plt = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def main():
    parser = argparse.ArgumentParser(
        description="Generate training plots from Ultralytics results.csv"
    )
    parser.add_argument("--csv", required=True, help="Path to results.csv")
    parser.add_argument("--out", required=True, help="Directory to save plots")
    args = parser.parse_args()

    if pd is None or plt is None:
        raise SystemExit(f"pandas/matplotlib not installed: {_IMPORT_ERROR}")

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)

    plt.figure()
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("mAP vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "map_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision vs Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "precision_recall.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "train_loss.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "val_loss.png"), dpi=300)
    plt.close()

    last = df.iloc[-1]
    best_idx = df["metrics/mAP50(B)"].idxmax()
    best = df.loc[best_idx]

    summary_path = os.path.join(args.out, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("FINAL METRICS\n")
        handle.write(f"mAP50: {last['metrics/mAP50(B)']:.4f}\n")
        handle.write(f"mAP50-95: {last['metrics/mAP50-95(B)']:.4f}\n")
        handle.write(f"Precision: {last['metrics/precision(B)']:.4f}\n")
        handle.write(f"Recall: {last['metrics/recall(B)']:.4f}\n\n")
        handle.write("BEST EPOCH\n")
        handle.write(f"Epoch: {int(best['epoch'])}\n")
        handle.write(f"Best mAP50: {best['metrics/mAP50(B)']:.4f}\n")


if __name__ == "__main__":
    main()
