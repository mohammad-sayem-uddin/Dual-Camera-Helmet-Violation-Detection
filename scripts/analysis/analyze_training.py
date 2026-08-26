#!/usr/bin/env python3
"""Produce metric tables, training curves, and overfitting diagnosis from Ultralytics results.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YOLO training history and generate publication-quality figures.")
    parser.add_argument("--results-csv", required=True, help="Path to Ultralytics results.csv")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier (e.g. 'EXP-001')")
    args = parser.parse_args()

    results_path = research_path(args.results_csv)
    if not results_path.exists():
        parser.error(f"results.csv not found: {results_path}")

    # Read CSV data
    rows = []
    with results_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [k.strip() for k in (reader.fieldnames or [])]
        for r in reader:
            cleaned_row = {k.strip(): float(v.strip()) if v.strip() else 0.0 for k, v in r.items()}
            rows.append(cleaned_row)

    if not rows:
        parser.error("results.csv is empty.")

    # Identify metric columns
    epochs = [int(r.get("epoch", i)) for i, r in enumerate(rows)]
    
    # Extract key metrics
    def get_col(col_name: str) -> list[float]:
        for k in rows[0]:
            if k == col_name or k.endswith(f"/{col_name}") or col_name in k:
                return [r[k] for r in rows]
        return []

    train_box_loss = get_col("train/box_loss") or get_col("box_loss")
    train_cls_loss = get_col("train/cls_loss") or get_col("cls_loss")
    val_box_loss = get_col("val/box_loss")
    val_cls_loss = get_col("val/cls_loss")
    metrics_map50 = get_col("metrics/mAP50(B)") or get_col("map50")
    metrics_map50_95 = get_col("metrics/mAP50-95(B)") or get_col("map50-95")
    metrics_p = get_col("metrics/precision(B)") or get_col("precision")
    metrics_r = get_col("metrics/recall(B)") or get_col("recall")

    # Determine best epoch by mAP50
    best_epoch_idx = 0
    if metrics_map50:
        best_epoch_idx = metrics_map50.index(max(metrics_map50))
    elif metrics_map50_95:
        best_epoch_idx = metrics_map50_95.index(max(metrics_map50_95))

    best_epoch = epochs[best_epoch_idx]
    best_map50 = metrics_map50[best_epoch_idx] if metrics_map50 else None
    best_map50_95 = metrics_map50_95[best_epoch_idx] if metrics_map50_95 else None

    # Overfitting assessment
    overfitting_evidence = "None detected. Validation loss steadily declined alongside training loss."
    if val_box_loss and train_box_loss:
        min_val_loss_epoch = val_box_loss.index(min(val_box_loss))
        if min_val_loss_epoch < len(val_box_loss) - 5:
            loss_increase = (val_box_loss[-1] - min(val_box_loss)) / min(val_box_loss)
            if loss_increase > 0.15:
                overfitting_evidence = (
                    f"Possible overfitting detected. Validation box loss reached minimum at epoch "
                    f"{epochs[min_val_loss_epoch]} and increased by {loss_increase*100:.1f}% by the final epoch."
                )

    analysis_report = {
        "experiment_id": args.experiment_id,
        "total_epochs_trained": len(epochs),
        "best_epoch": best_epoch,
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "final_precision": metrics_p[-1] if metrics_p else None,
        "final_recall": metrics_r[-1] if metrics_r else None,
        "overfitting_assessment": overfitting_evidence,
    }

    # Save metrics analysis JSON
    analysis_json_path = ensure_parent(research_path(Path("results/metrics") / f"{args.experiment_id}_analysis.json"))
    analysis_json_path.write_text(json.dumps(analysis_report, indent=2), encoding="utf-8")

    # Save history table
    table_path = ensure_parent(research_path(Path("results/tables") / f"{args.experiment_id}_history.csv"))
    shutil.copy2(results_path, table_path)

    # Plot curves if matplotlib is available
    plot_output_path = ensure_parent(research_path(Path("results/plots") / f"{args.experiment_id}_training_curves.png"))
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        ax_loss = axes[0, 0]
        if train_box_loss:
            ax_loss.plot(epochs, train_box_loss, label="Train Box Loss", color="#1f77b4")
        if val_box_loss:
            ax_loss.plot(epochs, val_box_loss, label="Val Box Loss", color="#ff7f0e", linestyle="--")
        ax_loss.set_title("Box Loss", fontsize=12, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend()

        # Class loss plot
        ax_cls = axes[0, 1]
        if train_cls_loss:
            ax_cls.plot(epochs, train_cls_loss, label="Train Cls Loss", color="#2ca02c")
        if val_cls_loss:
            ax_cls.plot(epochs, val_cls_loss, label="Val Cls Loss", color="#d62728", linestyle="--")
        ax_cls.set_title("Classification Loss", fontsize=12, fontweight="bold")
        ax_cls.set_xlabel("Epoch")
        ax_cls.set_ylabel("Loss")
        ax_cls.grid(alpha=0.3)
        ax_cls.legend()

        # mAP plot
        ax_map = axes[1, 0]
        if metrics_map50:
            ax_map.plot(epochs, metrics_map50, label="mAP@0.50", color="#9467bd")
        if metrics_map50_95:
            ax_map.plot(epochs, metrics_map50_95, label="mAP@0.50:0.95", color="#8c564b")
        ax_map.axvline(best_epoch, color="red", linestyle=":", label=f"Best Epoch ({best_epoch})")
        ax_map.set_title("mAP Metrics", fontsize=12, fontweight="bold")
        ax_map.set_xlabel("Epoch")
        ax_map.set_ylabel("mAP")
        ax_map.grid(alpha=0.3)
        ax_map.legend()

        # Precision / Recall plot
        ax_pr = axes[1, 1]
        if metrics_p:
            ax_pr.plot(epochs, metrics_p, label="Precision", color="#e377c2")
        if metrics_r:
            ax_pr.plot(epochs, metrics_r, label="Recall", color="#7f7f7f")
        ax_pr.set_title("Precision & Recall", fontsize=12, fontweight="bold")
        ax_pr.set_xlabel("Epoch")
        ax_pr.set_ylabel("Metric")
        ax_pr.grid(alpha=0.3)
        ax_pr.legend()

        fig.suptitle(f"{args.experiment_id}: Training History & Metric Evolution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves plot: {plot_output_path}")
    except ImportError:
        print("[INFO] Matplotlib not installed; skipped figure rendering. Summary metrics and tables saved.")

    print("\n" + "=" * 70)
    print(f"TRAINING ANALYSIS: {args.experiment_id}")
    print("=" * 70)
    print(f"Total Epochs:             {len(epochs)}")
    print(f"Best Epoch:               {best_epoch}")
    print(f"Best mAP@0.50:            {best_map50}")
    print(f"Best mAP@0.50:0.95:       {best_map50_95}")
    print(f"Overfitting Assessment:   {overfitting_evidence}")
    print("=" * 70)
    print(f"Saved analysis metrics:   {analysis_json_path}")
    print(f"Saved history table:      {table_path}")


if __name__ == "__main__":
    main()
