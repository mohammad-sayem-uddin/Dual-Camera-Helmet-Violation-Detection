#!/usr/bin/env python3
"""Standardized YOLOv8 evaluation pipeline for research benchmarks."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def format_num(val: float | int) -> float:
    return round(float(val), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO checkpoint on validation or test split.")
    parser.add_argument("--weights", required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="auto", help="Device ('mps', 'cuda:0', 'cpu', 'auto')")
    parser.add_argument("--name", required=True, help="Evaluation identifier (e.g. 'EXP-001-test')")
    parser.add_argument("--dry-run", action="store_true", help="Validate weights and dataset paths without running evaluation")
    args = parser.parse_args()

    weights_path = research_path(args.weights)
    data_path = research_path(args.data)

    if not weights_path.exists() and not args.dry_run:
        parser.error(f"Weights file not found: {weights_path}")
    if not data_path.exists():
        parser.error(f"Dataset configuration not found: {data_path}")

    print("=" * 70)
    print(f"EVALUATION PIPELINE: {args.name}")
    print("=" * 70)
    print(f"Model Checkpoint:         {weights_path}")
    print(f"Dataset Config:           {data_path}")
    print(f"Evaluation Split:         {args.split.upper()}")
    print(f"Resolution:               {args.imgsz}x{args.imgsz}")
    print(f"Confidence Threshold:     {args.conf}")
    print(f"IoU Threshold:            {args.iou}")
    print(f"Device:                   {args.device}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Evaluation configuration and paths validated successfully.")
        return

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        sys.exit(f"Error: Ultralytics/PyTorch not installed. Please run pip install -r requirements.txt.\nDetails: {e}")

    # Resolve hardware device
    req_device = args.device
    if req_device in (None, "auto", ""):
        if torch.cuda.is_available():
            resolved_device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = req_device

    eval_output_dir = research_path(Path("results/evaluation") / args.name)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    val_results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=resolved_device,
        workers=0,
        plots=True,
        project=str(eval_output_dir.parent),
        name=eval_output_dir.name,
        exist_ok=True,
    )

    box = val_results.box
    precision = format_num(box.mp)
    recall = format_num(box.mr)
    f1 = format_num(2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) > 0 else 0.0
    map50 = format_num(box.map50)
    map50_95 = format_num(box.map)

    # Extract per-class names and metrics
    class_names = val_results.names
    per_class_stats = {}
    for idx, name in class_names.items():
        p_val = format_num(box.p[idx]) if idx < len(box.p) else 0.0
        r_val = format_num(box.r[idx]) if idx < len(box.r) else 0.0
        ap50_val = format_num(box.ap50[idx]) if idx < len(box.ap50) else 0.0
        ap_val = format_num(box.ap[idx]) if idx < len(box.ap) else 0.0
        per_class_stats[name] = {
            "precision": p_val,
            "recall": r_val,
            "f1": format_num(2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0,
            "map50": ap50_val,
            "map50_95": ap_val,
        }

    report = {
        "evaluation_id": args.name,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "weights": str(weights_path),
        "dataset_yaml": str(data_path),
        "split": args.split,
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
        },
        "overall_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map50_95": map50_95,
        },
        "per_class_metrics": per_class_stats,
    }

    metrics_output = ensure_parent(research_path(Path("results/metrics") / f"{args.name}.json"))
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Copy generated confusion matrices and PR curves to results subfolders
    for plot_file in eval_output_dir.glob("*.png"):
        if "confusion_matrix" in plot_file.name:
            dest = ensure_parent(research_path(Path("results/confusion_matrices") / f"{args.name}_{plot_file.name}"))
            shutil.copy2(plot_file, dest)
        elif "curve" in plot_file.name or "PR" in plot_file.name or "F1" in plot_file.name:
            dest = ensure_parent(research_path(Path("results/plots") / f"{args.name}_{plot_file.name}"))
            shutil.copy2(plot_file, dest)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Split:               {args.split.upper()}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"mAP@0.50:            {map50:.4f}")
    print(f"mAP@0.50:0.95:       {map50_95:.4f}")
    print("-" * 70)
    print("Per-Class Metrics:")
    for cname, cmetrics in per_class_stats.items():
        print(f"  [{cname}] P: {cmetrics['precision']:.4f}, R: {cmetrics['recall']:.4f}, F1: {cmetrics['f1']:.4f}, mAP50: {cmetrics['map50']:.4f}, mAP50-95: {cmetrics['map50_95']:.4f}")
    print("=" * 70)
    print(f"Saved evaluation metrics JSON: {metrics_output}")


if __name__ == "__main__":
    main()
