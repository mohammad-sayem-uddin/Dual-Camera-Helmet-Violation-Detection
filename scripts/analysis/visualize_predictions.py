#!/usr/bin/env python3
"""Generate and save model visual predictions for qualitative research review."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import research_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual detection predictions for qualitative review.")
    parser.add_argument("--weights", required=True, help="Path to model weights checkpoint (.pt)")
    parser.add_argument("--source", default="data/annotated/kawke_dibona_v1/test/images", help="Image file or directory")
    parser.add_argument("--name", required=True, help="Run name (e.g. 'EXP-001-test-preds')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", default="auto", help="Inference device ('mps', 'cuda:0', 'cpu', 'auto')")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to predict and save")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without running inference")
    args = parser.parse_args()

    weights_path = research_path(args.weights)
    source_path = research_path(args.source)

    if not weights_path.exists() and not args.dry_run:
        parser.error(f"Weights file not found: {weights_path}")
    if not source_path.exists():
        parser.error(f"Source path not found: {source_path}")

    print("=" * 70)
    print(f"PREDICTION VISUALIZER: {args.name}")
    print("=" * 70)
    print(f"Model Checkpoint:         {weights_path}")
    print(f"Source Images:            {source_path}")
    print(f"Confidence Threshold:     {args.conf}")
    print(f"Resolution:               {args.imgsz}x{args.imgsz}")
    print(f"Device:                   {args.device}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Prediction configuration and paths validated successfully.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"Error: Ultralytics not installed. Please run pip install -r requirements.txt.\nDetails: {e}")

    output_dir = research_path(Path("results/predictions"))
    
    # If source is a directory and max-images is specified, select subset
    source_to_predict: str | list[str] = str(source_path)
    if source_path.is_dir() and args.max_images is not None:
        images = sorted(p for p in source_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        source_to_predict = [str(p) for p in images[:args.max_images]]
        print(f"Selected subset of {len(source_to_predict)} images for qualitative review.")

    model = YOLO(str(weights_path))
    model.predict(
        source=source_to_predict,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
    )

    print(f"\nWrote prediction visualizations to: {output_dir / args.name}\n")


if __name__ == "__main__":
    main()
