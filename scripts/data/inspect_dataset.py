#!/usr/bin/env python3
"""Create a read-only quality report for a YOLO dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.yolo_dataset import inspect_yolo_dataset
from src.utils.paths import ensure_parent, research_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a YOLO dataset without modifying data.")
    parser.add_argument("--dataset", default="data/annotated/kawke_dibona_v1", help="Path to dataset root")
    parser.add_argument("--classes", type=int, default=2, help="Expected number of classes")
    parser.add_argument("--min-box-size", type=float, default=0.0, help="Minimum valid box dimension (normalized)")
    parser.add_argument("--output", default="results/metrics/dataset_audit.json", help="Path to save JSON audit report")
    args = parser.parse_args()

    dataset = research_path(args.dataset)
    if not dataset.exists():
        parser.error(f"Dataset directory does not exist: {dataset}")

    print(f"Auditing YOLO dataset at: {dataset}", flush=True)
    report = inspect_yolo_dataset(dataset, class_count=args.classes, min_box_size=args.min_box_size)

    output = ensure_parent(research_path(args.output))
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("DATASET AUDIT SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Images:               {report['total_images']}", flush=True)
    print(f"Readable Images:            {report['readable_images']}", flush=True)
    print(f"Corrupted Images:           {report['corrupted_images']}", flush=True)
    print(f"Total Label Files:          {report['total_label_files']}", flush=True)
    print(f"Annotated Images:           {report['annotated_images']}", flush=True)
    print(f"Empty (Background) Labels:  {report['empty_label_files']}", flush=True)
    print(f"Missing Annotations:        {report['images_without_annotations']}", flush=True)
    print(f"Orphan Annotations:         {report['orphan_annotations']}", flush=True)
    print(f"Annotation Issues:          {report['annotation_issues']}", flush=True)
    print(f"Total Bounding Boxes:       {report['total_objects']}", flush=True)
    print(f"Class Distribution:         {report['class_counts']}", flush=True)
    print(f"Image Resolutions:          {report['dimension_distribution']}", flush=True)
    print(f"Exact Duplicate Groups:     {report['exact_duplicate_groups']}", flush=True)

    if report.get("splits"):
        print("\nSplit Breakdown:", flush=True)
        for sname, sinfo in report["splits"].items():
            print(f"  [{sname}] Images: {sinfo['images']}, Annotated: {sinfo['annotated_images']}, Empty: {sinfo['empty_labels']}, Objects: {sinfo['objects']}, Classes: {sinfo['class_counts']}", flush=True)

    print("=" * 60, flush=True)
    print(f"Saved machine-readable audit report: {output}\n", flush=True)


if __name__ == "__main__":
    main()
