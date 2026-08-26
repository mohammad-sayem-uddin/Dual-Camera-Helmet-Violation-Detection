#!/usr/bin/env python3
"""Materialize a group-exclusive split into a valid YOLOv8 dataset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path
from src.utils.yaml_loader import dump_yaml


def materialize_dataset(manifest_csv: Path, output_dir: Path, use_symlinks: bool = True) -> dict[str, int]:
    """Materialize split from manifest CSV into train/val/test directories."""
    root = research_path(".")
    output_dir = output_dir.resolve()
    
    # Clean output dir if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in ("train", "val", "test"):
                continue

            rel_img_path = row["image_path"]
            src_img = (root / rel_img_path).resolve()
            
            # Find matching label
            src_lbl = src_img.parent.parent / "labels" / (src_img.stem + ".txt")
            if not src_lbl.exists():
                src_lbl = src_img.with_suffix(".txt")

            if not src_img.exists():
                print(f"[WARN] Image not found: {src_img}")
                continue

            dest_img_dir = output_dir / split / "images"
            dest_lbl_dir = output_dir / split / "labels"
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            dest_img = dest_img_dir / src_img.name
            dest_lbl = dest_lbl_dir / (src_img.stem + ".txt")

            if use_symlinks:
                try:
                    if dest_img.exists():
                        dest_img.unlink()
                    dest_img.symlink_to(src_img)
                    if src_lbl.exists():
                        if dest_lbl.exists():
                            dest_lbl.unlink()
                        dest_lbl.symlink_to(src_lbl)
                    else:
                        dest_lbl.write_text("", encoding="utf-8")
                except OSError:
                    # Fallback to copy if symlinks not supported
                    shutil.copy2(src_img, dest_img)
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, dest_lbl)
                    else:
                        dest_lbl.write_text("", encoding="utf-8")
            else:
                shutil.copy2(src_img, dest_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dest_lbl)
                else:
                    dest_lbl.write_text("", encoding="utf-8")

            counts[split] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a group-exclusive split CSV into a ready-to-train YOLOv8 dataset.")
    parser.add_argument("--manifest", default="data/splits/v1_group_split.csv", help="Path to group split manifest CSV")
    parser.add_argument("--output-dir", default="data/processed/v1_group_split", help="Target dataset directory")
    parser.add_argument("--dataset-yaml", default="configs/dataset_group_split.yaml", help="Path to generate dataset YAML config")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    args = parser.parse_args()

    manifest_path = research_path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"Manifest CSV not found: {manifest_path}")

    out_dir = research_path(args.output_dir)
    yaml_path = research_path(args.dataset_yaml)

    print("=" * 70)
    print("MATERIALIZING GROUP-EXCLUSIVE DATASET")
    print("=" * 70)
    print(f"Source Manifest:          {manifest_path}")
    print(f"Target Directory:         {out_dir}")
    print(f"Dataset YAML Output:      {yaml_path}")
    print(f"Method:                   {'File Copy' if args.copy else 'Symbolic Links (Fast & Zero-Disk Overhead)'}")
    print("=" * 70)

    counts = materialize_dataset(manifest_path, out_dir, use_symlinks=not args.copy)

    # Generate dataset YAML
    rel_out_dir = str(out_dir.relative_to(research_path(".")))
    dataset_config = {
        "path": rel_out_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": {
            0: "no_helmet",
            1: "helmet"
        }
    }

    yaml_path = ensure_parent(yaml_path)
    yaml_path.write_text(dump_yaml(dataset_config), encoding="utf-8")

    print("\n" + "=" * 70)
    print("DATASET MATERIALIZATION COMPLETE")
    print("=" * 70)
    print(f"Train Images:             {counts['train']}")
    print(f"Validation Images:        {counts['val']}")
    print(f"Test Images:              {counts['test']}")
    print(f"Total Images:             {sum(counts.values())}")
    print(f"Dataset YAML:             {yaml_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
