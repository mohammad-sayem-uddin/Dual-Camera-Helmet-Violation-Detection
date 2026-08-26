#!/usr/bin/env python3
"""Create size-matched random-frame split (EXP-003) to isolate training set size confound."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def create_matched_random_split(
    source_groups_path: Path,
    target_train_count: int,
    output_manifest: Path,
    seed: int = 42,
) -> dict[str, int]:
    source_groups_path = research_path(source_groups_path)
    output_manifest = ensure_parent(research_path(output_manifest))

    train_rows = []
    val_rows = []
    test_rows = []

    with source_groups_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip().lower()
            if split == "train":
                train_rows.append(row)
            elif split in ("valid", "val"):
                val_rows.append(row)
            elif split == "test":
                test_rows.append(row)

    print("=" * 70)
    print("CREATING SIZE-MATCHED RANDOM SPLIT (EXP-003)")
    print("=" * 70)
    print(f"Original Random Train Count:  {len(train_rows)}")
    print(f"Target Size-Matched Train:    {target_train_count}")
    print(f"Validation Count (Unchanged): {len(val_rows)}")
    print(f"Test Count (Unchanged):       {len(test_rows)}")
    print(f"Random Seed:                  {seed}")
    print("=" * 70)

    if target_train_count > len(train_rows):
        raise ValueError(f"Target count {target_train_count} exceeds available train images {len(train_rows)}")

    # Deterministic subsample
    rng = random.Random(seed)
    sampled_train = rng.sample(train_rows, target_train_count)

    out_rows = []
    for r in sampled_train:
        out_rows.append({"image_path": r["image_path"], "group_id": r["group_id"], "split": "train"})
    for r in val_rows:
        out_rows.append({"image_path": r["image_path"], "group_id": r["group_id"], "split": "val"})
    for r in test_rows:
        out_rows.append({"image_path": r["image_path"], "group_id": r["group_id"], "split": "test"})

    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "group_id", "split"])
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {"train": len(sampled_train), "val": len(val_rows), "test": len(test_rows)}
    print(f"Saved size-matched manifest to: {output_manifest}")
    print(f"Final Split Counts: {counts}")
    print("=" * 70)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create size-matched random split manifest.")
    parser.add_argument("--source", default="data/metadata/source_groups.csv", help="Path to source_groups.csv")
    parser.add_argument("--target-train", type=int, default=3866, help="Target number of train images")
    parser.add_argument("--output", default="data/splits/v1_random_matched.csv", help="Output manifest path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    create_matched_random_split(Path(args.source), args.target_train, Path(args.output), args.seed)


if __name__ == "__main__":
    main()
