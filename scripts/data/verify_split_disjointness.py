#!/usr/bin/env python3
"""Mathematically verify that train, val, and test splits have strictly 0% sequence overlap."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import research_path


def verify_disjointness(manifest_path: Path) -> bool:
    manifest_path = research_path(manifest_path)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        return False

    split_groups = defaultdict(set)
    group_images = defaultdict(lambda: defaultdict(list))

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"image_path", "group_id", "split"}.issubset(reader.fieldnames or set()):
            print(f"Error: Manifest requires image_path, group_id, split columns", file=sys.stderr)
            return False

        for row in reader:
            split = row["split"].strip().lower()
            group_id = row["group_id"].strip()
            img_path = row["image_path"].strip()

            split_groups[split].add(group_id)
            group_images[group_id][split].append(img_path)

    splits = sorted(split_groups.keys())
    print("=" * 70)
    print(f"SPLIT DISJOINTNESS AUDIT: {manifest_path.name}")
    print("=" * 70)
    for s in splits:
        total_imgs = sum(len(group_images[g][s]) for g in split_groups[s])
        print(f"Split '{s}': {len(split_groups[s])} unique groups, {total_imgs} images")
    print("-" * 70)

    # Check pairwise intersections
    violations = 0
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_groups[s1] & split_groups[s2]
            if overlap:
                violations += len(overlap)
                print(f"❌ LEAKAGE VIOLATION: {len(overlap)} groups overlap between '{s1}' and '{s2}':")
                for g in sorted(overlap)[:10]:
                    print(f"   - Group '{g}': {len(group_images[g][s1])} in {s1}, {len(group_images[g][s2])} in {s2}")
                if len(overlap) > 10:
                    print(f"   ... and {len(overlap) - 10} more.")
            else:
                print(f"✅ {s1.upper()} ∩ {s2.upper()} = ∅ (Strictly Zero Overlap)")

    print("=" * 70)
    if violations == 0:
        print("MATHEMATICAL PROOF VERIFIED: Split is strictly sequence-exclusive (0.0% leakage).")
        print("=" * 70)
        return True
    else:
        print(f"FAIL: {violations} overlapping group instances detected.")
        print("=" * 70)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify split disjointness across groups.")
    parser.add_argument("--manifest", required=True, help="Path to split manifest CSV")
    args = parser.parse_args()

    success = verify_disjointness(Path(args.manifest))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
