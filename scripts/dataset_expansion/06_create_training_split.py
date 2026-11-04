#!/usr/bin/env python3
"""06_create_training_split.py: Sequence-Isolated Group Partitioning Engine.

Features:
- Enforces strict video-sequence / recording isolation (0.0% sequence overlap).
- Integrates near-duplicate connected component clustering.
- Excludes uncertain and unverified images from training sets.
- Deterministic group-balanced partitioning (Seed: 42, ~70% Train, 15% Val, 15% Test).
- Mathematically verifies zero cross-split leakage: Train ∩ Val ∩ Test = ∅.
- Produces dataset_split_manifest.csv and visualization charts.
"""
import os
import sys
import random
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPANSION_ROOT = PROJECT_ROOT / "data/helmet_dataset_expansion"
MANIFESTS_DIR = EXPANSION_ROOT / "manifests"
REPORTS_DIR = EXPANSION_ROOT / "reports"
VIZ_DIR = EXPANSION_ROOT / "visualizations"


def main():
    print("=" * 80)
    print("STEP 6: SEQUENCE-ISOLATED DATASET PARTITIONING")
    print("=" * 80)

    # 1. Load inventories and manifests
    inv_file = MANIFESTS_DIR / "inventory_raw.csv"
    final_manifest_file = MANIFESTS_DIR / "final_verified_manifest.csv"
    near_dup_file = MANIFESTS_DIR / "near_duplicate_candidates.csv"

    if not final_manifest_file.exists():
        print(f"ERROR: Final manifest {final_manifest_file} not found. Run Audit Pass 3 first.")
        sys.exit(1)

    # Load inventory mapping for group prefix
    inv_map = {}
    with open(inv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv_map[row["filename"]] = row

    # Load verified manifest
    verified_rows = []
    with open(final_manifest_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verified_rows.append(row)

    total_images = len(verified_rows)
    print(f"Total images in dataset: {total_images}")

    # Separate eligible images from excluded / uncertain images
    eligible_images = []
    excluded_uncertain = []

    for r in verified_rows:
        fname = r["filename"]
        is_unc = (r.get("is_uncertain") == "True" or r.get("status") == "uncertain")
        if is_unc:
            excluded_uncertain.append(r)
        else:
            eligible_images.append(r)

    print(f"Eligible training-ready candidates: {len(eligible_images)}")
    print(f"Excluded uncertain images:          {len(excluded_uncertain)}")

    # Group clustering
    # Each image belongs to group_prefix
    # Near duplicates merge groups
    file_to_group = {}
    for r in eligible_images:
        fname = r["filename"]
        grp = inv_map.get(fname, {}).get("group_prefix", fname)
        file_to_group[fname] = grp

    # If near duplicates exist, merge their groups using Disjoint Set Union (DSU)
    parent = {}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    all_groups = set(file_to_group.values())
    for g in all_groups:
        parent[g] = g

    if near_dup_file.exists():
        with open(near_dup_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                f1 = row.get("file_1")
                f2 = row.get("file_2")
                if f1 in file_to_group and f2 in file_to_group:
                    g1 = file_to_group[f1]
                    g2 = file_to_group[f2]
                    union(g1, g2)

    # Reassign files to unified sequence clusters
    cluster_to_files = defaultdict(list)
    for r in eligible_images:
        fname = r["filename"]
        orig_grp = file_to_group[fname]
        unified_grp = find(orig_grp)
        cluster_to_files[unified_grp].append(r)

    unified_clusters = list(cluster_to_files.keys())
    print(f"Unified atomic sequence groups: {len(unified_clusters)}")

    # Deterministic Partitioning (Seed: 42)
    # Sort groups deterministically first
    unified_clusters.sort()
    rng = random.Random(42)
    rng.shuffle(unified_clusters)

    target_total = len(eligible_images)
    train_target = int(target_total * 0.70)
    val_target = int(target_total * 0.15)
    test_target = target_total - train_target - val_target

    train_groups = []
    val_groups = []
    test_groups = []

    train_count = 0
    val_count = 0
    test_count = 0

    for grp in unified_clusters:
        n_imgs = len(cluster_to_files[grp])
        # Assign to split with highest remaining deficit
        deficit_train = (train_target - train_count) / train_target if train_target > 0 else 0
        deficit_val = (val_target - val_count) / val_target if val_target > 0 else 0
        deficit_test = (test_target - test_count) / test_target if test_target > 0 else 0

        best_split = max([("train", deficit_train), ("val", deficit_val), ("test", deficit_test)], key=lambda x: x[1])[0]

        if best_split == "train":
            train_groups.append(grp)
            train_count += n_imgs
        elif best_split == "val":
            val_groups.append(grp)
            val_count += n_imgs
        else:
            test_groups.append(grp)
            test_count += n_imgs

    # Mathematical Disjointness Proof
    set_train = set(train_groups)
    set_val = set(val_groups)
    set_test = set(test_groups)

    assert len(set_train.intersection(set_val)) == 0, "FATAL: Train and Val sequence overlap detected!"
    assert len(set_train.intersection(set_test)) == 0, "FATAL: Train and Test sequence overlap detected!"
    assert len(set_val.intersection(set_test)) == 0, "FATAL: Val and Test sequence overlap detected!"

    print("\nMATHEMATICAL DISJOINTNESS VERIFICATION:")
    print("Train Groups ∩ Val Groups  = ∅ (0.0% overlap)")
    print("Train Groups ∩ Test Groups = ∅ (0.0% overlap)")
    print("Val Groups ∩ Test Groups   = ∅ (0.0% overlap)")
    print("Zero sequence leakage confirmed.")

    # Build split manifest
    split_records = []
    split_obj_counts = defaultdict(lambda: {"riders": 0, "helmets": 0, "no_helmets": 0})

    for grp in train_groups:
        for r in cluster_to_files[grp]:
            split_records.append({**r, "split": "train", "sequence_group": grp})
            split_obj_counts["train"]["riders"] += int(r["rider_count"])
            split_obj_counts["train"]["helmets"] += int(r["helmet_count"])
            split_obj_counts["train"]["no_helmets"] += int(r["no_helmet_count"])

    for grp in val_groups:
        for r in cluster_to_files[grp]:
            split_records.append({**r, "split": "val", "sequence_group": grp})
            split_obj_counts["val"]["riders"] += int(r["rider_count"])
            split_obj_counts["val"]["helmets"] += int(r["helmet_count"])
            split_obj_counts["val"]["no_helmets"] += int(r["no_helmet_count"])

    for grp in test_groups:
        for r in cluster_to_files[grp]:
            split_records.append({**r, "split": "test", "sequence_group": grp})
            split_obj_counts["test"]["riders"] += int(r["rider_count"])
            split_obj_counts["test"]["helmets"] += int(r["helmet_count"])
            split_obj_counts["test"]["no_helmets"] += int(r["no_helmet_count"])

    # Excluded images manifest
    for r in excluded_uncertain:
        split_records.append({**r, "split": "excluded", "sequence_group": "uncertain_quarantine"})

    split_manifest_csv = MANIFESTS_DIR / "dataset_split_manifest.csv"
    with open(split_manifest_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "status", "rider_count", "helmet_count", "no_helmet_count", "is_uncertain", "split", "sequence_group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_records)

    print(f"\nSaved sequence-isolated split manifest to {split_manifest_csv}")

    print("\n" + "=" * 80)
    print("SPLIT DISTRIBUTION PROFILE")
    print("=" * 80)
    print(f"Train:    {len(train_groups)} groups | {train_count} images ({train_count/target_total*100:.1f}%) | "
          f"{split_obj_counts['train']['riders']} riders (h: {split_obj_counts['train']['helmets']}, nh: {split_obj_counts['train']['no_helmets']})")
    print(f"Val:      {len(val_groups)} groups | {val_count} images ({val_count/target_total*100:.1f}%) | "
          f"{split_obj_counts['val']['riders']} riders (h: {split_obj_counts['val']['helmets']}, nh: {split_obj_counts['val']['no_helmets']})")
    print(f"Test:     {len(test_groups)} groups | {test_count} images ({test_count/target_total*100:.1f}%) | "
          f"{split_obj_counts['test']['riders']} riders (h: {split_obj_counts['test']['helmets']}, nh: {split_obj_counts['test']['no_helmets']})")
    print(f"Excluded: {len(excluded_uncertain)} uncertain/quarantined images")

    # Generate Visualization Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sequence-Isolated Dataset Split Distribution (Seed 42)", fontsize=14, fontweight="bold")

    # Image Split Bar Chart
    splits = ["Train", "Validation", "Test"]
    img_counts = [train_count, val_count, test_count]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    axes[0].bar(splits, img_counts, color=colors, edgecolor="black", alpha=0.8)
    axes[0].set_title("Image Count by Split")
    axes[0].set_ylabel("Images")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for i, v in enumerate(img_counts):
        axes[0].text(i, v + 15, f"{v}\n({v/target_total*100:.1f}%)", ha="center", fontweight="bold")

    # Object Class Breakdown Bar Chart
    x = range(3)
    w = 0.35
    helmets_list = [split_obj_counts[s.lower()]["helmets"] for s in ["train", "val", "test"]]
    no_helmets_list = [split_obj_counts[s.lower()]["no_helmets"] for s in ["train", "val", "test"]]
    axes[1].bar([p - w/2 for p in x], helmets_list, width=w, label="Helmet (Class 1)", color="#00c800", edgecolor="black", alpha=0.8)
    axes[1].bar([p + w/2 for p in x], no_helmets_list, width=w, label="No Helmet (Class 0)", color="#e60000", edgecolor="black", alpha=0.8)
    axes[1].set_title("Object Instance Distribution (Helmet vs. No Helmet)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("Object Instances")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    chart_path = VIZ_DIR / "dataset_split_distribution.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved split distribution chart to {chart_path}")
    print("\nSEQUENCE-ISOLATED SPLIT GENERATION COMPLETE.")


if __name__ == "__main__":
    main()
