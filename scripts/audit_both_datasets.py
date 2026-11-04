#!/usr/bin/env python3
"""audit_both_datasets.py

Performs comprehensive dual-dataset audit:
1. Reconstructs missing old dataset label .txt files from verified labels.cache to data/processed/old_dataset_v1_fixed/
2. Audits Old Dataset (5,547 images) and New Dataset (1,841 images).
3. Cross-dataset Exact Duplicate Analysis (SHA-256).
4. Cross-dataset Perceptual Duplicate Analysis (dHash / difference hash).
5. Cross-dataset Sequence Overlap Analysis.
6. Exports machine-readable audit report: results/dataset_audit/
"""
import os
import sys
import hashlib
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/Users/sayemuddin/Desktop/research_helmet")
OLD_SRC_DIR = PROJECT_ROOT / "data/annotated/kawke_dibona_v1"
OLD_SPLIT_CSV = PROJECT_ROOT / "data/splits/v1_group_split_fixed.csv"
OLD_FIXED_DIR = PROJECT_ROOT / "data/processed/old_dataset_v1_fixed"

NEW_V2_DIR = PROJECT_ROOT / "data/helmet_dataset_expansion/training_ready_v2"
NEW_RAW_DIR = PROJECT_ROOT / "data/helmet_dataset_expansion/original"

AUDIT_OUT_DIR = PROJECT_ROOT / "results/dataset_audit"
AUDIT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_dhash(image, hash_size=8):
    """Compute difference hash (dHash) for perceptual duplication detection."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count("1")


def materialize_old_dataset_labels():
    """Ensure old dataset has a clean standard directory structure with images and labels."""
    print("--- Materializing Old Dataset Standard Directory Structure ---")
    OLD_FIXED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load cache dictionaries
    cache_data = {}
    for sp in ['train', 'valid', 'test']:
        p = OLD_SRC_DIR / sp / 'labels.cache'
        c = np.load(p, allow_pickle=True).item()
        for item in c['labels']:
            fname = Path(item['im_file']).name
            cache_data[fname] = item

    print(f"Loaded {len(cache_data)} cached label entries for old dataset.")

    # Read split CSV
    with open(OLD_SPLIT_CSV, newline='', encoding='utf-8') as f:
        split_rows = list(csv.DictReader(f))

    print(f"Read {len(split_rows)} rows from {OLD_SPLIT_CSV.name}.")

    # Materialize train, val, test folders with symlinks/copies of images and text labels
    stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'boxes': 0, 'cls0': 0, 'cls1': 0})
    for row in split_rows:
        img_rel = row['image_path']
        sp = row['split'] # train, val, test
        src_img = PROJECT_ROOT / img_rel
        fname = src_img.name
        stem = src_img.stem

        out_img_dir = OLD_FIXED_DIR / sp / "images"
        out_lbl_dir = OLD_FIXED_DIR / sp / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        target_img = out_img_dir / fname
        if not target_img.exists():
            # Use relative symlink or copy
            os.link(src_img, target_img)

        target_lbl = out_lbl_dir / f"{stem}.txt"
        if fname in cache_data:
            item = cache_data[fname]
            cls_arr = item['cls']
            bbox_arr = item['bboxes']
            lines = []
            for c_id, bbox in zip(cls_arr, bbox_arr):
                cid = int(c_id[0]) if hasattr(c_id, '__len__') else int(c_id)
                xc, yc, w, h = bbox
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                if cid == 0:
                    stats[sp]['cls0'] += 1
                elif cid == 1:
                    stats[sp]['cls1'] += 1
                stats[sp]['boxes'] += 1

            with open(target_lbl, 'w', encoding='utf-8') as lf:
                lf.write("\n".join(lines) + ("\n" if lines else ""))
            stats[sp]['labels'] += 1
        stats[sp]['images'] += 1

    # Write data.yaml for old dataset
    data_yaml_content = f"""# YOLOv8 Dataset Configuration: Old Dataset (kawke_dibona_v1 - Sequence Isolated EXP-002-FIXED)
path: {OLD_FIXED_DIR}
train: train/images
val: val/images
test: test/images

names:
  0: no_helmet
  1: helmet

nc: 2
"""
    with open(OLD_FIXED_DIR / "data.yaml", "w", encoding="utf-8") as yf:
        yf.write(data_yaml_content)

    print("Old dataset materialization complete:")
    for sp, s in stats.items():
        print(f"  {sp.upper()}: {s['images']} images, {s['labels']} labels, {s['boxes']} boxes (Cls 0: {s['cls0']}, Cls 1: {s['cls1']})")
    return stats


def audit_datasets():
    print("\n--- Auditing Datasets & Cross-Dataset Leakage ---")
    
    # 1. Collect all images and hashes
    old_images = {}
    old_hashes = {}
    old_dhashes = {}

    with open(OLD_SPLIT_CSV, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = PROJECT_ROOT / r['image_path']
            fn = p.name
            old_images[fn] = {'path': p, 'split': r['split'], 'group': r['group_id']}

    print(f"Reading SHA-256 and dHash for {len(old_images)} Old Dataset images...")
    for fn, info in old_images.items():
        with open(info['path'], 'rb') as f:
            data = f.read()
            h = hashlib.sha256(data).hexdigest()
            old_hashes[fn] = h
        with Image.open(info['path']) as im:
            old_dhashes[fn] = compute_dhash(im)

    # 2. Collect all new images and hashes
    new_images = {}
    new_hashes = {}
    new_dhashes = {}

    # Read new dataset manifest
    new_manifest_path = PROJECT_ROOT / "data/helmet_dataset_expansion/training_ready_v2/dataset_manifest.csv"
    with open(new_manifest_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            fn = r['filename']
            sp = r['split']
            p = NEW_V2_DIR / f"images/{sp}/{fn}"
            new_images[fn] = {'path': p, 'split': sp, 'group': r['sequence_group']}

    print(f"Reading SHA-256 and dHash for {len(new_images)} New Dataset images...")
    for fn, info in new_images.items():
        with open(info['path'], 'rb') as f:
            data = f.read()
            h = hashlib.sha256(data).hexdigest()
            new_hashes[fn] = h
        with Image.open(info['path']) as im:
            new_dhashes[fn] = compute_dhash(im)

    # 3. Exact Duplicate Analysis (SHA-256)
    old_hash_to_fn = defaultdict(list)
    for fn, h in old_hashes.items():
        old_hash_to_fn[h].append(fn)

    new_hash_to_fn = defaultdict(list)
    for fn, h in new_hashes.items():
        new_hash_to_fn[h].append(fn)

    # Internal duplicates
    old_internal_dups = {h: fns for h, fns in old_hash_to_fn.items() if len(fns) > 1}
    new_internal_dups = {h: fns for h, fns in new_hash_to_fn.items() if len(fns) > 1}
    
    # Cross-dataset exact duplicates
    cross_exact_dups = set(old_hash_to_fn.keys()) & set(new_hash_to_fn.keys())

    print(f"\nExact Duplicate Findings:")
    print(f"  Old Dataset Internal Exact Duplicates: {len(old_internal_dups)}")
    print(f"  New Dataset Internal Exact Duplicates: {len(new_internal_dups)}")
    print(f"  Cross-Dataset Exact Duplicates: {len(cross_exact_dups)}")

    # 4. Perceptual Duplicate Analysis (Hamming distance <= 3)
    # Check cross-dataset near-duplicates
    cross_perceptual_matches = []
    print("\nEvaluating Cross-Dataset Perceptual Similarity (dHash Hamming distance <= 2)...")
    for n_fn, n_dh in new_dhashes.items():
        for o_fn, o_dh in old_dhashes.items():
            dist = hamming_distance(n_dh, o_dh)
            if dist <= 2:
                cross_perceptual_matches.append({
                    'new_image': n_fn,
                    'old_image': o_fn,
                    'hamming_dist': dist,
                    'new_split': new_images[n_fn]['split'],
                    'old_split': old_images[o_fn]['split']
                })

    print(f"  Cross-Dataset Perceptual Near-Duplicates found: {len(cross_perceptual_matches)}")

    # 5. Sequence Name Overlap Analysis
    old_groups = set(r['group'] for r in old_images.values())
    new_groups = set(r['group'] for r in new_images.values())
    shared_groups = old_groups & new_groups
    print(f"\nSequence Group Analysis:")
    print(f"  Old Dataset Unique Sequence Groups: {len(old_groups)}")
    print(f"  New Dataset Unique Sequence Groups: {len(new_groups)}")
    print(f"  Cross-Dataset Shared Sequence Groups: {len(shared_groups)}")

    # 6. Save comprehensive JSON report
    report = {
        "timestamp": "2026-11-04T09:15:00Z",
        "old_dataset": {
            "name": "kawke_dibona_v1 (EXP-002-FIXED)",
            "total_images": len(old_images),
            "splits": dict(Counter(info['split'] for info in old_images.values())),
            "unique_groups": len(old_groups),
            "internal_exact_duplicates": len(old_internal_dups),
        },
        "new_dataset": {
            "name": "helmet_dataset_expansion_v2",
            "total_images": len(new_images),
            "splits": dict(Counter(info['split'] for info in new_images.values())),
            "unique_groups": len(new_groups),
            "internal_exact_duplicates": len(new_internal_dups),
        },
        "cross_dataset_leakage": {
            "exact_sha256_duplicates": len(cross_exact_dups),
            "perceptual_near_duplicates_dist_le_2": len(cross_perceptual_matches),
            "shared_sequence_groups": len(shared_groups),
            "details_perceptual": cross_perceptual_matches[:20]
        }
    }

    report_path = AUDIT_OUT_DIR / "cross_dataset_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved comprehensive cross-dataset audit report to: {report_path}")
    return report


if __name__ == "__main__":
    materialize_old_dataset_labels()
    audit_datasets()
