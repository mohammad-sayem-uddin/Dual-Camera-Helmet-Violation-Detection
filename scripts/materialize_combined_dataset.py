#!/usr/bin/env python3
"""materialize_combined_dataset.py

Creates the unified zero-leakage combined dataset for EXP-COMBINED:
- Merges old_dataset_v1_fixed (EXP-002-FIXED) + helmet_dataset_expansion_v2
- Uses hardlinks to avoid duplicating disk space.
- Generates data.yaml and master combined manifest.
"""
import os
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = PROJECT_ROOT / "data/processed/old_dataset_v1_fixed"
NEW_DIR = PROJECT_ROOT / "data/helmet_dataset_expansion/training_ready_v2"
COMBINED_DIR = PROJECT_ROOT / "data/processed/combined_dataset_v2"

COMBINED_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []
stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'boxes': 0, 'cls0': 0, 'cls1': 0})

splits = ['train', 'val', 'test']

for sp in splits:
    out_img_dir = COMBINED_DIR / f"images/{sp}"
    out_lbl_dir = COMBINED_DIR / f"labels/{sp}"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Add Old Dataset split items
    old_img_dir = OLD_DIR / f"{sp}/images"
    old_lbl_dir = OLD_DIR / f"{sp}/labels"
    for img_path in sorted(old_img_dir.glob("*.jpg")):
        fname = f"old_{img_path.name}"
        stem = f"old_{img_path.stem}"
        lbl_path = old_lbl_dir / f"{img_path.stem}.txt"

        target_img = out_img_dir / fname
        target_lbl = out_lbl_dir / f"{stem}.txt"

        if not target_img.exists():
            os.link(img_path, target_img)

        # Process label
        boxes_in_lbl = 0
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            with open(target_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            for l in lines:
                parts = l.split()
                cid = int(parts[0])
                if cid == 0: stats[sp]['cls0'] += 1
                elif cid == 1: stats[sp]['cls1'] += 1
                boxes_in_lbl += 1
            stats[sp]['labels'] += 1
        stats[sp]['images'] += 1
        stats[sp]['boxes'] += boxes_in_lbl
        manifest_rows.append({
            'source_dataset': 'old_dataset_v1_fixed',
            'split': sp,
            'original_filename': img_path.name,
            'combined_filename': fname,
            'num_boxes': boxes_in_lbl
        })

    # 2. Add New Dataset split items
    new_img_dir = NEW_DIR / f"images/{sp}"
    new_lbl_dir = NEW_DIR / f"labels/{sp}"
    for img_path in sorted(new_img_dir.glob("*.jpg")):
        fname = f"new_{img_path.name}"
        stem = f"new_{img_path.stem}"
        lbl_path = new_lbl_dir / f"{img_path.stem}.txt"

        target_img = out_img_dir / fname
        target_lbl = out_lbl_dir / f"{stem}.txt"

        if not target_img.exists():
            os.link(img_path, target_img)

        # Process label
        boxes_in_lbl = 0
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            with open(target_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            for l in lines:
                parts = l.split()
                cid = int(parts[0])
                if cid == 0: stats[sp]['cls0'] += 1
                elif cid == 1: stats[sp]['cls1'] += 1
                boxes_in_lbl += 1
            stats[sp]['labels'] += 1
        stats[sp]['images'] += 1
        stats[sp]['boxes'] += boxes_in_lbl
        manifest_rows.append({
            'source_dataset': 'helmet_dataset_expansion_v2',
            'split': sp,
            'original_filename': img_path.name,
            'combined_filename': fname,
            'num_boxes': boxes_in_lbl
        })

# Write data.yaml
data_yaml_content = f"""# YOLOv8 Dataset Configuration: EXP-COMBINED (Old Sequence-Isolated + New Expansion v2)
path: {COMBINED_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: no_helmet
  1: helmet

nc: 2
"""

with open(COMBINED_DIR / "data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml_content)

# Write manifest CSV
manifest_csv_path = COMBINED_DIR / "combined_manifest.csv"
with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['source_dataset', 'split', 'original_filename', 'combined_filename', 'num_boxes'])
    writer.writeheader()
    writer.writerows(manifest_rows)

print("=" * 80)
print("EXP-COMBINED DATASET MATERIALIZATION COMPLETE")
print("=" * 80)
for sp in splits:
    s = stats[sp]
    print(f"Split {sp.upper():5s}: {s['images']} images | {s['labels']} labels | {s['boxes']} boxes (Cls 0: {s['cls0']}, Cls 1: {s['cls1']})")
print(f"Total Combined Images: {sum(s['images'] for s in stats.values())}")
print(f"Total Combined Boxes:  {sum(s['boxes'] for s in stats.values())}")
print(f"Config saved to: {COMBINED_DIR / 'data.yaml'}")
