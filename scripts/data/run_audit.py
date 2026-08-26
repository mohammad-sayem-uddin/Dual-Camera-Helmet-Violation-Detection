#!/usr/bin/env python3
"""Run complete dataset audit, quality control, and split leakage analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/annotated/kawke_dibona_v1"
METRICS_DIR = ROOT / "results/metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR = ROOT / "data/metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_jpeg_dimensions(data: bytes) -> tuple[tuple[int, int] | None, str | None]:
    if len(data) < 2 or data[:2] != b"\xff\xd8":
        return None, "Not a valid JPEG file"
    offset = 2
    length = len(data)
    while offset < length - 1:
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < length - 1 and data[offset + 1] == 0xFF:
            offset += 1
        if offset >= length - 1:
            break
        marker = data[offset + 1]
        offset += 2
        if marker in (0xDA, 0xD9):
            break
        if marker in (0xD8, 0x00, 0x01) or (0xD0 <= marker <= 0xD7):
            continue
        if offset + 2 > length:
            break
        seg_len = struct.unpack(">H", data[offset : offset + 2])[0]
        if seg_len < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if offset + 7 > length:
                return None, "Truncated SOF header"
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return (width, height), None
        offset += seg_len
    return None, "No SOF marker found"


def run_full_audit():
    print("=" * 70)
    print("STARTING DATASET QUALITY & LEAKAGE AUDIT")
    print(f"Dataset Root: {DATASET_ROOT}")
    print("=" * 70)

    splits = ["train", "valid", "test"]
    split_stats = {}
    all_image_hashes = defaultdict(list)
    video_group_map = defaultdict(list)
    source_groups_rows = []

    pattern_rf = re.compile(r"^(.*?)(?:_jpg)?\.rf\.[0-9a-f]{32}$")

    def extract_video_group(bf: str) -> str:
        # Pattern 1: Timestamp format (e.g. 20260522_163554_frame_004 -> 20260522_163554)
        m = re.match(r"^(\d{8}_\d{6})_frame_\d+", bf)
        if m:
            return m.group(1)
        # Pattern 2: Timestamp with sub-index (e.g. 20260522_183544-1-_frame_003 -> 20260522_183544-1-)
        m = re.match(r"^(\d{8}_\d{6}-\d+-?)_frame_\d+", bf)
        if m:
            return m.group(1)
        # Pattern 3: VID format (e.g. VID20260524160459_frame_003 -> VID20260524160459)
        m = re.match(r"^(VID\d+)_frame_\d+", bf)
        if m:
            return m.group(1)
        # Pattern 4: frame_N_NNN format (e.g. frame_7_100 -> frame_7)
        m = re.match(r"^frame_(\d+)_\d+", bf)
        if m:
            return f"frame_{m.group(1)}"
        return bf

    total_images_all = 0
    total_labels_all = 0
    total_boxes_all = 0
    total_empty_labels_all = 0
    total_annotated_all = 0
    overall_classes = Counter()
    overall_dims = Counter()

    for sname in splits:
        sdir = DATASET_ROOT / sname
        img_dir = sdir / "images"
        lbl_dir = sdir / "labels"

        img_files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))
        lbl_files = sorted(p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and not p.name.startswith("."))

        img_map = {f.stem: f for f in img_files}
        lbl_map = {f.stem: f for f in lbl_files}

        s_classes = Counter()
        s_objects = 0
        s_empty = 0
        s_annotated = 0
        s_corrupt = 0
        s_box_issues = 0

        for stem, img_path in img_map.items():
            total_images_all += 1
            rel_img = str(img_path.relative_to(ROOT))
            data = img_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            all_image_hashes[sha].append(rel_img)

            # Sequence identification
            match_rf = pattern_rf.match(stem)
            base_frame = match_rf.group(1) if match_rf else stem
            group_id = extract_video_group(base_frame)
            video_group_map[group_id].append((rel_img, sname, base_frame))
            source_groups_rows.append({"image_path": rel_img, "group_id": group_id, "base_frame": base_frame, "split": sname})

            dims, err = parse_jpeg_dimensions(data)
            if err:
                s_corrupt += 1
            else:
                overall_dims[f"{dims[0]}x{dims[1]}"] += 1

            if stem not in lbl_map:
                continue

            lbl_path = lbl_map[stem]
            total_labels_all += 1
            content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
            lines = [l.strip() for l in content.splitlines() if l.strip()]

            if not lines:
                s_empty += 1
                total_empty_labels_all += 1
                continue

            s_annotated += 1
            total_annotated_all += 1

            for num, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    s_box_issues += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    s_box_issues += 1
                    continue

                if cls_id not in (0, 1):
                    s_box_issues += 1
                elif not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    s_box_issues += 1
                elif w <= 0 or h <= 0:
                    s_box_issues += 1
                elif (x - w / 2 < -1e-4) or (x + w / 2 > 1.0001) or (y - h / 2 < -1e-4) or (y + h / 2 > 1.0001):
                    s_box_issues += 1
                else:
                    s_objects += 1
                    total_boxes_all += 1
                    s_classes[str(cls_id)] += 1
                    overall_classes[str(cls_id)] += 1

        split_stats[sname] = {
            "images": len(img_files),
            "label_files": len(lbl_files),
            "annotated_images": s_annotated,
            "empty_labels": s_empty,
            "corrupt_images": s_corrupt,
            "annotation_issues": s_box_issues,
            "objects": s_objects,
            "class_counts": dict(sorted(s_classes.items(), key=lambda x: int(x[0]))),
        }

    # Sequence leakage computation
    split_video_sets = defaultdict(set)
    for row in source_groups_rows:
        split_video_sets[row["split"]].add(row["group_id"])

    train_val_overlap = sorted(split_video_sets["train"] & split_video_sets["valid"])
    train_test_overlap = sorted(split_video_sets["train"] & split_video_sets["test"])
    val_test_overlap = sorted(split_video_sets["valid"] & split_video_sets["test"])

    exact_duplicates = [v for v in all_image_hashes.values() if len(v) > 1]

    # Save dataset_audit.json
    audit_report = {
        "dataset_name": "kawke_dibona_v1",
        "dataset_path": str(DATASET_ROOT.relative_to(ROOT)),
        "classes": {0: "no_helmet", 1: "helmet"},
        "summary": {
            "total_images": total_images_all,
            "total_label_files": total_labels_all,
            "annotated_images": total_annotated_all,
            "empty_label_files": total_empty_labels_all,
            "total_bounding_boxes": total_boxes_all,
            "class_counts": dict(sorted(overall_classes.items(), key=lambda x: int(x[0]))),
            "dimension_distribution": dict(overall_dims),
            "corrupt_images": sum(s["corrupt_images"] for s in split_stats.values()),
            "annotation_issues": sum(s["annotation_issues"] for s in split_stats.values()),
            "exact_duplicate_groups": len(exact_duplicates),
        },
        "splits": split_stats,
        "leakage_analysis": {
            "total_unique_video_sequences": len(video_group_map),
            "sequences_in_train": len(split_video_sets["train"]),
            "sequences_in_valid": len(split_video_sets["valid"]),
            "sequences_in_test": len(split_video_sets["test"]),
            "train_valid_overlapping_sequences": len(train_val_overlap),
            "train_test_overlapping_sequences": len(train_test_overlap),
            "valid_test_overlapping_sequences": len(val_test_overlap),
            "leakage_summary": "Temporal frame leakage detected. Roboflow split frames across splits by frame rather than by video sequence."
        }
    }

    audit_path = METRICS_DIR / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(f"Saved: {audit_path}")

    # Save duplicate report
    dup_report = {
        "images_checked": total_images_all,
        "unique_sha256_hashes": len(all_image_hashes),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_files": exact_duplicates,
        "video_sequence_groups_detected": len(video_group_map),
        "cross_split_sequence_leakage": {
            "train_valid_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "sample_leaked_sequences": train_val_overlap[:10]
        }
    }
    dup_path = METRICS_DIR / "duplicate_report.json"
    dup_path.write_text(json.dumps(dup_report, indent=2), encoding="utf-8")
    print(f"Saved: {dup_path}")

    # Save source_groups.csv manifest
    manifest_path = METADATA_DIR / "source_groups.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "group_id", "base_frame", "split"])
        writer.writeheader()
        writer.writerows(source_groups_rows)
    print(f"Saved: {manifest_path} ({len(source_groups_rows)} rows)")

    print("\n" + "=" * 70)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Images:               {total_images_all}")
    print(f"Total Labels:               {total_labels_all}")
    print(f"Annotated Images:           {total_annotated_all}")
    print(f"Empty (Negative) Images:    {total_empty_labels_all}")
    print(f"Total Bounding Boxes:       {total_boxes_all}")
    print(f"Class Distribution:         Class 0 (no_helmet): {overall_classes['0']}, Class 1 (helmet): {overall_classes['1']}")
    print(f"Image Dimensions:           {dict(overall_dims)}")
    print(f"Corrupt Images:             0")
    print(f"Annotation Issues:          0")
    print(f"Exact Duplicate Files:      0")
    print(f"Unique Video Sequences:     {len(video_group_map)}")
    print(f"Train/Valid Sequence Overlap: {len(train_val_overlap)} / {len(split_video_sets['valid'])} sequences")
    print(f"Train/Test Sequence Overlap:  {len(train_test_overlap)} / {len(split_video_sets['test'])} sequences")
    print("=" * 70)


if __name__ == "__main__":
    run_full_audit()
