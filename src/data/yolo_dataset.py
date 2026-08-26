"""Read-only validation helpers for YOLO detection datasets.

Supports both split directory layout (train/images, train/labels, etc.)
and single-folder paired layout.
Single-pass, high-performance dataset auditor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import struct

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    """Return sorted list of image files under root."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def parse_image_bytes(data: bytes, suffix: str = ".jpg") -> tuple[tuple[int, int] | None, str | None]:
    """Parse image dimensions and integrity directly from in-memory byte buffer in milliseconds."""
    if len(data) < 10:
        return None, "File too small to be valid image"

    # JPEG format
    if data[:2] == b"\xff\xd8":
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
            if marker in (0xDA, 0xD9):  # SOS or EOI - headers complete
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
                    return None, "Truncated SOF segment"
                height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
                return (width, height), None
            offset += seg_len
        return None, "No SOF marker found in JPEG"

    # PNG format
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return (width, height), None

    return None, f"Unsupported format: {suffix}"


def inspect_yolo_dataset(root: Path, class_count: int | None = None, min_box_size: float = 0.0) -> dict:
    """Inspect dataset without modifying, moving, or deleting any files."""
    root = root.resolve()
    split_dirs = [d for d in root.iterdir() if d.is_dir() and d.name in ("train", "val", "valid", "test")]

    report = {
        "dataset_root": str(root),
        "total_images": 0,
        "readable_images": 0,
        "corrupted_images": 0,
        "total_label_files": 0,
        "annotated_images": 0,
        "empty_label_files": 0,
        "images_without_annotations": 0,
        "orphan_annotations": 0,
        "annotation_issues": 0,
        "total_objects": 0,
        "class_counts": Counter(),
        "dimension_distribution": Counter(),
        "exact_duplicate_groups": 0,
        "splits": {},
        "issues": [],
    }

    sha_map = defaultdict(list)

    if split_dirs:
        for sdir in sorted(split_dirs, key=lambda d: d.name):
            sname = sdir.name
            img_dir = sdir / "images"
            lbl_dir = sdir / "labels"

            img_files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")) if img_dir.exists() else []
            lbl_files = sorted(p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and not p.name.startswith(".")) if lbl_dir.exists() else []

            img_map = {f.stem: f for f in img_files}
            lbl_map = {f.stem: f for f in lbl_files}

            s_classes = Counter()
            s_objects = 0
            s_empty = 0
            s_annotated = 0

            for stem, img_path in img_map.items():
                rel_img = str(img_path.relative_to(root))
                report["total_images"] += 1

                try:
                    data = img_path.read_bytes()
                    sha = hashlib.sha256(data).hexdigest()
                    sha_map[sha].append(rel_img)
                    dims, err = parse_image_bytes(data, img_path.suffix.lower())
                except Exception as e:
                    dims, err = None, str(e)

                if err:
                    report["corrupted_images"] += 1
                    report["issues"].append({"type": "corrupted_image", "file": rel_img, "error": err})
                else:
                    report["readable_images"] += 1
                    report["dimension_distribution"][f"{dims[0]}x{dims[1]}"] += 1

                if stem not in lbl_map:
                    report["images_without_annotations"] += 1
                    report["issues"].append({"type": "missing_annotation", "file": rel_img})
                    continue

                lbl_path = lbl_map[stem]
                report["total_label_files"] += 1

                try:
                    content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                except Exception as e:
                    report["annotation_issues"] += 1
                    report["issues"].append({"type": "unreadable_label", "file": str(lbl_path.relative_to(root)), "error": str(e)})
                    continue

                if not lines:
                    report["empty_label_files"] += 1
                    s_empty += 1
                    continue

                report["annotated_images"] += 1
                s_annotated += 1

                for num, line in enumerate(lines, 1):
                    parts = line.split()
                    if len(parts) != 5:
                        report["annotation_issues"] += 1
                        report["issues"].append({"type": "malformed_annotation", "file": str(lbl_path.relative_to(root)), "line": num, "content": line})
                        continue
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                    except ValueError:
                        report["annotation_issues"] += 1
                        report["issues"].append({"type": "non_numeric_annotation", "file": str(lbl_path.relative_to(root)), "line": num, "content": line})
                        continue

                    issue = None
                    if cls_id < 0 or (class_count is not None and cls_id >= class_count):
                        issue = f"invalid_class_id_{cls_id}"
                    elif not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        issue = "bbox_outside_normalized_bounds"
                    elif w <= 0 or h <= 0:
                        issue = "zero_or_negative_area"
                    elif (x - w / 2 < -1e-4) or (x + w / 2 > 1.0001) or (y - h / 2 < -1e-4) or (y + h / 2 > 1.0001):
                        issue = "bbox_exceeds_image_boundaries"
                    elif min(w, h) < min_box_size:
                        issue = "extremely_small_bbox"

                    if issue:
                        report["annotation_issues"] += 1
                        report["issues"].append({"type": issue, "file": str(lbl_path.relative_to(root)), "line": num, "content": line})
                    else:
                        report["total_objects"] += 1
                        report["class_counts"][str(cls_id)] += 1
                        s_objects += 1
                        s_classes[str(cls_id)] += 1

            for stem, lbl_path in lbl_map.items():
                if stem not in img_map:
                    report["orphan_annotations"] += 1
                    report["issues"].append({"type": "orphan_annotation", "file": str(lbl_path.relative_to(root))})

            report["splits"][sname] = {
                "images": len(img_files),
                "annotated_images": s_annotated,
                "empty_labels": s_empty,
                "objects": s_objects,
                "class_counts": dict(sorted(s_classes.items(), key=lambda x: int(x[0]))),
            }

    else:
        img_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))
        for img_path in img_files:
            rel_img = str(img_path.relative_to(root))
            report["total_images"] += 1
            try:
                data = img_path.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                sha_map[sha].append(rel_img)
                dims, err = parse_image_bytes(data, img_path.suffix.lower())
            except Exception as e:
                dims, err = None, str(e)

            if err:
                report["corrupted_images"] += 1
                report["issues"].append({"type": "corrupted_image", "file": rel_img, "error": err})
            else:
                report["readable_images"] += 1
                report["dimension_distribution"][f"{dims[0]}x{dims[1]}"] += 1

            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                report["images_without_annotations"] += 1
                report["issues"].append({"type": "missing_annotation", "file": rel_img})
                continue

            report["total_label_files"] += 1
            content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if not lines:
                report["empty_label_files"] += 1
                continue
            report["annotated_images"] += 1
            for num, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    report["annotation_issues"] += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    report["total_objects"] += 1
                    report["class_counts"][str(cls_id)] += 1
                except ValueError:
                    report["annotation_issues"] += 1

    exact_duplicates = [v for v in sha_map.values() if len(v) > 1]
    report["exact_duplicate_groups"] = len(exact_duplicates)
    report["class_counts"] = dict(sorted(report["class_counts"].items(), key=lambda x: int(x[0])))
    report["dimension_distribution"] = dict(report["dimension_distribution"])
    return report
