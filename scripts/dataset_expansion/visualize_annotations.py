#!/usr/bin/env python3
"""visualize_annotations.py: Render visual bounding box overlays using PIL.

Generates qualitative verification previews for audits and reports.
Green boxes: Class 1 (helmet)
Red boxes:   Class 0 (no_helmet)
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = PROJECT_ROOT / "data/helmet_dataset_expansion/original"

COLOR_MAP = {
    0: (230, 0, 0),    # Red for no_helmet
    1: (0, 200, 0)     # Green for helmet
}
NAME_MAP = {
    0: "no_helmet",
    1: "helmet"
}


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path) -> int:
    """Draw YOLO normalized boxes on an image and save output."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return 0

    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    box_count = 0

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w_img
                    yc = float(parts[2]) * h_img
                    bw = float(parts[3]) * w_img
                    bh = float(parts[4]) * h_img
                except ValueError:
                    continue

                xmin = int(round(xc - bw / 2.0))
                ymin = int(round(yc - bh / 2.0))
                xmax = int(round(xc + bw / 2.0))
                ymax = int(round(yc + bh / 2.0))

                xmin = max(0, min(w_img - 1, xmin))
                ymin = max(0, min(h_img - 1, ymin))
                xmax = max(0, min(w_img - 1, xmax))
                ymax = max(0, min(h_img - 1, ymax))

                color = COLOR_MAP.get(cls_id, (255, 255, 255))
                cname = NAME_MAP.get(cls_id, f"cls_{cls_id}")

                thickness = max(2, int(min(w_img, h_img) / 300))
                # Draw rectangle outline
                for t in range(thickness):
                    draw.rectangle([xmin - t, ymin - t, xmax + t, ymax + t], outline=color)

                # Draw label text badge
                badge_w = max(60, int(len(cname) * 8))
                badge_h = 16
                draw.rectangle([xmin, max(0, ymin - badge_h), xmin + badge_w, ymin], fill=color)
                draw.text((xmin + 4, max(0, ymin - badge_h) + 1), cname, fill=(255, 255, 255))
                box_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=90)
    return box_count


def render_directory_previews(image_dir: Path, label_dir: Path, output_dir: Path, limit: int = 50):
    """Render preview images for qualitative review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith('.')])
    
    rendered = 0
    for img_path in images:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        out_path = output_dir / f"{img_path.stem}_preview.jpg"
        draw_yolo_boxes(img_path, lbl_path, out_path)
        rendered += 1
        if limit and rendered >= limit:
            break
    print(f"Rendered {rendered} qualitative preview visualizations to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default=str(ORIGINAL_DIR))
    parser.add_argument("--lbl-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    render_directory_previews(Path(args.img_dir), Path(args.lbl_dir), Path(args.out_dir), args.limit)
