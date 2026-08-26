#!/usr/bin/env python3
"""Report exact and optional perceptual near-duplicate images without modifying data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.yolo_dataset import image_files
from src.utils.paths import ensure_parent, research_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dhash(path: Path) -> int | None:
    """Compute 64-bit difference hash using PIL or OpenCV if available."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = list(gray.getdata())
            # 8 rows x 8 differences
            diff = 0
            for r in range(8):
                for c in range(8):
                    left = pixels[r * 9 + c]
                    right = pixels[r * 9 + c + 1]
                    if left > right:
                        diff |= 1 << (r * 8 + c)
            return diff
    except ImportError:
        pass

    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        return sum((int(small[r, c]) > int(small[r, c + 1])) << (r * 8 + c) for r in range(8) for c in range(8))
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Report duplicates; never delete or move files.")
    parser.add_argument("--dataset", default="data/annotated/kawke_dibona_v1", help="Dataset directory to check")
    parser.add_argument("--near", action="store_true", help="Compare perceptual hashes (O(N^2) comparison)")
    parser.add_argument("--max-hamming-distance", type=int, default=5, help="Max bit distance for near-duplicates")
    parser.add_argument("--output", default="results/metrics/duplicate_report.json", help="Output report JSON")
    args = parser.parse_args()

    root = research_path(args.dataset)
    if not root.exists():
        parser.error(f"Dataset directory does not exist: {root}")

    print(f"Scanning for duplicates in: {root}")
    exact: dict[str, list[str]] = defaultdict(list)
    hashes: list[tuple[str, int]] = []
    
    files = image_files(root)
    for image in files:
        rel = str(image.relative_to(root))
        exact[sha256_file(image)].append(rel)
        if args.near:
            value = compute_dhash(image)
            if value is not None:
                hashes.append((rel, value))

    near: list[dict] = []
    if args.near:
        print(f"Comparing perceptual hashes across {len(hashes)} images...")
        for i, (left, a) in enumerate(hashes):
            for right, b in hashes[i + 1 :]:
                distance = (a ^ b).bit_count()
                if distance <= args.max_hamming_distance:
                    near.append({"image_a": left, "image_b": right, "hamming_distance": distance})

    exact_groups = [v for v in exact.values() if len(v) > 1]
    
    report = {
        "dataset_root": str(root),
        "images_checked": sum(map(len, exact.values())),
        "unique_exact_hashes": len(exact),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicates": exact_groups,
        "near_duplicates": near,
        "near_duplicate_method": "64-bit difference hash (PIL/OpenCV)" if args.near else "not requested",
    }

    output = ensure_parent(research_path(args.output))
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Exact duplicate groups found: {len(exact_groups)}")
    print(f"Wrote duplicate report to:    {output}")


if __name__ == "__main__":
    main()
