#!/usr/bin/env python3
"""Future group-exclusive splitting; never randomly distributes individual frames."""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path

parser = argparse.ArgumentParser(description="Create a split from reviewed video/sequence source groups.")
parser.add_argument("--manifest", required=True, help="CSV containing image_path,group_id")
parser.add_argument("--output", required=True)
parser.add_argument("--train", type=float, default=.70); parser.add_argument("--val", type=float, default=.15); parser.add_argument("--test", type=float, default=.15)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if abs(args.train + args.val + args.test - 1) > 1e-9: parser.error("train, val, and test must sum to 1.0")
groups = defaultdict(list)
with research_path(args.manifest).open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    if not reader.fieldnames or not {"image_path", "group_id"}.issubset(reader.fieldnames): parser.error("Manifest requires image_path and group_id.")
    for row in reader:
        if not row["group_id"].strip(): parser.error("Every image requires a reviewed, non-empty group_id.")
        groups[row["group_id"]].append(row["image_path"])
if len(groups) < 3: parser.error("At least three independent groups are required.")
ids = list(groups); random.Random(args.seed).shuffle(ids)
total = sum(map(len, groups.values())); targets = {"train": total*args.train, "val": total*args.val, "test": total*args.test}; counts = dict.fromkeys(targets, 0); assigned = {}
for group in ids:
    split = min(targets, key=lambda s: counts[s]/targets[s] if targets[s] else float("inf"))
    assigned[group] = split; counts[split] += len(groups[group])
output = ensure_parent(research_path(args.output))
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["image_path","group_id","split"]); writer.writeheader()
    for group, images in groups.items(): writer.writerows({"image_path": image, "group_id": group, "split": assigned[group]} for image in images)
print(f"Created group-exclusive split manifest: {output}\nImages by split: {counts}; groups: {len(groups)}")
