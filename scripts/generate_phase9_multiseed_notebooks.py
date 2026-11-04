#!/usr/bin/env python3
"""
Generate production-ready Kaggle notebooks and metadata for Phase 9 Multi-Seed Experiments.
Seeds to execute: 123, 2026 (Seed 42 is preserved and audited from Phase 6/7/8).
"""
import json
import os
from pathlib import Path

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.splitlines()]
    }

# Common header & setup code
setup_code = """# 1. Environment Setup & Hardware Verification
import subprocess
import sys
import time
import os
import json
import shutil
from pathlib import Path

print("=" * 70)
print("1. ENVIRONMENT SETUP & GPU VERIFICATION")
print("=" * 70)

# Install required dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyyaml", "pandas", "matplotlib", "seaborn"], check=True)

import torch
import ultralytics
from ultralytics import YOLO
import yaml
import pandas as pd
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"Ultralytics Version: {ultralytics.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Model: {device_name}")
    print(f"Total VRAM: {total_memory:.2f} GB")
else:
    raise RuntimeError("CRITICAL ERROR: CUDA GPU is not available! Halt immediately.")
"""

eval_helper_code = """# 3. Evaluation Helper Function for Multi-Population Validation
def evaluate_checkpoint(model_path, data_yaml_path, eval_name, output_dir):
    print(f"Evaluating {model_path} on {eval_name}...")
    model = YOLO(str(model_path))
    val_results = model.val(
        data=str(data_yaml_path),
        split="val",
        imgsz=640,
        batch=32,
        device=0,
        project=str(output_dir.parent),
        name=str(output_dir.name),
        plots=True,
        save_json=False,
        verbose=True,
        exist_ok=True
    )
    
    p = float(val_results.box.mp)
    r = float(val_results.box.mr)
    f1 = float(2 * p * r / (p + r + 1e-16))
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    
    per_class = {}
    names = val_results.names
    for i, cls_id in enumerate(val_results.box.ap_class_index):
        cls_name = names.get(cls_id, f"class_{cls_id}")
        cls_p = float(val_results.box.p[i]) if len(val_results.box.p) > i else 0.0
        cls_r = float(val_results.box.r[i]) if len(val_results.box.r) > i else 0.0
        cls_f1 = float(2 * cls_p * cls_r / (cls_p + cls_r + 1e-16))
        cls_map50 = float(val_results.box.ap50[i]) if len(val_results.box.ap50) > i else 0.0
        cls_map50_95 = float(val_results.box.ap[i]) if len(val_results.box.ap) > i else 0.0
        per_class[cls_name] = {
            "class_id": int(cls_id),
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
            "f1": round(cls_f1, 4),
            "map50": round(cls_map50, 4),
            "map50_95": round(cls_map50_95, 4)
        }
        
    cm = None
    if hasattr(val_results, "confusion_matrix") and hasattr(val_results.confusion_matrix, "matrix"):
        cm = val_results.confusion_matrix.matrix.tolist()
        
    speed = {}
    if hasattr(val_results, "speed"):
        speed = {k: round(v, 2) for k, v in val_results.speed.items()}
        
    return {
        "evaluation_name": eval_name,
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4)
        },
        "per_class": per_class,
        "confusion_matrix": cm,
        "speed_ms": speed
    }
"""

def generate_exp_new_notebook():
    cells = [
        cell("""# Dual-Camera Motorcycle Helmet Violation Detection
# PHASE 9: MULTI-SEED REPRODUCIBILITY STUDY
## Model: EXP-NEW (Seeds 123, 2026)
Dataset: Mobile Street Helmet Expansion v2 (1,285 train / 276 val / 280 test)
Multi-Population Evaluation: New Test (280), Old Test (829), Combined Test (1,109)
""", "markdown"),
        cell(setup_code),
        cell("""# 2. Dataset Mount Discovery & Path Verification
print("=" * 70)
print("2. DATASET MOUNT DISCOVERY & INTEGRITY VERIFICATION")
print("=" * 70)

input_dir = Path("/kaggle/input")
new_ds_path = None
old_ds_path = None

for p in input_dir.rglob("*"):
    if p.is_dir() and "helmet-dataset-expansion-v2" in p.name.lower():
        new_ds_path = p
    elif p.is_dir() and "old-dataset-v1-fixed" in p.name.lower():
        old_ds_path = p

if not new_ds_path:
    for p in input_dir.glob("**/*"):
        if (p / "images/train").exists():
            new_ds_path = p
            break

if not old_ds_path:
    for p in input_dir.glob("**/*"):
        if (p / "train/images").exists():
            old_ds_path = p
            break

print(f"Resolved New Dataset Root: {new_ds_path}")
print(f"Resolved Old Dataset Root: {old_ds_path}")

assert new_ds_path and new_ds_path.exists(), "New dataset not found in /kaggle/input!"
assert old_ds_path and old_ds_path.exists(), "Old dataset not found in /kaggle/input!"

n_train = len(list((new_ds_path / "images/train").glob("*.jpg")))
n_val   = len(list((new_ds_path / "images/val").glob("*.jpg")))
n_test  = len(list((new_ds_path / "images/test").glob("*.jpg")))
print(f"New Dataset Image Counts -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
assert n_train == 1285, f"Expected 1285 new train images, found {n_train}"
assert n_val == 276, f"Expected 276 new val images, found {n_val}"
assert n_test == 280, f"Expected 280 new test images, found {n_test}"

o_train = len(list((old_ds_path / "train/images").glob("*.jpg")))
o_val   = len(list((old_ds_path / "val/images").glob("*.jpg")))
o_test  = len(list((old_ds_path / "test/images").glob("*.jpg")))
print(f"Old Dataset Image Counts -> Train: {o_train}, Val: {o_val}, Test: {o_test}")
assert o_train == 3866, f"Expected 3866 old train images, found {o_train}"
assert o_val == 852, f"Expected 852 old val images, found {o_val}"
assert o_test == 829, f"Expected 829 old test images, found {o_test}"
"""),
        cell("""# 3. Materialize Combined Test Dataset & Prepare YAMLs
work_dir = Path("/kaggle/working")
comb_test_dir = work_dir / "combined_test_dataset"
comb_test_imgs = comb_test_dir / "images/val"
comb_test_lbls = comb_test_dir / "labels/val"
comb_test_imgs.mkdir(parents=True, exist_ok=True)
comb_test_lbls.mkdir(parents=True, exist_ok=True)

# Link Old Test (829 images)
old_test_imgs = old_ds_path / "test/images"
old_test_lbls = old_ds_path / "test/labels"
for img in old_test_imgs.glob("*.jpg"):
    dst_i = comb_test_imgs / f"old_{img.name}"
    dst_l = comb_test_lbls / f"old_{img.stem}.txt"
    src_l = old_test_lbls / f"{img.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img, dst_i)
        except: shutil.copy2(img, dst_i)
    if src_l.exists() and not dst_l.exists():
        try: os.symlink(src_l, dst_l)
        except: shutil.copy2(src_l, dst_l)

# Link New Test (280 images)
new_test_imgs = new_ds_path / "images/test"
new_test_lbls = new_ds_path / "labels/test"
for img in new_test_imgs.glob("*.jpg"):
    dst_i = comb_test_imgs / f"new_{img.name}"
    dst_l = comb_test_lbls / f"new_{img.stem}.txt"
    src_l = new_test_lbls / f"{img.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img, dst_i)
        except: shutil.copy2(img, dst_i)
    if src_l.exists() and not dst_l.exists():
        try: os.symlink(src_l, dst_l)
        except: shutil.copy2(src_l, dst_l)

tot_test = len(list(comb_test_imgs.glob("*.jpg")))
print(f"Materialized Combined Test Images: {tot_test} (Expected: 1,109)")
assert tot_test == 1109, f"Expected 1109 combined test images, found {tot_test}"

# Create Data YAMLs
train_new_yaml = work_dir / "train_new.yaml"
with open(train_new_yaml, "w") as f:
    f.write(f"path: {new_ds_path}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_new_yaml = work_dir / "eval_new_test.yaml"
with open(eval_new_yaml, "w") as f:
    f.write(f"path: {new_ds_path}\\ntrain: images/test\\nval: images/test\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_old_yaml = work_dir / "eval_old_test.yaml"
with open(eval_old_yaml, "w") as f:
    f.write(f"path: {old_ds_path}\\ntrain: test/images\\nval: test/images\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_comb_yaml = work_dir / "eval_comb_test.yaml"
with open(eval_comb_yaml, "w") as f:
    f.write(f"path: {comb_test_dir}\\ntrain: images/val\\nval: images/val\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

print("All YAML configs created successfully!")
"""),
        cell(eval_helper_code),
        cell("""# 4. Execute Multi-Seed EXP-NEW Training (Seeds 123, 2026)
SEEDS = [123, 2026]
results_by_seed = {}

for seed in SEEDS:
    print("=" * 70)
    print(f"STARTING EXP-NEW RUN: SEED {seed}")
    print("=" * 70)
    
    run_dir = work_dir / "exp_new" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Official COCO-pretrained YOLOv8n initialization
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    
    train_results = model.train(
        data=str(train_new_yaml),
        epochs=50,
        patience=20,
        batch=32,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001667,
        weight_decay=0.0005,
        momentum=0.9,
        seed=seed,
        deterministic=True,
        device=0,
        project=str(run_dir.parent),
        name=f"seed_{seed}",
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )
    
    dur = time.time() - t0
    best_weights = run_dir / "weights/best.pt"
    print(f"Seed {seed} Training finished in {dur:.2f}s. Best weights: {best_weights}")
    
    # Evaluate across 3 test populations
    eval_new = evaluate_checkpoint(best_weights, eval_new_yaml, f"New Held-Out Test (Seed {seed})", run_dir / "eval_new")
    eval_old = evaluate_checkpoint(best_weights, eval_old_yaml, f"Old CCTV Held-Out Test (Seed {seed})", run_dir / "eval_old")
    eval_comb = evaluate_checkpoint(best_weights, eval_comb_yaml, f"Combined Held-Out Test (Seed {seed})", run_dir / "eval_comb")
    
    summary = {
        "experiment": "EXP-NEW",
        "seed": seed,
        "training_duration_seconds": dur,
        "training_duration_minutes": round(dur / 60, 2),
        "best_weights": str(best_weights),
        "evaluations": {
            "new_test": eval_new,
            "old_test": eval_old,
            "combined_test": eval_comb
        }
    }
    
    (run_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    results_by_seed[str(seed)] = summary
    print(f"Completed and saved evaluation for Seed {seed}!")

print("=" * 70)
print("ALL EXP-NEW MULTI-SEED RUNS COMPLETE!")
print("=" * 70)
(work_dir / "exp_new_multiseed_summary.json").write_text(json.dumps(results_by_seed, indent=2))
""")
    ]
    return create_notebook(cells)

def generate_exp_comb_notebook():
    cells = [
        cell("""# Dual-Camera Motorcycle Helmet Violation Detection
# PHASE 9: MULTI-SEED REPRODUCIBILITY STUDY
## Model: EXP-COMBINED (Seeds 123, 2026)
Dataset: Combined Multi-Domain (5,151 train / 1,128 val / 1,109 test)
Multi-Population Evaluation: New Test (280), Old Test (829), Combined Test (1,109)
""", "markdown"),
        cell(setup_code),
        cell("""# 2. Dataset Mount Discovery & Path Verification
print("=" * 70)
print("2. DATASET MOUNT DISCOVERY & INTEGRITY VERIFICATION")
print("=" * 70)

input_dir = Path("/kaggle/input")
new_ds_path = None
old_ds_path = None

for p in input_dir.rglob("*"):
    if p.is_dir() and "helmet-dataset-expansion-v2" in p.name.lower():
        new_ds_path = p
    elif p.is_dir() and "old-dataset-v1-fixed" in p.name.lower():
        old_ds_path = p

if not new_ds_path:
    for p in input_dir.glob("**/*"):
        if (p / "images/train").exists():
            new_ds_path = p
            break

if not old_ds_path:
    for p in input_dir.glob("**/*"):
        if (p / "train/images").exists():
            old_ds_path = p
            break

print(f"Resolved New Dataset Root: {new_ds_path}")
print(f"Resolved Old Dataset Root: {old_ds_path}")

assert new_ds_path and new_ds_path.exists(), "New dataset not found in /kaggle/input!"
assert old_ds_path and old_ds_path.exists(), "Old dataset not found in /kaggle/input!"

n_train = len(list((new_ds_path / "images/train").glob("*.jpg")))
n_val   = len(list((new_ds_path / "images/val").glob("*.jpg")))
n_test  = len(list((new_ds_path / "images/test").glob("*.jpg")))
print(f"New Dataset Image Counts -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
assert n_train == 1285, f"Expected 1285 new train images, found {n_train}"
assert n_val == 276, f"Expected 276 new val images, found {n_val}"
assert n_test == 280, f"Expected 280 new test images, found {n_test}"

o_train = len(list((old_ds_path / "train/images").glob("*.jpg")))
o_val   = len(list((old_ds_path / "val/images").glob("*.jpg")))
o_test  = len(list((old_ds_path / "test/images").glob("*.jpg")))
print(f"Old Dataset Image Counts -> Train: {o_train}, Val: {o_val}, Test: {o_test}")
assert o_train == 3866, f"Expected 3866 old train images, found {o_train}"
assert o_val == 852, f"Expected 852 old val images, found {o_val}"
assert o_test == 829, f"Expected 829 old test images, found {o_test}"
"""),
        cell("""# 3. Materialize Combined Dataset & Test Dataset
work_dir = Path("/kaggle/working")
comb_dir = work_dir / "combined_dataset"
comb_tr_imgs = comb_dir / "images/train"
comb_tr_lbls = comb_dir / "labels/train"
comb_va_imgs = comb_dir / "images/val"
comb_va_lbls = comb_dir / "labels/val"

comb_tr_imgs.mkdir(parents=True, exist_ok=True)
comb_tr_lbls.mkdir(parents=True, exist_ok=True)
comb_va_imgs.mkdir(parents=True, exist_ok=True)
comb_va_lbls.mkdir(parents=True, exist_ok=True)

# Link Old Train (3,866)
for img_p in (old_ds_path / "train/images").glob("*.jpg"):
    dst_i = comb_tr_imgs / f"old_{img_p.name}"
    lbl_p = old_ds_path / "train/labels" / f"{img_p.stem}.txt"
    dst_l = comb_tr_lbls / f"old_{img_p.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img_p, dst_i)
        except: shutil.copy2(img_p, dst_i)
    if lbl_p.exists() and not dst_l.exists():
        try: os.symlink(lbl_p, dst_l)
        except: shutil.copy2(lbl_p, dst_l)

# Link New Train (1,285)
for img_p in (new_ds_path / "images/train").glob("*.jpg"):
    dst_i = comb_tr_imgs / f"new_{img_p.name}"
    lbl_p = new_ds_path / "labels/train" / f"{img_p.stem}.txt"
    dst_l = comb_tr_lbls / f"new_{img_p.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img_p, dst_i)
        except: shutil.copy2(img_p, dst_i)
    if lbl_p.exists() and not dst_l.exists():
        try: os.symlink(lbl_p, dst_l)
        except: shutil.copy2(lbl_p, dst_l)

# Link Old Val (852)
for img_p in (old_ds_path / "val/images").glob("*.jpg"):
    dst_i = comb_va_imgs / f"old_{img_p.name}"
    lbl_p = old_ds_path / "val/labels" / f"{img_p.stem}.txt"
    dst_l = comb_va_lbls / f"old_{img_p.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img_p, dst_i)
        except: shutil.copy2(img_p, dst_i)
    if lbl_p.exists() and not dst_l.exists():
        try: os.symlink(lbl_p, dst_l)
        except: shutil.copy2(lbl_p, dst_l)

# Link New Val (276)
for img_p in (new_ds_path / "images/val").glob("*.jpg"):
    dst_i = comb_va_imgs / f"new_{img_p.name}"
    lbl_p = new_ds_path / "labels/val" / f"{img_p.stem}.txt"
    dst_l = comb_va_lbls / f"new_{img_p.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img_p, dst_i)
        except: shutil.copy2(img_p, dst_i)
    if lbl_p.exists() and not dst_l.exists():
        try: os.symlink(lbl_p, dst_l)
        except: shutil.copy2(lbl_p, dst_l)

tr_cnt = len(list(comb_tr_imgs.glob("*.jpg")))
va_cnt = len(list(comb_va_imgs.glob("*.jpg")))
print(f"Materialized Combined Training Images: {tr_cnt} (Expected: 5,151)")
print(f"Materialized Combined Validation Images: {va_cnt} (Expected: 1,128)")
assert tr_cnt == 5151, f"Expected 5151 train images, found {tr_cnt}"
assert va_cnt == 1128, f"Expected 1128 val images, found {va_cnt}"

# Materialize Combined Test Set (1,109)
comb_test_dir = work_dir / "combined_test_dataset"
comb_test_imgs = comb_test_dir / "images/val"
comb_test_lbls = comb_test_dir / "labels/val"
comb_test_imgs.mkdir(parents=True, exist_ok=True)
comb_test_lbls.mkdir(parents=True, exist_ok=True)

old_test_imgs = old_ds_path / "test/images"
old_test_lbls = old_ds_path / "test/labels"
for img in old_test_imgs.glob("*.jpg"):
    dst_i = comb_test_imgs / f"old_{img.name}"
    dst_l = comb_test_lbls / f"old_{img.stem}.txt"
    src_l = old_test_lbls / f"{img.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img, dst_i)
        except: shutil.copy2(img, dst_i)
    if src_l.exists() and not dst_l.exists():
        try: os.symlink(src_l, dst_l)
        except: shutil.copy2(src_l, dst_l)

new_test_imgs = new_ds_path / "images/test"
new_test_lbls = new_ds_path / "labels/test"
for img in new_test_imgs.glob("*.jpg"):
    dst_i = comb_test_imgs / f"new_{img.name}"
    dst_l = comb_test_lbls / f"new_{img.stem}.txt"
    src_l = new_test_lbls / f"{img.stem}.txt"
    if not dst_i.exists():
        try: os.symlink(img, dst_i)
        except: shutil.copy2(img, dst_i)
    if src_l.exists() and not dst_l.exists():
        try: os.symlink(src_l, dst_l)
        except: shutil.copy2(src_l, dst_l)

tot_test = len(list(comb_test_imgs.glob("*.jpg")))
print(f"Materialized Combined Test Images: {tot_test} (Expected: 1,109)")
assert tot_test == 1109, f"Expected 1109 combined test images, found {tot_test}"

# Create Data YAMLs
train_comb_yaml = work_dir / "train_combined.yaml"
with open(train_comb_yaml, "w") as f:
    f.write(f"path: {comb_dir}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_new_yaml = work_dir / "eval_new_test.yaml"
with open(eval_new_yaml, "w") as f:
    f.write(f"path: {new_ds_path}\\ntrain: images/test\\nval: images/test\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_old_yaml = work_dir / "eval_old_test.yaml"
with open(eval_old_yaml, "w") as f:
    f.write(f"path: {old_ds_path}\\ntrain: test/images\\nval: test/images\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

eval_comb_yaml = work_dir / "eval_comb_test.yaml"
with open(eval_comb_yaml, "w") as f:
    f.write(f"path: {comb_test_dir}\\ntrain: images/val\\nval: images/val\\nnames:\\n  0: no_helmet\\n  1: helmet\\nnc: 2\\n")

print("All Combined YAML configs created successfully!")
"""),
        cell(eval_helper_code),
        cell("""# 4. Execute Multi-Seed EXP-COMBINED Training (Seeds 123, 2026)
SEEDS = [123, 2026]
results_by_seed = {}

for seed in SEEDS:
    print("=" * 70)
    print(f"STARTING EXP-COMBINED RUN: SEED {seed}")
    print("=" * 70)
    
    run_dir = work_dir / "exp_combined" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Official COCO-pretrained YOLOv8n initialization
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    
    train_results = model.train(
        data=str(train_comb_yaml),
        epochs=50,
        patience=20,
        batch=32,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001667,
        weight_decay=0.0005,
        momentum=0.9,
        seed=seed,
        deterministic=True,
        device=0,
        project=str(run_dir.parent),
        name=f"seed_{seed}",
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )
    
    dur = time.time() - t0
    best_weights = run_dir / "weights/best.pt"
    print(f"Seed {seed} Training finished in {dur:.2f}s. Best weights: {best_weights}")
    
    # Evaluate across 3 test populations
    eval_new = evaluate_checkpoint(best_weights, eval_new_yaml, f"New Held-Out Test (Seed {seed})", run_dir / "eval_new")
    eval_old = evaluate_checkpoint(best_weights, eval_old_yaml, f"Old CCTV Held-Out Test (Seed {seed})", run_dir / "eval_old")
    eval_comb = evaluate_checkpoint(best_weights, eval_comb_yaml, f"Combined Held-Out Test (Seed {seed})", run_dir / "eval_comb")
    
    summary = {
        "experiment": "EXP-COMBINED",
        "seed": seed,
        "training_duration_seconds": dur,
        "training_duration_minutes": round(dur / 60, 2),
        "best_weights": str(best_weights),
        "evaluations": {
            "new_test": eval_new,
            "old_test": eval_old,
            "combined_test": eval_comb
        }
    }
    
    (run_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    results_by_seed[str(seed)] = summary
    print(f"Completed and saved evaluation for Seed {seed}!")

print("=" * 70)
print("ALL EXP-COMBINED MULTI-SEED RUNS COMPLETE!")
print("=" * 70)
(work_dir / "exp_combined_multiseed_summary.json").write_text(json.dumps(results_by_seed, indent=2))
""")
    ]
    return create_notebook(cells)

def main():
    exp_new_dir = Path("notebooks/phase9_multiseed/exp_new")
    exp_comb_dir = Path("notebooks/phase9_multiseed/exp_comb")
    exp_new_dir.mkdir(parents=True, exist_ok=True)
    exp_comb_dir.mkdir(parents=True, exist_ok=True)
    
    nb_new = generate_exp_new_notebook()
    (exp_new_dir / "kaggle_phase9_exp_new.ipynb").write_text(json.dumps(nb_new, indent=2))
    
    meta_new = {
        "id": "allfaz/helmet-detection-phase9-exp-new",
        "title": "Helmet Detection Phase 9 EXP-NEW MultiSeed",
        "code_file": "kaggle_phase9_exp_new.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "allfaz/helmet-dataset-expansion-v2",
            "allfaz/old-dataset-v1-fixed"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    (exp_new_dir / "kernel-metadata.json").write_text(json.dumps(meta_new, indent=2))
    
    nb_comb = generate_exp_comb_notebook()
    (exp_comb_dir / "kaggle_phase9_exp_comb.ipynb").write_text(json.dumps(nb_comb, indent=2))
    
    meta_comb = {
        "id": "allfaz/helmet-detection-phase9-exp-comb",
        "title": "Helmet Detection Phase 9 EXP-COMBINED MultiSeed",
        "code_file": "kaggle_phase9_exp_comb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "allfaz/helmet-dataset-expansion-v2",
            "allfaz/old-dataset-v1-fixed"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    (exp_comb_dir / "kernel-metadata.json").write_text(json.dumps(meta_comb, indent=2))
    
    print("Successfully generated Phase 9 Multi-Seed Notebooks and Metadata!")

if __name__ == "__main__":
    main()
