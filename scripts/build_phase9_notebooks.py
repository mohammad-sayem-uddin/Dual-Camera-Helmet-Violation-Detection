#!/usr/bin/env python3
"""Build self-contained, reproducible Kaggle notebooks for Phase 9 Multi-Seed Experiments."""
import json
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

eval_helper_code = """# Evaluation Helper Function for Multi-Population Validation
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

print("Build script template ready.")

# Complete generator for EXP-NEW multi-seed notebook
exp_new_cells = [
    cell("""# Dual-Camera Motorcycle Helmet Violation Detection
# PHASE 9: MULTI-SEED REPRODUCIBILITY & STATISTICAL ROBUSTNESS STUDY
## Experiment Group A: EXP-NEW (Seeds 42, 123, 2026)
""", "markdown"),
    cell(setup_code),
    cell("""# 2. Dataset Discovery & Path Setup for EXP-NEW
print("=" * 70)
print("2. DATASET DISCOVERY & PATH VERIFICATION")
print("=" * 70)

input_dir = Path("/kaggle/input")
new_ds_path = None
old_ds_path = None

for p in input_dir.iterdir():
    name = p.name.lower()
    if "expansion" in name or "helmet-dataset" in name:
        if (p / "training_ready").exists():
            new_ds_path = p / "training_ready"
        elif (p / "images/train").exists():
            new_ds_path = p
    if "old" in name or "sequence" in name or "fixed" in name or "dibona" in name or "kawke" in name:
        if (p / "test/images").exists():
            old_ds_path = p
        elif (p / "images/test").exists():
            old_ds_path = p

print(f"New Expansion Dataset Path: {new_ds_path}")
print(f"Old CCTV Dataset Path: {old_ds_path}")
assert new_ds_path is not None, "Failed to find New Expansion Dataset!"
assert old_ds_path is not None, "Failed to find Old CCTV Dataset!"
"""),
    cell("""# 3. Create Multi-Population Validation YAML Files
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

# Write YAMLs
eval_new_yaml = work_dir / "eval_new_test.yaml"
eval_new_yaml.write_text(f\"\"\"train: {new_ds_path}/images/train
val: {new_ds_path}/images/test
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

eval_old_yaml = work_dir / "eval_old_test.yaml"
eval_old_yaml.write_text(f\"\"\"train: {old_ds_path}/train/images
val: {old_test_imgs}
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

eval_comb_yaml = work_dir / "eval_comb_test.yaml"
eval_comb_yaml.write_text(f\"\"\"train: {new_ds_path}/images/train
val: {comb_test_imgs}
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

train_new_yaml = work_dir / "train_new.yaml"
train_new_yaml.write_text(f\"\"\"train: {new_ds_path}/images/train
val: {new_ds_path}/images/val
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

print("YAML configurations created successfully!")
"""),
    cell(eval_helper_code),
    cell("""# 4. Execute Multi-Seed EXP-NEW Training (Seeds 42, 123, 2026)
SEEDS = [42, 123, 2026]
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
    eval_new = evaluate_checkpoint(best_weights, eval_new_yaml, f"New Test (Seed {seed})", run_dir / "eval_new")
    eval_old = evaluate_checkpoint(best_weights, eval_old_yaml, f"Old CCTV Test (Seed {seed})", run_dir / "eval_old")
    eval_comb = evaluate_checkpoint(best_weights, eval_comb_yaml, f"Combined Test (Seed {seed})", run_dir / "eval_comb")
    
    summary = {
        "experiment": "EXP-NEW",
        "seed": seed,
        "training_duration_seconds": dur,
        "evaluations": {
            "new_test": eval_new,
            "old_test": eval_old,
            "combined_test": eval_comb
        }
    }
    
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    results_by_seed[seed] = summary

print("All EXP-NEW runs completed successfully!")
(work_dir / "exp_new_multiseed_summary.json").write_text(json.dumps(results_by_seed, indent=2))
""")
]

nb_exp_new = create_notebook(exp_new_cells)
Path("notebooks/phase9_multiseed/kaggle_phase9_exp_new.ipynb").write_text(json.dumps(nb_exp_new, indent=2))
print("Saved kaggle_phase9_exp_new.ipynb!")

# Complete generator for EXP-COMBINED multi-seed notebook
exp_comb_cells = [
    cell("""# Dual-Camera Motorcycle Helmet Violation Detection
# PHASE 9: MULTI-SEED REPRODUCIBILITY & STATISTICAL ROBUSTNESS STUDY
## Experiment Group B: EXP-COMBINED (Seeds 42, 123, 2026)
""", "markdown"),
    cell(setup_code),
    cell("""# 2. Dataset Discovery & Path Setup for EXP-COMBINED
print("=" * 70)
print("2. DATASET DISCOVERY & COMBINED DATASET MATERIALIZATION")
print("=" * 70)

input_dir = Path("/kaggle/input")
new_ds_path = None
old_ds_path = None

for p in input_dir.iterdir():
    name = p.name.lower()
    if "expansion" in name or "helmet-dataset" in name:
        if (p / "training_ready").exists():
            new_ds_path = p / "training_ready"
        elif (p / "images/train").exists():
            new_ds_path = p
    if "old" in name or "sequence" in name or "fixed" in name or "dibona" in name or "kawke" in name:
        if (p / "test/images").exists():
            old_ds_path = p
        elif (p / "images/test").exists():
            old_ds_path = p

print(f"New Expansion Dataset Path: {new_ds_path}")
print(f"Old CCTV Dataset Path: {old_ds_path}")
assert new_ds_path is not None, "Failed to find New Expansion Dataset!"
assert old_ds_path is not None, "Failed to find Old CCTV Dataset!"

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

# Link Old Training Images & Labels (3,866)
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

# Link New Training Images & Labels (1,285)
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

# Link Old Validation Images & Labels (852)
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

# Link New Validation Images & Labels (276)
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
"""),
    cell("""# 3. Create Multi-Population Validation YAML Files
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

# Write YAMLs
eval_new_yaml = work_dir / "eval_new_test.yaml"
eval_new_yaml.write_text(f\"\"\"train: {comb_tr_imgs}
val: {new_ds_path}/images/test
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

eval_old_yaml = work_dir / "eval_old_test.yaml"
eval_old_yaml.write_text(f\"\"\"train: {comb_tr_imgs}
val: {old_test_imgs}
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

eval_comb_yaml = work_dir / "eval_comb_test.yaml"
eval_comb_yaml.write_text(f\"\"\"train: {comb_tr_imgs}
val: {comb_test_imgs}
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

train_comb_yaml = work_dir / "train_combined.yaml"
train_comb_yaml.write_text(f\"\"\"train: {comb_tr_imgs}
val: {comb_va_imgs}
nc: 2
names: ['no_helmet', 'helmet']
\"\"\")

print("YAML configurations created successfully!")
"""),
    cell(eval_helper_code),
    cell("""# 4. Execute Multi-Seed EXP-COMBINED Training (Seeds 42, 123, 2026)
SEEDS = [42, 123, 2026]
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
    eval_new = evaluate_checkpoint(best_weights, eval_new_yaml, f"New Test (Seed {seed})", run_dir / "eval_new")
    eval_old = evaluate_checkpoint(best_weights, eval_old_yaml, f"Old CCTV Test (Seed {seed})", run_dir / "eval_old")
    eval_comb = evaluate_checkpoint(best_weights, eval_comb_yaml, f"Combined Test (Seed {seed})", run_dir / "eval_comb")
    
    summary = {
        "experiment": "EXP-COMBINED",
        "seed": seed,
        "training_duration_seconds": dur,
        "evaluations": {
            "new_test": eval_new,
            "old_test": eval_old,
            "combined_test": eval_comb
        }
    }
    
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    results_by_seed[seed] = summary

print("All EXP-COMBINED runs completed successfully!")
(work_dir / "exp_combined_multiseed_summary.json").write_text(json.dumps(results_by_seed, indent=2))
""")
]

nb_exp_comb = create_notebook(exp_comb_cells)
Path("notebooks/phase9_multiseed/kaggle_phase9_exp_combined.ipynb").write_text(json.dumps(nb_exp_comb, indent=2))
print("Saved kaggle_phase9_exp_combined.ipynb!")
