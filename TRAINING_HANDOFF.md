# TrafficVision-AI: Motorcycle Helmet Detection Research

## Research Readiness & Training Handoff Report

**Project**: Context-Aware Motorcycle Helmet Violation Detection Research  
**Repository**: `research_helmet/`  
**Status**: Ready for Baseline Training (`EXP-001`)  
**Date**: August 25, 2026  

---

## 1. Final Repository Structure

The repository maintains a clean, decoupled research architecture with zero web application or database dependencies:

```text
research_helmet/
├── configs/
│   ├── dataset.yaml                               # Active dataset configuration for YOLOv8
│   └── experiments/
│       ├── baseline_template.yaml                 # Generic experiment template
│       └── exp_001_baseline.yaml                  # EXP-001 YOLOv8n baseline configuration
├── data/
│   ├── annotated/
│   │   └── kawke_dibona_v1/                       # Integrated Roboflow annotated dataset
│   │       ├── data.yaml                          # Dataset specification
│   │       ├── README.roboflow.txt                # Export provenance
│   │       ├── train/ (images: 4854, labels: 4854)
│   │       ├── valid/ (images: 347, labels: 347)
│   │       └── test/  (images: 346, labels: 346)
│   ├── metadata/
│   │   ├── README.md
│   │   └── source_groups.csv                      # Video sequence group mapping (5547 rows)
│   ├── processed/README.md                        # Processed dataset derivatives
│   ├── raw/README.md                              # Raw source videos/images
│   └── splits/
│       ├── README.md
│       └── v1_group_split.csv                     # Leakage-free sequence-level split manifest
├── experiments/                                   # Destination for versioned run directories (EXP-###)
├── models/
│   ├── checkpoints/                               # Mirrored best.pt and last.pt checkpoints
│   └── final/                                     # Publication-ready model weights
├── research_notes/
│   └── research_note.txt                          # Scientific objectives & theoretical scope
├── results/
│   ├── confusion_matrices/                        # Normalized & raw confusion matrix plots
│   ├── metrics/                                   # Machine-readable JSON audit & evaluation logs
│   │   ├── dataset_audit.json                     # Comprehensive dataset audit metrics
│   │   └── duplicate_report.json                  # SHA-256 duplicate & leakage metrics
│   ├── plots/                                     # Multi-panel loss & PR curve visualizations
│   ├── predictions/                               # Visual qualitative inference outputs
│   └── tables/                                    # Exported metric history tables
├── scripts/
│   ├── analysis/
│   │   ├── analyze_training.py                    # Training curve extractor & overfitting analyzer
│   │   └── visualize_predictions.py               # Qualitative visual prediction generator
│   ├── data/
│   │   ├── check_duplicates.py                    # Duplicate detection utility
│   │   ├── create_group_split.py                  # Group-exclusive split manifest generator
│   │   ├── inspect_dataset.py                     # Dataset validator
│   │   └── run_audit.py                           # Full automated audit runner
│   ├── evaluation/
│   │   └── evaluate_yolo.py                       # Standardized evaluation pipeline
│   └── training/
│       └── train_yolo.py                          # Config-driven YOLO training runner
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── yolo_dataset.py                        # Dataset inspection & validation core library
│   ├── evaluation/
│   │   └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── paths.py                               # Workspace path resolution helper
│       └── yaml_loader.py                         # Robust YAML parser with pure-Python fallback
├── README.md                                      # Workspace guide
├── requirements.txt                               # Research Python dependencies
└── TRAINING_HANDOFF.md                            # [THIS DOCUMENT]
```

---

## 2. Dataset Location & Structure

* **Dataset Path**: `data/annotated/kawke_dibona_v1/`
* **Dataset Config**: `configs/dataset.yaml`
* **Format**: YOLOv8 PyTorch format (`train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`).
* **Coordinate Space**: Normalized bounding boxes `[class_id, x_center, y_center, width, height]` with all values in $[0.0, 1.0]$.
* **Image Format**: 100% 640×640 24-bit RGB JPEGs.

---

## 3. Dataset Classes

| Class ID | Class Name | Description | Bounding Box Scope |
| :---: | :---: | :--- | :--- |
| **`0`** | `no_helmet` | Motorcycle rider/passenger without a protective helmet | Head region (hair, bare head, cap, turban, mask) |
| **`1`** | `helmet` | Motorcycle rider/passenger wearing a helmet | Outer boundary of full-face, open-face, or half helmet |

---

## 4. Dataset Audit Quantitative Summary

*Extracted directly from the real dataset via `scripts/data/run_audit.py` and saved to `results/metrics/dataset_audit.json`.*

| Split | Total Images | Label Files | Annotated Images | Empty (Negative) Images | Total Bounding Boxes | Class 0 (`no_helmet`) | Class 1 (`helmet`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,854 | 4,854 | 4,848 | 6 | 7,203 | 4,254 (59.1%) | 2,949 (40.9%) |
| **Valid** | 347 | 347 | 347 | 0 | 524 | 308 (58.8%) | 216 (41.2%) |
| **Test** | 346 | 346 | 345 | 1 | 521 | 326 (62.6%) | 195 (37.4%) |
| **Total** | **5,547** | **5,547** | **5,540** | **7** | **8,248** | **4,888 (59.3%)** | **3,360 (40.7%)** |

---

## 5. Data Quality Assurance Findings

1. **Image Readability**: 5,547 / 5,547 images are valid and fully readable JPEGs. **0 corrupted files**.
2. **Annotation Completeness**: Every image has an exact matching `.txt` label file. **0 missing pairs** and **0 orphan labels**.
3. **Class ID Integrity**: 100% of bounding box classes are strictly `0` or `1`. **0 invalid class IDs**.
4. **Bounding Box Geometry**: All 8,248 bounding box coordinates are within normalized $[0, 1]$ bounds with positive width and height. **0 out-of-bound boxes** and **0 malformed lines**.
5. **Exact Duplicates**: SHA-256 hash comparison across all 5,547 files shows **0 exact duplicate image files**.
6. **Negative Background Samples**: Exactly 7 images contain empty label files (6 in train, 1 in test), providing valid negative background examples that help suppress false positives.

---

## 6. Sequence Grouping & Data Leakage Analysis

> [!WARNING]
> **Data Leakage in Default Roboflow Split**:
> The default Roboflow export randomly partitioned individual frames rather than grouping by video sequence.

* **Unique Base Frames**: 2,311 source frames (1,618 in train with 3× offline augmentations = 4,854 images; 347 in valid; 346 in test).
* **Identified Video Sequences**: 275 distinct video sequences (identified by timestamp prefixes `YYYYMMDD_HHMMSS` and sequence IDs `frame_XX_YYY`).
* **Cross-Split Sequence Overlap**:
  * **114 out of 133 sequences** in `valid` are ALSO present in `train`.
  * **111 out of 131 sequences** in `test` are ALSO present in `train`.

### Scientific Implication
While no exact image duplicate exists between splits, adjacent frames from the same motorcycle passes appear in both train and validation/test.
* For **EXP-001 (Baseline)**: We train on the standard Roboflow split to establish a benchmark baseline comparable to public benchmarks.
* For **EXP-002+ (Leakage-Free Comparison)**: Use the group-exclusive split generated in `data/splits/v1_group_split.csv` to measure true generalization to unseen camera sequences.

---

## 7. Preprocessing & Augmentation Strategy

### YOLOv8 Native Preprocessing
* **Resolution**: Standardized 640×640.
* **Aspect Ratio**: Letterbox padding with reflection/constant gray border to preserve natural aspect ratios without distortion.
* **Color Space**: BGR to RGB normalization with floating-point scaling ($[0, 255] \to [0.0, 1.0]$).

### Baseline Augmentation Policy (`EXP-001`)
* **Horizontal Flip (`fliplr: 0.5`)**: Enabled (traffic flow has left/right symmetry).
* **Vertical Flip (`flipud: 0.0`)**: Disabled (motorcycles and riders are strictly upright).
* **Color Jitter (`hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4`)**: Enabled to handle diverse daytime, glare, and shadow conditions.
* **Scale Jitter (`scale: 0.5`) & Translation (`translate: 0.1`)**: Enabled to handle varying camera distances and rider positions.
* **Mosaic (`mosaic: 1.0`)**: Enabled to teach multi-scale object detection.
* **MixUp (`mixup: 0.0`)**: Disabled (helmets are small targets; blending images washes out critical helmet boundary features).
* **Rotation (`degrees: 0.0`)**: Disabled (traffic cameras maintain fixed horizon levels).

---

## 8. EXP-001 Baseline Experiment Specification

* **Experiment ID**: `EXP-001`
* **Configuration File**: `configs/experiments/exp_001_baseline.yaml`
* **Model**: `yolov8n.pt` (Nano architecture: ~3.2M parameters)
* **Pretrained Weights**: ImageNet / COCO transfer learning backbone
* **Batch Size**: 16
* **Epochs**: 50
* **Patience**: 25 (Early stopping)
* **Seed**: 42 (Fully deterministic)
* **Device**: `auto` (Selects Apple Silicon MPS, CUDA GPU, or CPU)

---

## 9. Standard Execution Commands

### Step 1: Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install research requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Quality Control & Leakage Audit
```bash
# Run automated dataset audit & duplicate scan
python scripts/data/run_audit.py
```

### Step 3: Run Baseline Training (`EXP-001`)
```bash
# Optional dry-run check
python scripts/training/train_yolo.py --config configs/experiments/exp_001_baseline.yaml --dry-run

# Start baseline training
python scripts/training/train_yolo.py --config configs/experiments/exp_001_baseline.yaml
```

### Step 4: Standardized Evaluation
```bash
# Evaluate best checkpoint on VALIDATION split (for model selection)
python scripts/evaluation/evaluate_yolo.py \
  --weights models/checkpoints/EXP-001/best.pt \
  --data configs/dataset.yaml \
  --split val \
  --name EXP-001-val

# Evaluate best checkpoint on TEST split (RESERVED FOR FINAL RESULTS ONLY)
python scripts/evaluation/evaluate_yolo.py \
  --weights models/checkpoints/EXP-001/best.pt \
  --data configs/dataset.yaml \
  --split test \
  --name EXP-001-test
```

### Step 5: Training Analysis & Publication Curves
```bash
# Extract metric tables, best epoch, and multi-panel training curves
python scripts/analysis/analyze_training.py \
  --results-csv experiments/EXP-001/training/results.csv \
  --experiment-id EXP-001

# Generate qualitative visual prediction examples
python scripts/analysis/visualize_predictions.py \
  --weights models/checkpoints/EXP-001/best.pt \
  --source data/annotated/kawke_dibona_v1/test/images \
  --name EXP-001-test-preds \
  --max-images 50
```

---

## 10. Rules for Scientific Integrity

1. **Never Tune on the Test Set**: Hyperparameters, architecture choices, and confidence thresholds must only be tuned using the validation set (`val`). The test set (`test`) must remain strictly unseen until final publication evaluation.
2. **Never Modify Existing Experiment Runs**: Once an experiment ID (e.g. `EXP-001`) is executed, never overwrite or modify its directory. Create `EXP-002`, `EXP-003`, etc. with an explicit changelog in the config.
3. **Preserve Raw & Annotated Data**: Never delete or overwrite raw or annotated images. Any preprocessing must be stored under `data/processed/` with documented deterministic scripts.
4. **Log Hardware & Seeds**: Always record random seeds and device configurations (MPS / CUDA) to ensure exact reproducibility across machines.
