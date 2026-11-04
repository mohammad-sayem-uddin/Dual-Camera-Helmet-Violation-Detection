# EXP-002-FIXED Experiment Summary: Corrected Sequence-Exclusive Group Split

**Experiment ID**: `EXP-002-FIXED`  
**Description**: YOLOv8 nano detector trained on true sequence-level group split (169 video groups, 0.0% sequence overlap across train/val/test, 3,866 train images).  
**Execution Date**: August 26–27, 2026  
**Status**: Completed & Evaluated (Model Frozen)  
**Checkpoint Checksum (SHA256)**: `2388f2248aaf5ab4dae49d26a9552203c1c6fd43bbfe0ad1264fadca975978b8`  

---

## 1. Experiment Specification

* **Model Architecture**: `yolov8n.pt` (Pretrained backbone, 3,006,038 parameters)
* **Dataset Config**: `configs/dataset_group_split_fixed.yaml` (`data/processed/v1_group_split_fixed/`)
* **Split Characteristics**:
  * **Train**: **3,866 images** (118 video sequences)
  * **Validation**: **852 images** (26 video sequences)
  * **Test**: **829 images** (25 video sequences)
  * **Total Sequences**: 169 true video groups, strictly sequence-exclusive (0.0% overlap mathematically verified)
* **Hyperparameters**:
  * Image Resolution: 640×640
  * Batch Size: 16
  * Epochs: 50
  * Initial Optimizer: AdamW (`lr0: 0.001667`, `momentum: 0.9`, `weight_decay: 0.0005`)
  * Early Stopping: `patience: 25` (Completed full 50 epochs)
  * Random Seed: 42 (Deterministic execution on Apple Silicon MPS)
  * Augmentation Policy: `fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1`, `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4`

---

## 2. Training Execution & Convergence

* **Training Hardware**: Apple Silicon MPS (`mps:0`, Apple M1)
* **Total Training Duration**: 21,589.69s ($\approx 6.0\text{ hours}$)
* **Total Epochs Completed**: 50 / 50
* **Best Validation Epoch**: **Epoch 40** (Selection criterion: maximum validation $\text{mAP}@0.50 = 85.23\%$)
* **Training Dynamics**:
  * Train Box Loss: Decreased from `2.0620` (Epoch 1) to `1.3418` (Epoch 50)
  * Train Class Loss: Decreased from `3.1583` (Epoch 1) to `0.6384` (Epoch 50)
  * Validation Box Loss: Reached minimum of `1.9772` at Epoch 32
  * Validation Class Loss: Reached minimum of `0.9416` at Epoch 40

---

## 3. Standardized Validation Evaluation (`EXP-002-FIXED-val`)

*Evaluated on frozen `best.pt` checkpoint across all 852 sequence-isolated validation images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **852** | **1264** | **0.8417** | **0.8124** | **0.8268** | **0.7796** | **0.3410** |
| `no_helmet` (0) | 496 | 728 | 0.8099 | 0.7294 | 0.7675 | 0.6919 | 0.3070 |
| `helmet` (1) | 437 | 536 | 0.8736 | 0.8953 | 0.8843 | 0.8673 | 0.3750 |

---

## 4. Frozen Final Test Evaluation (`EXP-002-FIXED-test`)

*Evaluated once on frozen `best.pt` checkpoint across all 829 sequence-isolated test images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **829** | **1170** | **0.7431** | **0.7228** | **0.7328** | **0.6694** | **0.2896** |
| `no_helmet` (0) | 400 | 600 | 0.7970 | 0.7001 | 0.7454 | 0.6791 | 0.2861 |
| `helmet` (1) | 471 | 570 | 0.6891 | 0.7456 | 0.7163 | 0.6598 | 0.2930 |

---

## 5. Controlled Baseline Comparison: EXP-003 vs. EXP-002-FIXED

*Both models trained on exactly 3,866 images with identical hyperparameters, architecture, and seed=42.*

| Metric | EXP-003 (Random Frame Split) | EXP-002-FIXED (Sequence Group Split) | $\Delta_{\text{sequence}}$ (Sequence Isolation Effect) |
| :--- | :---: | :---: | :---: |
| **Test Precision** | 0.8518 | 0.7431 | -0.1087 (-12.76%) |
| **Test Recall** | 0.8464 | 0.7228 | -0.1236 (-14.60%) |
| **Test F1-Score** | 0.8491 | 0.7328 | -0.1163 (-13.70%) |
| **Test $\text{mAP}@0.50$** | **0.8247** | **0.6694** | **-0.1553 (-18.83%)** |
| **Test $\text{mAP}@0.50:0.95$** | 0.3963 | 0.2896 | -0.1067 (-26.92%) |

**Key Finding**: When controlling exactly for training set size (3,866 images), evaluating on true sequence-isolated video footage produces a large and substantial **15.53 percentage point drop in test mAP@50** ($\Delta_{\text{sequence}} = -0.1553$). This provides pilot evidence that sequence leakage in random-frame evaluation creates severe artificial performance inflation.
