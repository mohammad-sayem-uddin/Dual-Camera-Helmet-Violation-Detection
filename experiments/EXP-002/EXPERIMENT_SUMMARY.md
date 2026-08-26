# EXP-002 Experiment Summary

**Experiment ID**: `EXP-002`  
**Description**: YOLOv8 nano detector trained on the sequence-isolated group split (measuring true out-of-sequence generalization).  
**Execution Date**: August 26, 2026  
**Status**: Completed & Evaluated (Model Frozen)  
**Checkpoint Checksum (SHA256)**: `21b17a014938fff69b581a7de0d1da5fdc2f1c2ce8f7eb08e15a61019214cdd5`  

---

## 1. Experiment Specification

* **Model Architecture**: `yolov8n.pt` (Pretrained backbone, 3,006,038 parameters)
* **Dataset Config**: `configs/dataset_group_split.yaml` (`data/processed/v1_group_split/`)
* **Split Characteristics**:
  * Train: 3,873 images across 187 unique video sequences
  * Validation: 837 images across 44 unique video sequences
  * Test: 837 images across 44 unique video sequences
  * *Note*: Strict **0.0% sequence overlap** across all splits.
* **Hyperparameters**:
  * Image Resolution: 640×640
  * Batch Size: 16
  * Epochs: 50
  * Initial Optimizer: AdamW (`lr0: 0.001667`, `momentum: 0.9`, `weight_decay: 0.0005`)
  * Early Stopping: `patience: 25` (Completed full 50 epochs)
  * Random Seed: 42 (Deterministic execution on MPS)
  * Augmentation Policy: Identical to EXP-001 (`fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1`, `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4`).

---

## 2. Training Execution & Convergence

* **Training Hardware**: Apple Silicon MPS (`mps:0`, Apple M1)
* **Total Training Duration**: 23,610.26s ($\approx 6.56\text{ hours}$)
* **Total Epochs Completed**: 50 / 50
* **Best Validation Epoch**: **Epoch 35** (Selection criterion: maximum validation $\text{mAP}@0.50 = 79.96\%$)
* **Training Dynamics**:
  * Train Box Loss: Decreased from `2.1948` (Epoch 1) to `1.3438` (Epoch 50)
  * Train Class Loss: Decreased from `3.3095` (Epoch 1) to `0.6354` (Epoch 50)
  * Validation Box Loss: Reached minimum of `1.7609` at Epoch 23
  * Validation Class Loss: Reached minimum of `0.9232` at Epoch 49
  * Overfitting Assessment: Validation metrics remained stable throughout epochs 20–50 without divergence.

---

## 3. Standardized Validation Evaluation (`EXP-002-val`)

*Evaluated on frozen `best.pt` checkpoint across all 837 validation images (44 sequence-isolated videos).*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **837** | **1,341** | **0.8034** | **0.7655** | **0.7840** | **0.7274** | **0.3282** |
| `no_helmet` (0) | 528 | 745 | 0.7886 | 0.7762 | 0.7823 | 0.7307 | 0.3437 |
| `helmet` (1) | 328 | 596 | 0.8182 | 0.7549 | 0.7852 | 0.7241 | 0.3127 |

---

## 4. Frozen Final Test Evaluation (`EXP-002-test`)

*Evaluated once on frozen `best.pt` checkpoint across all 837 unseen test images (44 completely unseen sequence-isolated videos).*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **837** | **1,115** | **0.7325** | **0.7136** | **0.7229** | **0.6444** | **0.2923** |
| `no_helmet` (0) | 554 | 717 | 0.7410 | 0.7420 | 0.7415 | 0.6586 | 0.2827 |
| `helmet` (1) | 370 | 398 | 0.7239 | 0.6852 | 0.7040 | 0.6303 | 0.3020 |

---

## 5. Observations & Scientific Context

1. **Generalization Under Video Isolation**: When evaluated on strictly unseen video sequences, the model retains strong operational detection capability ($72.29\%$ F1, $64.44\%$ mAP@50).
2. **Generalization Gap**: The difference between EXP-001 ($83.42\%$ mAP50) and EXP-002 ($64.44\%$ mAP50) provides direct empirical evidence of the temporal correlation bias inherent in random frame-level splitting.
