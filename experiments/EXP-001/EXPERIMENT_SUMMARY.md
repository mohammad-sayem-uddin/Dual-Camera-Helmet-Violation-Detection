# EXP-001 Experiment Summary

**Experiment ID**: `EXP-001`  
**Description**: YOLOv8 nano baseline detector on the conventional Roboflow frame-level dataset.  
**Execution Date**: August 25–26, 2026  
**Status**: Completed & Evaluated (Model Frozen)  
**Checkpoint Checksum (SHA256)**: `a8610ee01e72254d11626054dae376d661357160aa42f2058c4697bc56f6c1e3`  

---

## 1. Experiment Specification

* **Model Architecture**: `yolov8n.pt` (Pretrained backbone, 3,006,038 parameters)
* **Dataset Config**: `configs/dataset.yaml` (`data/annotated/kawke_dibona_v1/`)
* **Split Characteristics**:
  * Train: 4,854 images (3× Roboflow offline augmentations on 1,618 base frames)
  * Validation: 347 images
  * Test: 346 images
  * *Note*: Contains temporal video sequence overlap between splits (114/133 sequences in valid and 111/131 in test overlap with train).
* **Hyperparameters**:
  * Image Resolution: 640×640
  * Batch Size: 16
  * Epochs: 50
  * Initial Optimizer: AdamW (`lr0: 0.001667`, `momentum: 0.9`, `weight_decay: 0.0005`)
  * Early Stopping: `patience: 25` (Completed full 50 epochs)
  * Random Seed: 42 (Deterministic execution on MPS)
  * Augmentation Policy: `fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1`, `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4` (mixup, shear, degrees disabled).

---

## 2. Training Execution & Convergence

* **Training Hardware**: Apple Silicon MPS (`mps:0`, Apple M1)
* **Total Training Duration**: 26,917.32s ($\approx 7.48\text{ hours}$)
* **Total Epochs Completed**: 50 / 50
* **Best Validation Epoch**: **Epoch 40** (Selection criterion: maximum validation $\text{mAP}@0.50 = 86.31\%$)
* **Training Dynamics**:
  * Train Box Loss: Decreased from `2.1524` (Epoch 1) to `1.2829` (Epoch 50)
  * Train Class Loss: Decreased from `3.1231` (Epoch 1) to `0.4578` (Epoch 50)
  * Validation Box Loss: Reached minimum of `1.6441` at Epoch 48
  * Validation Class Loss: Reached minimum of `0.6558` at Epoch 48
  * Overfitting Assessment: No severe overfitting observed; validation loss tracks training loss steadily.

---

## 3. Standardized Validation Evaluation (`EXP-001-val`)

*Evaluated on frozen `best.pt` checkpoint across all 347 validation images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **347** | **524** | **0.8356** | **0.8763** | **0.8554** | **0.8161** | **0.3673** |
| `no_helmet` (0) | 223 | 308 | 0.8459 | 0.8344 | 0.8401 | 0.7990 | 0.3736 |
| `helmet` (1) | 157 | 216 | 0.8252 | 0.9181 | 0.8692 | 0.8333 | 0.3610 |

---

## 4. Frozen Final Test Evaluation (`EXP-001-test`)

*Evaluated once on frozen `best.pt` checkpoint across all 346 unseen test images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **346** | **521** | **0.8688** | **0.8566** | **0.8626** | **0.8342** | **0.4135** |
| `no_helmet` (0) | 232 | 326 | 0.8591 | 0.8413 | 0.8501 | 0.8177 | 0.4015 |
| `helmet` (1) | 153 | 195 | 0.8785 | 0.8718 | 0.8752 | 0.8508 | 0.4255 |

---

## 5. Observations & Scientific Context

1. **Detection Performance**: The model achieves high recall ($87.18\%$) on helmets and balanced detection ($85.01\%$ F1) on violators (`no_helmet`).
2. **Roboflow Split Confound**: Because 85.7% of validation sequences and 84.7% of test sequences share video origins with training frames, these performance metrics may reflect temporal autocorrelation. This hypothesis will be rigorously tested in **EXP-002** using the sequence-isolated dataset.
