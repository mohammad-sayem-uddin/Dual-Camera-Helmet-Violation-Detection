# EXP-003 Experiment Summary: Size-Matched Random-Frame Control

**Experiment ID**: `EXP-003`  
**Description**: YOLOv8 nano detector trained on random-frame split with size-matched training set (3,866 images) to isolate the data quantity confound.  
**Execution Date**: August 26, 2026  
**Status**: Completed & Evaluated (Model Frozen)  
**Checkpoint Checksum (SHA256)**: `dc4d4b50f9e66d44415f3b3106767bb53198cdddfe9fcb95d720e19363980822`  

---

## 1. Experiment Specification

* **Model Architecture**: `yolov8n.pt` (Pretrained backbone, 3,006,038 parameters)
* **Dataset Config**: `configs/dataset_random_matched.yaml` (`data/processed/v1_random_matched/`)
* **Split Characteristics**:
  * **Train**: **3,866 images** (subsampled deterministically with seed=42 from EXP-001's 4,854 train images to match EXP-002-FIXED)
  * **Validation**: 347 images
  * **Test**: 346 images
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
* **Total Training Duration**: 22,151.58s ($\approx 6.15\text{ hours}$)
* **Total Epochs Completed**: 50 / 50
* **Best Validation Epoch**: **Epoch 46** (Selection criterion: maximum validation $\text{mAP}@0.50 = 86.48\%$)
* **Training Dynamics**:
  * Train Box Loss: Decreased from `2.1546` (Epoch 1) to `1.3720` (Epoch 50)
  * Train Class Loss: Decreased from `3.3180` (Epoch 1) to `0.6378` (Epoch 50)
  * Validation Box Loss: Reached minimum of `1.8310` at Epoch 29
  * Validation Class Loss: Reached minimum of `0.8469` at Epoch 46

---

## 3. Standardized Validation Evaluation (`EXP-003-val`)

*Evaluated on frozen `best.pt` checkpoint across all 347 validation images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **347** | **524** | **0.8132** | **0.8316** | **0.8223** | **0.7891** | **0.3644** |
| `no_helmet` (0) | 223 | 308 | 0.8365 | 0.7974 | 0.8165 | 0.7890 | 0.3747 |
| `helmet` (1) | 157 | 216 | 0.7899 | 0.8657 | 0.8261 | 0.7893 | 0.3541 |

---

## 4. Frozen Final Test Evaluation (`EXP-003-test`)

*Evaluated once on frozen `best.pt` checkpoint across all 346 unseen test images.*

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score | $\text{mAP}@0.50$ | $\text{mAP}@0.50:0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **346** | **521** | **0.8518** | **0.8464** | **0.8491** | **0.8247** | **0.3963** |
| `no_helmet` (0) | 232 | 326 | 0.8332 | 0.8160 | 0.8245 | 0.8041 | 0.3934 |
| `helmet` (1) | 153 | 195 | 0.8703 | 0.8769 | 0.8736 | 0.8454 | 0.3992 |

---

## 5. Controlled Baseline Comparison: EXP-001 vs. EXP-003

| Metric | EXP-001 (4,854 Train Imgs) | EXP-003 (3,866 Train Imgs) | $\Delta_{\text{size}}$ (Data Quantity Effect) |
| :--- | :---: | :---: | :---: |
| **Test Precision** | 0.8688 | 0.8518 | -0.0170 (-1.96%) |
| **Test Recall** | 0.8566 | 0.8464 | -0.0102 (-1.19%) |
| **Test F1-Score** | 0.8626 | 0.8491 | -0.0135 (-1.57%) |
| **Test $\text{mAP}@0.50$** | **0.8342** | **0.8247** | **-0.0095 (-1.14%)** |
| **Test $\text{mAP}@0.50:0.95$** | 0.4135 | 0.3963 | -0.0172 (-4.16%) |

**Key Finding**: Reducing training dataset size by 20.3% under random frame splitting produces only a negligible **0.95 percentage point drop** in test mAP@50 ($\Delta_{\text{size}} = -0.0095$). This empirically isolates the training size confound.
