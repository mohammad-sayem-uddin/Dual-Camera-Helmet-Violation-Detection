# Phase 5 Verification & Pre-Training Readiness Report
**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Experiment Lead:** Senior Computer Vision Research Engineer & Kaggle Execution Agent  
**Date:** November 4, 2026  
**Status:** **PHASE 5 COMPLETE — AWAITING USER APPROVAL TO START FULL TRAINING**

---

## 1. Executive Summary

Phase 5 has been executed and empirically verified on Kaggle's cloud infrastructure. Both research datasets were formatted, packaged, uploaded as private Kaggle datasets, and validated. A dedicated 2-epoch end-to-end smoke test kernel was executed on an NVIDIA Tesla T4 GPU (`CUDA:0`), confirming CUDA device visibility, correct dataset mounting, gradient backpropagation, checkpoint serialization, and artifact retrieval.

Full 50-epoch training of `EXP-NEW` and `EXP-COMBINED` remains strictly halted pending formal user approval.

---

## 2. Mandatory Verification Checklist

### 2.1 Kaggle Datasets Upload & Status Confirmation
Both datasets are uploaded as private datasets under the authenticated Kaggle account `allfaz` (User ID: `18614709`) and have achieved `Ready` status:
1. **New Expansion Dataset**: `allfaz/helmet-dataset-expansion-v2`
   - **Upload Status**: `Ready`
   - **Archive Size**: 1.93 GB
   - **Total Images**: 1,841 (Train: 1,285 | Val: 276 | Test: 280)
   - **Total Label Files**: 1,841 `.txt` files + 1 `data.yaml` = 1,842 files
   - **Total Ground-Truth Instances**: 3,733 tightly audited rider head boxes (2,026 `helmet`, 1,707 `no_helmet`)
2. **Old Fixed Baseline Dataset**: `allfaz/old-dataset-v1-fixed`
   - **Upload Status**: `Ready`
   - **Archive Size**: 428.6 MB
   - **Total Images**: 5,547 (Train: 3,866 | Val: 852 | Test: 829)
   - **Total Label Files**: 5,547 `.txt` files + 1 `data.yaml` = 5,548 files
   - **Total Ground-Truth Instances**: 8,248 boxes (3,360 `helmet`, 4,888 `no_helmet`)

---

### 2.2 Notebook Dataset Mounting & Environment Verification
The smoke test kernel (`allfaz/helmet-detection-smoke-test`, Version 3) executed on Kaggle GPU and dynamically traversed `/kaggle/input` to confirm filesystem mounting:
- **New Dataset Mount**: `/kaggle/input/datasets/allfaz/helmet-dataset-expansion-v2`
  - Verified paths: `images/train` (1,285 imgs), `images/val` (276 imgs), `images/test` (280 imgs)
- **Old Dataset Mount**: `/kaggle/input/datasets/allfaz/old-dataset-v1-fixed`
  - Verified paths: `images/train` (3,866 imgs), `images/val` (852 imgs), `images/test` (829 imgs)
- **Software Runtime**:
  - Python: `3.13.16`
  - PyTorch: `2.11.0+cu130`
  - Ultralytics YOLOv8: `8.4.202`
  - CUDA: `13.0`

---

### 2.3 Hardware & GPU VRAM Audit
- **GPU Model Detected**: **NVIDIA Tesla T4**
- **GPU Device Count**: 2 physical devices attached to container
- **Active Device**: `CUDA:0 (Tesla T4, 14912MiB)`
- **Available Dedicated VRAM**: **14.56 GB** (14,912 MiB)
- **Peak Memory Allocated during Training**: 5.0 GB (33.5% of available VRAM at batch size 32)

---

### 2.4 Smoke Test Empirical Runtime & Convergence Dynamics
The smoke test executed a 2-epoch run of YOLOv8n (`yolov8n.pt`, 3.13M parameters, 8.6 GFLOPs) on the new expansion dataset:
- **Total Wall-Clock Run Duration**: **260.31 seconds (4.34 minutes)**
- **Epoch 1 Execution**:
  - Initial label scanning (276 val labels): 15.2 seconds
  - Batch training (41 batches, 1,285 images): ~105.8 seconds (cold cache / worker setup)
  - Train losses: `box_loss: 1.4007`, `cls_loss: 4.8758`, `dfl_loss: 1.0389`
  - Val metrics: Precision = 0.363, Recall = 0.650, mAP50 = 0.334, mAP50-95 = 0.226
- **Epoch 2 Execution**:
  - Pure batch training (41 batches @ 2.5 it/s): **16.2 seconds**
  - Validation pass (5 batches on 276 images): **1.6 seconds**
  - Train losses: `box_loss: 1.0772`, `cls_loss: 0.7632`, `dfl_loss: 0.9191`
  - Val metrics (post-epoch): Precision = 0.909, Recall = 0.870, mAP50 = 0.911, mAP50-95 = 0.640
- **Final Validation on Best Checkpoint (276 val images, 597 instances)**:
  - `all`: Precision = 0.910, Recall = 0.870, **mAP50 = 0.911**, **mAP50-95 = 0.640**
  - `no_helmet` (203 instances): Precision = 0.852, Recall = 0.828, **mAP50 = 0.845**, **mAP50-95 = 0.554**
  - `helmet` (394 instances): Precision = 0.968, Recall = 0.911, **mAP50 = 0.977**, **mAP50-95 = 0.726**
  - Per-image latency: 0.4ms preprocess, 3.8ms inference, 1.9ms postprocess (5.7ms total, ~175 FPS on T4).

---

### 2.5 Checkpoint Serialization & Artifact Verification
All training checkpoints and evaluation artifacts were generated on Kaggle and downloaded locally to `notebooks/smoke_test/output/`:
- `best.pt`: 6,500,778 bytes (6.20 MB) — Validated PyTorch state dict.
- `last.pt`: 6,500,778 bytes (6.20 MB) — Validated PyTorch state dict.
- `results.csv`: Complete per-epoch metric log.
- Visual curves: `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`.
- Output log: `helmet-detection-smoke-test.log` (167 KB).

---

### 2.6 Estimated Runtime and GPU Quota Consumption for Full Experiments

| Experiment | Training Dataset Size | Batches / Epoch (bs=32) | Empirical Sec / Epoch | Max Epochs (Patience=20) | Est. Training Time | Multi-Population Eval (3 Test Sets) | Total Run Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-NEW** | 1,285 images | 41 | ~18–22 s | 50 | ~15–18 min | ~5 min | **~20–23 min (~0.35 hrs)** |
| **EXP-COMBINED** | 5,151 images | 161 | ~65–75 s | 50 | ~55–62 min | ~5 min | **~60–67 min (~1.05 hrs)** |
| **Total** | — | — | — | — | — | — | **~80–90 min (~1.40 hrs)** |

#### Kaggle GPU Quota Balance
- **Total Weekly GPU Allowance**: 108,000.00 seconds (30.00 hours)
- **Time Consumed to Date** (including smoke test & setup): **368.64 seconds (0.10 hours)**
- **Quota Remaining**: **107,631.36 seconds (29.90 hours, 99.66% remaining)**
- **Projected Quota Remaining After Full Runs**: **~28.5 hours (95.0% of weekly allowance preserved)**

---

### 2.7 Controlled Experiment Configuration (EXP-NEW vs. EXP-COMBINED)

To eliminate confounding variables and ensure clean causal attribution, both experiments share an identical hyperparameter control harness:

| Parameter | EXP-NEW | EXP-COMBINED | Control Rationale |
| :--- | :--- | :--- | :--- |
| **Base Architecture** | YOLOv8n (`yolov8n.pt`) | YOLOv8n (`yolov8n.pt`) | Constant parameter capacity (3.13M params) |
| **Input Resolution** | $640 \times 640$ | $640 \times 640$ | Equal spatial sampling density |
| **Batch Size** | 32 | 32 | Equal gradient accumulation dynamics |
| **Optimizer** | AdamW | AdamW | Robust adaptive gradient optimization |
| **Initial Learning Rate** (`lr0`) | 0.001667 | 0.001667 | Standardized learning rate |
| **Momentum / Weight Decay** | 0.9 / 0.0005 | 0.9 / 0.0005 | Identical regularization penalty |
| **Random Seed** | 42 (Deterministic) | 42 (Deterministic) | Eliminates weight initialization variance |
| **Epochs / Early Stopping** | 50 / Patience 20 | 50 / Patience 20 | Identical convergence budget |
| **Augmentation Policy** | Default YOLOv8 (mosaic 1.0, fliplr 0.5, hsv) | Default YOLOv8 (mosaic 1.0, fliplr 0.5, hsv) | Equal augmentation domain |
| **Evaluation Suites** | 1. New Test (280 imgs)<br>2. Old Test (829 imgs)<br>3. Combined Test (1,109 imgs) | 1. New Test (280 imgs)<br>2. Old Test (829 imgs)<br>3. Combined Test (1,109 imgs) | Direct cross-dataset generalization comparison |

---

### 2.8 Global Data Leakage Audit Certification
- **Sequence Isolation Guarantee**:
  - Old dataset contains 169 discrete video capture sessions.
  - New expansion dataset contains 565 discrete video sequences.
  - Across all 734 total sequence groups, the intersection is empty: $169 \cap 565 = \emptyset$.
  - Train, validation, and test splits within each dataset and across the combined dataset are partitioned exclusively at the group level:
    - Old Split: 118 train groups, 26 val groups, 25 test groups (strictly disjoint).
    - New Split: 438 train groups, 74 val groups, 53 test groups (strictly disjoint).
    - Combined Split: 556 train groups, 100 val groups, 78 test groups (strictly disjoint).
- **Exact File Collision Audit**:
  - SHA-256 cryptographic hashes evaluated across all 7,388 images: **0 exact collisions**.

---

### 2.9 Mathematical Formulation and Critical Limitations of the dHash Near-Duplicate Check

To satisfy rigorous scientific reporting standards, the properties and limitations of the Difference Hash (`dHash`) perceptual duplicate audit are formally documented:

#### Mathematical Mechanism
1. Each image $I(x, y)$ is converted to 8-bit grayscale luminance $Y$ and rescaled via bilinear interpolation to a small $9 \times 8$ pixel matrix:
   $$\tilde{Y} \in \mathbb{R}^{9 \times 8}$$
2. The horizontal gradient $\Delta_x$ is computed between adjacent columns:
   $$D(i, j) = \tilde{Y}(i + 1, j) - \tilde{Y}(i, j), \quad \text{for } i \in [0, 7], \, j \in [0, 7]$$
3. A 64-bit binary hash $H$ is constructed by thresholding the gradient:
   $$H(i, j) = \begin{cases} 1 & \text{if } D(i, j) > 0 \\ 0 & \text{otherwise} \end{cases}$$
4. Perceptual distance between two images $A$ and $B$ is measured via the Hamming distance $d_H(H_A, H_B) = \sum_{k=1}^{64} (H_A[k] \oplus H_B[k])$. Pairs with $d_H \le 2$ are flagged as near-duplicates.

#### What dHash Proves
- Across all $5,547 \times 1,841 = 10,211,927$ pairwise cross-dataset comparisons, the minimum observed Hamming distance was $d_H = 11$ (well above the threshold of 2).
- This conclusively proves there are **zero identical or near-identical full-frame images** (even with uniform rescaling, minor contrast adjustments, or moderate JPEG compression) shared between the old and new datasets.

#### Critical Scientific Limitations
1. **No Scale or Crop Invariance**: If an image is tightly cropped to a vehicle or subject before insertion into one dataset, or if spatial margins are substantially altered, the relative luminance gradient across the $9 \times 8$ grid changes fundamentally ($d_H \gg 2$). dHash **cannot** detect cropped sub-images.
2. **No Viewpoint or Temporal Invariance**: Different camera perspectives of the same intersection or time-displaced frames of the same motorcycle will exhibit distinct hash bitmasks.
3. **No Non-Linear Photometric Invariance**: Extreme color grade transformations, heavy occlusions, or watermarking can disrupt gradient signs.

> [!IMPORTANT]
> **Scientific Caveat**: The absence of dHash matches demonstrates that no uncropped identical video frames were duplicated between datasets. However, the conclusive guarantee against sequence leakage rests on **metadata provenance and temporal/geographical sequence isolation** (the old dataset was acquired from historical fixed surveillance feeds in Dhaka, whereas the new expansion was independently captured from separate vehicle-mounted mobile and handheld recordings).

---

## 3. Current Readiness State

All pre-training prerequisites are complete:
1. `allfaz/old-dataset-v1-fixed` is live and verified on Kaggle.
2. `allfaz/helmet-dataset-expansion-v2` is live and verified on Kaggle.
3. GPU runtime environment, CUDA visibility, and dynamic path resolution are validated.
4. Model weights, optimizer gradients, and metrics log serialization are verified via local artifact inspection.
5. Quota impact is minimal (~1.40 hours estimated, with 29.90 hours remaining).

**Next Step**: Awaiting user approval to launch Phase 6 (`EXP-NEW`) full 50-epoch training kernel on Kaggle GPU.
