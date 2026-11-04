# Comprehensive Kaggle Research Plan & Dual-Dataset Forensic Audit

**DOCUMENT:** `reports/KAGGLE_RESEARCH_PLAN_AND_AUDIT.md`  
**PROJECT:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**RESEARCH AGENT:** Senior Computer Vision Research Engineer & Kaggle Execution Agent  
**KAGGLE ACCOUNT:** `allfaz` (User ID: 18614709)  
**ACCELERATOR QUOTA:** 108,000s (30.0 hours) GPU remaining | 72,000s TPU remaining  
**STATUS:** **`PHASES 1–4 COMPLETED — AWAITING USER APPROVAL BEFORE GPU TRAINING`**

---

## 1. Executive Summary

This report establishes the empirical foundation, cryptographic audit, and experimental plan for the two controlled YOLOv8 helmet violation detection experiments:
1. **EXP-NEW**: YOLOv8n trained exclusively on the sequence-isolated, precision-refined new motorcycle expansion dataset (`helmet_dataset_expansion_v2`).
2. **EXP-COMBINED**: YOLOv8n trained on the merged, zero-leakage combined dataset (Old Sequence-Isolated `kawke_dibona_v1` + New `helmet_dataset_expansion_v2`).

Both datasets have undergone complete integrity, semantic class alignment, SHA-256 exact-duplicate, perceptual dHash near-duplicate, and video-sequence leakage audits.

---

## 2. Phase 1: Repository Reconnaissance Findings

- **Workspace**: `/Users/sayemuddin/Desktop/research_helmet`
- **Git Repository State**: Standalone research workspace. An incomplete `.git` directory was detected (missing `HEAD`/`config`), preventing direct Git CLI execution. No uncommitted Git state is at risk.
- **Old Dataset Provenance**:
  - Located at `data/annotated/kawke_dibona_v1/` (5,547 images).
  - Labels reconstructed from verified Ultralytics `labels.cache` into standard YOLO format at `data/processed/old_dataset_v1_fixed/`.
  - Ground-truth split defined in `data/splits/v1_group_split_fixed.csv` across 169 video sequence groups (Train: 117, Val: 30, Test: 22).
- **New Dataset Provenance**:
  - Raw images: `Motorcyle dataset/` (1,841 images, 1024×1024).
  - Precision-refined dataset: `data/helmet_dataset_expansion/training_ready_v2/` with `data.yaml`.
  - 392 oversized bounding boxes precision-trimmed to tight cranial perimeters.
  - 523 uncertain head instances across 429 images strictly quarantined (0.0% leakage).
  - Split defined across 565 sequence groups (Train: 438, Val: 74, Test: 53).
- **Historical Benchmark Results**:
  - `EXP-001` (Random Frame Split on Old Data): Test mAP@50 = **83.42%**, F1 = **86.26%**.
  - `EXP-002` / `EXP-002-FIXED` (Sequence-Exclusive Split on Old Data): Test mAP@50 = **64.44%**, F1 = **72.29%**.
  - **Leakage Penalty**: A confirmed **-18.98 pp mAP drop** proving that random frame splitting inflates performance via temporal autocorrelation.

---

## 3. Phase 2: Kaggle Reconnaissance Findings

- **Kaggle MCP Connectivity**: 100% operational (all 76 tools active).
- **Authenticated Identity**:
  - Username: `allfaz`
  - User ID: `18614709`
  - Performance Tier: Contributor (Competitions, Datasets, Notebooks)
- **Current Accelerator Quota**:
  - GPU Quota: Exactly **108,000 seconds (30.0 hours)** allowed.
  - GPU Time Used: **0 seconds** (100% quota available).
  - GPU Time Reserved: **0 seconds**.
  - TPU Quota: **72,000 seconds (20.0 hours)** allowed (0s used).
  - Quota Reset Timestamp: `2026-11-07T00:00:00Z`.
- **Existing Kaggle Datasets Under Account**: **0 datasets**. Neither the old nor the new dataset is currently present on Kaggle.
- **Existing Kaggle Notebooks Under Account**: 1 default starter notebook (`allfaz/notebook8acbfe7f7c`).
- **Accelerator Selection**: Preferred accelerator is **NVIDIA T4 GPU** (16GB VRAM).
- **Upload Requirement**: **YES**. Both datasets must be uploaded to Kaggle as private research datasets or packaged for the private training notebook.

---

## 4. Phase 3: Dataset Audit & Global Leakage Analysis

A dedicated automated audit script (`scripts/audit_both_datasets.py`) was executed to evaluate cryptographic duplicates, perceptual similarity, and sequence boundaries across both datasets.

### 4.1 Master Dataset Census & Class Consistency

| Dataset Metric | Old Dataset (`kawke_dibona_v1`) | New Dataset (`expansion_v2`) | Merged Combined (`combined_v2`) |
| :--- | :---: | :---: | :---: |
| **Total Images** | **5,547** | **1,841** | **7,388** |
| **Total Bounding Boxes** | **8,248** | **3,733** | **11,981** |
| **Class 0 (`no_helmet`) Boxes** | 4,888 (59.26%) | 1,707 (45.73%) | 6,595 (55.05%) |
| **Class 1 (`helmet`) Boxes** | 3,360 (40.74%) | 2,026 (54.27%) | 5,386 (44.95%) |
| **Class Mapping** | `0: no_helmet, 1: helmet` | `0: no_helmet, 1: helmet` | `0: no_helmet, 1: helmet` (100% Match) |
| **Corrupted / Unreadable Images** | 0 | 0 | 0 |
| **Missing Image-Label Pairs** | 0 | 0 | 0 |
| **Out-of-Bounds Coordinates** | 0 | 0 | 0 |
| **Unique Video Sequence Groups** | **169** | **565** | **734** |

### 4.2 Cross-Dataset Cryptographic & Perceptual Leakage Findings

1. **Exact SHA-256 Duplicates**: Exactly **0** cross-dataset duplicates. Every image across both datasets is physically and digitally unique.
2. **Perceptual Near-Duplicates (dHash $\text{Hamming Distance} \le 2$)**: Exactly **0** cross-dataset matches. No recompressed, slightly cropped, or shared video frames exist between the two datasets.
3. **Cross-Dataset Sequence Overlap**: Exactly **0** shared sequence groups ($169 \cap 565 = \emptyset$).
4. **Conclusion**: The two datasets represent completely distinct real-world traffic collections. They can be safely combined across splits without any temporal autocorrelation leakage.

### 4.3 Split Distribution Matrix

| Split Population | Old Dataset (EXP-002-FIXED) | New Dataset (EXP-NEW) | Combined (EXP-COMBINED) |
| :--- | :---: | :---: | :---: |
| **Train Images** | 3,866 (5,870 boxes) | 1,285 (2,584 boxes) | **5,151 images (8,454 boxes)** |
| — *Class 0 (`no_helmet`)* | 3,634 | 1,236 | 4,870 (57.61%) |
| — *Class 1 (`helmet`)* | 2,236 | 1,348 | 3,584 (42.39%) |
| **Val Images** | 852 (1,208 boxes) | 276 (597 boxes) | **1,128 images (1,805 boxes)** |
| — *Class 0 (`no_helmet`)* | 654 | 203 | 857 (47.48%) |
| — *Class 1 (`helmet`)* | 554 | 394 | 948 (52.52%) |
| **Test Images (Held-Out)** | 829 (1,170 boxes) | 280 (552 boxes) | **1,109 images (1,722 boxes)** |
| — *Class 0 (`no_helmet`)* | 600 | 268 | 868 (50.41%) |
| — *Class 1 (`helmet`)* | 570 | 284 | 854 (49.59%) |
| **Ultralytics Loader Smoke Test** | PASSED (4.3k it/s) | PASSED (598.9 it/s) | **PASSED (4.3k it/s)** |

---

## 5. Phase 4: Controlled Experiment Plan & Quota Budget

To establish rigorous scientific attribution, both experiments share an identical model family, initialization checkpoint, hyperparameter configuration, and evaluation harness.

### 5.1 Experimental Configuration Matrix

| Experimental Parameter | EXP-NEW | EXP-COMBINED | Control Rationale |
| :--- | :--- | :--- | :--- |
| **Experiment Purpose** | Train on New Dataset Only | Train on Merged (Old + New) | Measure multi-dataset generalization gain |
| **Model Architecture** | YOLOv8n (`yolov8n.pt`) | YOLOv8n (`yolov8n.pt`) | Identical parameter count (3.2M params) |
| **Pretrained Weights** | Official Ultralytics COCO | Official Ultralytics COCO | Common initialization checkpoint |
| **Random Seed** | 42 (Deterministic) | 42 (Deterministic) | Eliminates weight init divergence |
| **Input Resolution** | 640×640 | 640×640 | Consistent baseline resolution |
| **Batch Size** | 32 (T4 GPU 16GB) | 32 (T4 GPU 16GB) | Maximizes GPU tensor core utilization |
| **Epoch Budget** | 50 epochs | 50 epochs | Standard convergence window |
| **Early Stopping** | Patience = 20 epochs | Patience = 20 epochs | Prevents quota waste after convergence |
| **Optimizer** | AdamW (`lr0=0.001667`) | AdamW (`lr0=0.001667`) | Matched to EXP-002-FIXED baseline |
| **Augmentation Policy** | Mosaic=1.0, FlipLR=0.5, Scale=0.5 | Mosaic=1.0, FlipLR=0.5, Scale=0.5 | Identical spatial-color perturbation |

### 5.2 Common Held-Out Evaluation Protocol

To prevent invalid cross-test comparisons, both trained models will be evaluated across **three distinct held-out test populations**:
1. **New Dataset Test Set (280 images, 552 boxes)**: Evaluates in-domain performance on the expansion dataset.
2. **Old Dataset Test Set (829 images, 1,170 boxes)**: Evaluates zero-shot cross-domain generalization (for EXP-NEW) vs. in-domain generalization (for EXP-COMBINED).
3. **Combined Held-Out Test Set (1,109 images, 1,722 boxes)**: Evaluates global multi-distribution robustness.

### 5.3 GPU Quota Budget & Runtime Estimation

| Phase / Run | Training Samples | Estimated Epoch Time | Total Epochs | Estimated GPU Runtime | Quota Impact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Notebook Setup & Smoke Test** | 50 (test) | ~2 sec | 2 | ~1.5 min | 0.03 hrs |
| **EXP-NEW Training** | 1,285 | ~8–10 sec | 50 | ~8–10 min | **0.15 hrs** |
| **EXP-COMBINED Training** | 5,151 | ~30–35 sec | 50 | ~26–30 min | **0.50 hrs** |
| **Multi-Population Evaluation** | 1,109 (test) | — | 3 suites | ~5–7 min | **0.10 hrs** |
| **Total Estimated Consumption** | — | — | — | **~45–50 minutes** | **~0.78 hours** |

> [!NOTE]
> **GPU Quota Safety Margin**: Out of the available **30.0 hours (108,000s)**, the entire experimental pipeline will consume only **~0.78 hours (< 2.7% of total weekly quota)**. Over 29.2 hours will remain safely preserved for subsequent seed replications or resolution investigations.

---

## 6. Execution Pathway Upon Approval

Once approved by the user, execution proceeds through:
1. **Packaging**: Create compact private archives of `helmet_dataset_expansion_v2` and `old_dataset_v1_fixed`.
2. **Kaggle Deployment**: Upload datasets via Kaggle MCP and configure private execution notebook.
3. **Smoke Test**: Run a 2-epoch dry run on Kaggle T4 GPU to verify CUDA visibility, tensor shapes, and loss convergence.
4. **EXP-NEW Execution**: Run full 50-epoch training and evaluate on all 3 test populations.
5. **EXP-COMBINED Execution**: Run full 50-epoch training and evaluate on all 3 test populations.
6. **Artifact Retrieval**: Download `best.pt`, `last.pt`, confusion matrices, and metric JSON files to persistent local storage.
7. **Scientific Attribution & Statistical Reporting**: Generate comparative P/R/F1/mAP curves, per-class error analysis, and paired statistical significance tests.
