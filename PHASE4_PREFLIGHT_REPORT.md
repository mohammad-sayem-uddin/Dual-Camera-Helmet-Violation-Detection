# Phase 4 Pre-Flight Audit: Multi-Seed Statistical Validation

**Document**: Scientific Audit & Verification Report (`PHASE4_PREFLIGHT_REPORT.md`)  
**Project**: Dual-Camera Helmet Violation Detection  
**Auditor**: Senior Machine Learning Research Engineer & Reproducibility Auditor  
**Date**: August 27, 2026  
**Audit Target**: Multi-Seed Statistical Validation Protocol (n = 5 seeds)  

---

## 1. Executive Summary

A comprehensive pre-flight audit was conducted to verify the mathematical validity, configuration parity, checkpoint integrity, statistical methodology, and orchestration pipelines before initiating **Phase 4 Multi-Seed Statistical Validation**.

### Key Findings of the Audit:
1. **Sequence Isolation Verification**: The group-exclusive split (`data/splits/v1_group_split_fixed.csv`) has **mathematically verified 0.0% sequence overlap** across train, val, and test splits ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$). All 169 source video recordings are strictly atomic.
2. **Size-Matched Control Verification**: `EXP-003` training set contains **exactly 3,866 images**, identical to `EXP-002-FIXED` ($3,866 == 3,866$).
3. **Configuration Parity**: A field-by-field parameter audit confirmed 100% parity across model architecture (`yolov8n.pt`), image resolution ($640\times 640$), batch size ($16$), epochs ($50$), optimizer ($\text{AdamW}$), learning rate schedules, and all 13 online augmentation parameters.
4. **Seed 42 Reusability**: Existing `EXP-003` and `EXP-002-FIXED` seed 42 checkpoints and evaluations were audited against recorded SHA256 hashes, test sets, and convergence curves. Both runs are **100% valid and marked as REUSABLE**.
5. **Statistical Testing Suite**: Exact paired Student's t-tests ($df=4$), exact Wilcoxon signed-rank permutation tests ($2^5=32$ permutations), 95% confidence intervals, and paired Cohen's $d$ effect sizes are implemented and validated.
6. **Orchestration**: The multi-seed runner (`scripts/training/run_phase4_suite.py`) enforces collision-safe naming, logs hardware and timestamp metadata, protects frozen runs, and automates end-to-end evaluation.

---

## 2. Repository Status

* **Repository Root**: `/Users/sayemuddin/Desktop/research_helmet`
* **Python Environment**: Python 3.11.15, PyTorch 2.13.0, Ultralytics 8.4.129, Apple Silicon MPS.
* **Dataset Manifests**:
  - `data/metadata/source_groups.csv` (5,547 images across 169 unique atomic video groups)
  - `data/splits/v1_group_split_fixed.csv` (3,866 train, 852 val, 829 test)
  - `data/splits/v1_random_matched.csv` (3,866 train, 347 val, 346 test)
* **Status**: Clean, version-controlled, and audited.

---

## 3. EXP-003 Configuration Verification

* **Configuration File**: `configs/experiments/exp_003_size_matched_baseline.yaml`
* **Dataset Config**: `configs/dataset_random_matched.yaml`
* **Split Directory**: `data/processed/v1_random_matched/`
* **Train / Val / Test Counts**: 3,866 train, 347 val, 346 test images.
* **Architecture**: YOLOv8 nano (`yolov8n.pt`, 3,006,038 parameters).
* **Optimizer**: AdamW (`lr0=0.001667`, `momentum=0.937`, `weight_decay=0.0005`).
* **Verification Status**: **PASSED**.

---

## 4. EXP-002-FIXED Configuration Verification

* **Configuration File**: `configs/experiments/exp_002_group_split_fixed.yaml`
* **Dataset Config**: `configs/dataset_group_split_fixed.yaml`
* **Split Directory**: `data/processed/v1_group_split_fixed/`
* **Train / Val / Test Counts**: 3,866 train, 852 val, 829 test images.
* **Architecture**: YOLOv8 nano (`yolov8n.pt`, 3,006,038 parameters).
* **Optimizer**: AdamW (`lr0=0.001667`, `momentum=0.937`, `weight_decay=0.0005`).
* **Verification Status**: **PASSED**.

---

## 5. Configuration Difference Table (Field-by-Field Audit)

| Configuration Parameter | EXP-003 (Size Control) | EXP-002-FIXED (Sequence Split) | Parameter Match? | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `model` | `yolov8n.pt` | `yolov8n.pt` | **MATCH (Identical)** | Pretrained YOLOv8 Nano backbone |
| `pretrained` | `true` | `true` | **MATCH (Identical)** | Standard COCO initialization |
| `imgsz` | `640` | `640` | **MATCH (Identical)** | Native 640×640 spatial resolution |
| `batch` | `16` | `16` | **MATCH (Identical)** | Batch size per optimizer step |
| `epochs` | `50` | `50` | **MATCH (Identical)** | Standard convergence budget |
| `patience` | `25` | `25` | **MATCH (Identical)** | Early stopping validation threshold |
| `optimizer` | `"auto"` (AdamW) | `"auto"` (AdamW) | **MATCH (Identical)** | Ultralytics default optimizer |
| `learning_rate` | `null` (default) | `null` (default) | **MATCH (Identical)** | Standard auto-scaled LR |
| `lrf` | `0.01` | `0.01` | **MATCH (Identical)** | Final cosine LR fraction |
| `momentum` | `0.937` | `0.937` | **MATCH (Identical)** | SGD/AdamW momentum factor |
| `weight_decay` | `0.0005` | `0.0005` | **MATCH (Identical)** | L2 regularization weight |
| `warmup_epochs` | `3.0` | `3.0` | **MATCH (Identical)** | Linear warmup duration |
| `warmup_momentum` | `0.8` | `0.8` | **MATCH (Identical)** | Initial warmup momentum |
| `warmup_bias_lr` | `0.1` | `0.1` | **MATCH (Identical)** | Warmup bias learning rate |
| `deterministic` | `true` | `true` | **MATCH (Identical)** | Deterministic cuDNN/MPS kernels |
| `device` | `"auto"` (MPS) | `"auto"` (MPS) | **MATCH (Identical)** | Apple Silicon GPU acceleration |
| `workers` | `4` | `4` | **MATCH (Identical)** | DataLoader worker threads |
| `save_period` | `10` | `10` | **MATCH (Identical)** | Checkpoint snapshot frequency |
| `hsv_h` | `0.015` | `0.015` | **MATCH (Identical)** | Color hue augmentation |
| `hsv_s` | `0.7` | `0.7` | **MATCH (Identical)** | Saturation augmentation |
| `hsv_v` | `0.4` | `0.4` | **MATCH (Identical)** | Exposure / brightness augmentation |
| `degrees` | `0.0` | `0.0` | **MATCH (Identical)** | Rotation disabled |
| `translate` | `0.1` | `0.1` | **MATCH (Identical)** | Spatial translation jitter |
| `scale` | `0.5` | `0.5` | **MATCH (Identical)** | Image scaling jitter (0.5 to 1.5) |
| `shear` | `0.0` | `0.0` | **MATCH (Identical)** | Shear transformation disabled |
| `perspective` | `0.0` | `0.0` | **MATCH (Identical)** | Perspective distortion disabled |
| `flipud` | `0.0` | `0.0` | **MATCH (Identical)** | Vertical flip disabled |
| `fliplr` | `0.5` | `0.5` | **MATCH (Identical)** | Horizontal flip (50% probability) |
| `mosaic` | `1.0` | `1.0` | **MATCH (Identical)** | 4-image mosaic composition |
| `mixup` | `0.0` | `0.0` | **MATCH (Identical)** | Mixup disabled |
| `copy_paste` | `0.0` | `0.0` | **MATCH (Identical)** | Copy-paste disabled |
| `dataset_yaml` | `configs/dataset_random_matched.yaml` | `configs/dataset_group_split_fixed.yaml` | **INTENDED DIFFERENCE** | Isolates split methodology |
| `dataset_version` | `v1_random_matched` | `v1_group_split_fixed` | **INTENDED DIFFERENCE** | Target dataset directories |

*Conclusion*: Zero unintended configuration divergences exist. The only operational difference is the dataset splitting protocol.

---

## 6. Group Disjointness Proof

The group split manifest `data/splits/v1_group_split_fixed.csv` was audited using `scripts/data/verify_split_disjointness.py`.

### Mathematical Disjointness Results:
* **Train Split**: 117 unique video groups, 3,866 images.
* **Val Split**: 30 unique video groups, 852 images.
* **Test Split**: 22 unique video groups, 829 images.
* **Pairwise Intersections**:
  $$\text{TEST} \cap \text{TRAIN} = \emptyset \quad (\text{Overlap} = 0 \text{ groups}, 0 \text{ images})$$
  $$\text{TEST} \cap \text{VAL} = \emptyset \quad (\text{Overlap} = 0 \text{ groups}, 0 \text{ images})$$
  $$\text{TRAIN} \cap \text{VAL} = \emptyset \quad (\text{Overlap} = 0 \text{ groups}, 0 \text{ images})$$
* **Group Pattern Integrity**:
  - `VID*` recordings (28 groups, 328 images): Grouped by parent capture ID.
  - Timestamp recordings `YYYYMMDD_HHMMSS` (53 groups, 927 images): Grouped by capture timestamp.
  - Suffix recordings `YYYYMMDD_HHMMSS-N-` (2 groups, 36 images): Grouped by camera capture session.
  - `frame_N` recordings (86 groups, 4,256 images): Grouped by video burst ID.
* **Total Atomic Groups**: **169 unique groups** across 5,547 total images.
* **Verification Status**: **PASSED (0.0% Leakage Proven)**.

---

## 7. EXP-003 Size-Matching Verification

* **EXP-002-FIXED Training Images**: 3,866 images
* **EXP-003 Training Images**: 3,866 images
* **Subsampling Methodology**: Subsampled deterministically from EXP-001's 4,854 images using `seed=42`.
* **Manifest Verified**: `data/splits/v1_random_matched.csv` contains exactly 3,866 train, 347 val, and 346 test images.
* **Verification Status**: **PASSED (Exact Equality Verified: 3,866 == 3,866)**.

---

## 8. Existing Seed 42 Audit

Both seed 42 pilot experiments were thoroughly audited:

### 1. `EXP-003` (Seed 42)
* **Execution Date**: August 26, 2026
* **Training Arguments**: `seed=42`, `device='mps'`, `epochs=50`, `batch=16`, `imgsz=640`.
* **Duration**: 22,151.58s (6.15 h). Completed all 50 epochs.
* **Best Validation Epoch**: Epoch 46 (Val mAP50 = 78.91%).
* **Checkpoint**: `models/checkpoints/EXP-003/best.pt`
* **SHA256**: `dc4d4b50f9e66d44415f3b3106767bb53198cdddfe9fcb95d720e19363980822` (6,250,410 bytes).
* **Test Evaluation**: `results/metrics/EXP-003-test.json` (mAP50 = 0.8247, F1 = 0.8491).
* **Status**: **VALID & REUSABLE**.

### 2. `EXP-002-FIXED` (Seed 42)
* **Execution Date**: August 26–27, 2026
* **Training Arguments**: `seed=42`, `device='mps'`, `epochs=50`, `batch=16`, `imgsz=640`.
* **Duration**: 21,589.69s (6.00 h). Completed all 50 epochs.
* **Best Validation Epoch**: Epoch 40 (Val mAP50 = 77.96%).
* **Checkpoint**: `models/checkpoints/EXP-002-FIXED/best.pt`
* **SHA256**: `2388f2248aaf5ab4dae49d26a9552203c1c6fd43bbfe0ad1264fadca975978b8` (6,250,538 bytes).
* **Test Evaluation**: `results/metrics/EXP-002-FIXED-test.json` (mAP50 = 0.6694, F1 = 0.7328).
* **Status**: **VALID & REUSABLE**.

---

## 9. Checkpoint Integrity Verification

| Checkpoint | File Path | File Size | SHA256 Checksum | Integrity Status |
| :--- | :--- | :---: | :--- | :---: |
| **EXP-001** | `models/checkpoints/EXP-001/best.pt` | 6,250,154 B | `a8610ee01e72254d11626054dae376d661357160aa42f2058c4697bc56f6c1e3` | **VERIFIED** |
| **EXP-003** | `models/checkpoints/EXP-003/best.pt` | 6,250,410 B | `dc4d4b50f9e66d44415f3b3106767bb53198cdddfe9fcb95d720e19363980822` | **VERIFIED** |
| **EXP-002-FIXED** | `models/checkpoints/EXP-002-FIXED/best.pt` | 6,250,538 B | `2388f2248aaf5ab4dae49d26a9552203c1c6fd43bbfe0ad1264fadca975978b8` | **VERIFIED** |

All three checkpoints are frozen on disk and correspond bit-for-bit with the recorded experiment metadata.

---

## 10. Test Set Protocol Overlap Analysis

A detailed comparison between the test sets of `EXP-003` and `EXP-002-FIXED` was conducted:

| Test Set Characteristic | EXP-003 Test Set | EXP-002-FIXED Test Set | Intersection / Overlap |
| :--- | :---: | :---: | :---: |
| **Total Test Images** | 346 | 829 | **55 images (15.9% of EXP-003, 6.6% of FIXED)** |
| **Source Video Groups Represented** | 127 groups | 22 groups | 16 groups |
| **Training Set Sequence Overlap** | High (~100% of test groups in train) | **0.0% (0 test groups in train)** | N/A |
| **Total Annotated Objects** | 521 | 1,170 | N/A |
| **`no_helmet` Objects** | 326 (62.6%) | 600 (51.3%) | N/A |
| **`helmet` Objects** | 195 (37.4%) | 570 (48.7%) | N/A |
| **Negative (Background) Images** | 1 image | 1 image | 0 images |

### Critical Scientific Note on Evaluation Methodology:
> [!IMPORTANT]
> The comparison between `EXP-003` and `EXP-002-FIXED` estimates the difference between **two distinct evaluation paradigms**:
> 1. **Standard Random-Frame Benchmark Evaluation**: Test images are random frames from the same underlying video distribution as the training data.
> 2. **Sequence-Exclusive Generalization Evaluation**: Test images are drawn entirely from previously unseen video captures.
> Because the test splits represent different evaluation regimes, this is an **evaluation-protocol-level comparison**, not a sample-by-sample paired image evaluation.

---

## 11. Statistical Methodology Review

The statistical aggregation module (`scripts/analysis/compute_statistical_comparison.py`) was reviewed against statistical standards for $n = 5$ random seeds:

### Implemented Statistical Metrics:
1. **Sample Mean ($\mu$) & Sample Standard Deviation ($\sigma$)** for each evaluation condition.
2. **Mean Paired Difference ($\bar{D}$)** across matched seeds:
   $$\bar{D} = \frac{1}{n} \sum_{i=1}^n (\text{Metric}_{\text{EXP-003}, i} - \text{Metric}_{\text{EXP-002-FIXED}, i})$$
3. **95% Confidence Interval**:
   $$\text{CI}_{95\%} = \bar{D} \pm t_{\text{crit}} \cdot \frac{s_D}{\sqrt{n}}, \quad \text{where } t_{\text{crit}} = 2.7764 \text{ for } df = 4$$
4. **Paired Student's t-test**: Computes exact two-tailed $p$-values via the Student-t distribution CDF.
5. **Exact Wilcoxon Signed-Rank Test**: Calculates the exact non-parametric permutation distribution across all $2^5 = 32$ rank configurations without large-sample normal approximations.
6. **Paired Cohen's $d$ Effect Size**:
   $$d = \frac{\bar{D}}{s_D}$$

### Methodological Guardrails:
* Normality is not assumed or claimed for $n = 5$. Both parametric ($t$-test) and exact non-parametric (Wilcoxon) statistics are reported side-by-side.
* The analysis explicitly defines variance across random seeds as **training-stochasticity variability**, distinct from dataset sampling uncertainty.

---

## 12. Multi-Seed Runner Review

* **Runner Script**: `scripts/training/run_phase4_suite.py`
* **Features Verified**:
  1. CLI parameterization (`--seeds 42 123 2026 7 99`, `--dry-run`, `--only-stats`).
  2. Automatic skip/reuse of completed, verified runs (`seed=42`).
  3. Collision-free directory naming (`experiments/EXP-003_seed_<seed>/`, `experiments/EXP-002-FIXED_seed_<seed>/`).
  4. Automatic checkpoint preservation and SHA256 checksum calculation.
  5. Sequential execution of training $\to$ validation evaluation $\to$ test evaluation $\to$ training history analysis.
  6. Automated execution of statistical comparison reporting at pipeline completion.

---

## 13. Compute and Storage Estimate

Estimates based on actual recorded run durations on Apple Silicon MPS (`mps:0`, Apple M1):

| Condition | Training Imgs | Epochs | Measured Duration per Run | Runs to Execute | Subtotal Compute Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-003** (Seeds 123, 2026, 7, 99) | 3,866 | 50 | 6.15 hours (22,151.6s) | 4 | **24.60 hours** |
| **EXP-002-FIXED** (Seeds 123, 2026, 7, 99) | 3,866 | 50 | 6.00 hours (21,589.7s) | 4 | **24.00 hours** |
| **Seed 42 (Existing Runs)** | 3,866 | 50 | 12.15 hours (completed) | 0 (Reused) | **0.00 hours** |
| **Total Remaining Compute** | — | — | — | **8 runs** | **48.60 hours ($\approx 2.02$ days)** |

### Storage Requirements:
* Storage per run (checkpoints, logs, training curves, test outputs): $\approx 150\text{ MB}$.
* Total storage for 8 new runs: $\approx 1.2\text{ GB}$.
* Available disk space on filesystem: **21.0 GB** (17.5× safety margin).

---

## 14. Git and Reproducibility Status

* **Current Branch**: `main` (clean tracking).
* **Modified Core Scripts**:
  - `scripts/training/train_yolo.py` (added `--seed` and `--experiment-id` CLI overrides).
  - `scripts/evaluation/evaluate_yolo.py` (added macOS MPS `workers=0` stability).
  - `scripts/analysis/compute_statistical_comparison.py` (added exact $t$-test, Wilcoxon, Cohen's $d$, 95% CI).
* **New Validated Tools**:
  - `scripts/training/run_phase4_suite.py` (Phase 4 multi-seed orchestrator).
  - `PHASE3_PILOT_REPORT.md` (Phase 3 report).
* **.gitignore Rules**: Confirmed protection of `.venv/`, `data/annotated/`, `data/processed/`, `models/checkpoints/`, `*.pt`, and `results/evaluation/`.
* **Recommended Commit Command**:
  ```bash
  git add scripts/ configs/ experiments/EXP-002-FIXED/ experiments/EXP-003/ results/metrics/ results/tables/ PHASE3_PILOT_REPORT.md
  git commit -m "feat: complete Phase 3 pilot and prepare Phase 4 multi-seed validation framework"
  ```

---

## 15. Known Scientific Limitations

1. **Small Sample Size ($n = 5$)**:
   While $n = 5$ random seeds is standard for object detection benchmarking given compute constraints (~48 GPU hours), power for non-parametric tests is bounded ($p_{\text{min}} = 2/32 = 0.0625$ for a two-sided Wilcoxon test). Effect sizes (Cohen's $d$) and confidence intervals must be prioritized alongside $p$-values.
2. **Fixed Split Realizations**:
   The multi-seed protocol varies optimization and data-loading stochasticity (weights initialization, mini-batch shuffling, mosaic augmentation) across a fixed dataset split partition. It isolates training variability rather than resampling variance over the space of all possible video splits.

---

## 16. Explicit Phase 4 Training Plan

When approved, Phase 4 will execute the following 8 training and evaluation workflows sequentially:

```
[EXP-003 Suite — Size-Matched Random Control]
  1. EXP-003_seed_123   (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  2. EXP-003_seed_2026  (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  3. EXP-003_seed_7     (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  4. EXP-003_seed_99    (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)

[EXP-002-FIXED Suite — Sequence-Exclusive Group Split]
  5. EXP-002-FIXED_seed_123   (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  6. EXP-002-FIXED_seed_2026  (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  7. EXP-002-FIXED_seed_7     (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)
  8. EXP-002-FIXED_seed_99    (Train 50 epochs -> Val Eval -> Test Eval -> Analysis)

[Statistical Aggregation]
  9. Run compute_statistical_comparison.py across all 5 seeds (42, 123, 2026, 7, 99).
 10. Generate publication-ready summary tables, metrics JSON, and final research report.
```

---

# FINAL DECISION

## **PHASE 4 STATUS: READY TO START**

All 16 audit dimensions have passed inspection. No blocking bugs, leakage paths, configuration mismatches, or file corruptions exist.

### Exact 8 New Runs Scheduled for Execution:
**EXP-003 (Random-Frame Size-Matched Control, 3,866 Train Images)**:
1. `EXP-003_seed_123` (Seed 123)
2. `EXP-003_seed_2026` (Seed 2026)
3. `EXP-003_seed_7` (Seed 7)
4. `EXP-003_seed_99` (Seed 99)

**EXP-002-FIXED (Sequence-Exclusive Split, 3,866 Train Images, 0.0% Overlap)**:
5. `EXP-002-FIXED_seed_123` (Seed 123)
6. `EXP-002-FIXED_seed_2026` (Seed 2026)
7. `EXP-002-FIXED_seed_7` (Seed 7)
8. `EXP-002-FIXED_seed_99` (Seed 99)

*(Seed 42 for both EXP-003 and EXP-002-FIXED is verified, frozen, and will be reused without retraining).*

---
*Awaiting your explicit authorization to launch the Phase 4 training suite.*
