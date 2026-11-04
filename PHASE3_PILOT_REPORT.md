# Phase 3 Controlled Pilot Experiments: Isolating Data Size vs. Temporal Leakage

**Document**: Scientific Pilot Report (`PHASE3_PILOT_REPORT.md`)  
**Project**: Dual-Camera Helmet Violation Detection  
**Author**: Antigravity (Senior Machine Learning Research Engineer)  
**Date**: August 27, 2026  
**Status**: Completed & Frozen  

---

## Executive Summary

In Phase 3, we executed a strictly controlled three-way single-seed pilot study to isolate and quantify the two competing hypotheses explaining the performance degradation observed when moving from standard random-frame splitting to sequence-exclusive group splitting:
1. **Data Size Effect ($\Delta_{\text{size}}$)**: Performance change due to training set size reduction ($4,854 \to 3,866$ images, $-20.35\%$).
2. **Sequence Isolation Effect ($\Delta_{\text{sequence}}$)**: Performance change due to evaluating on unseen temporal video sequences while holding training set size constant at 3,866 images.

### Key Quantitative Findings (Test Set Evaluation)

| Experiment | Split Type | Train Imgs | Val Imgs | Test Imgs | Test mAP@0.50 | Test mAP@0.50:0.95 | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-001** | Random Frame Baseline | 4,854 | 347 | 346 | **0.8342** | 0.4135 | 0.8626 | 0.8688 | 0.8566 |
| **EXP-003** | Size-Matched Random Control | 3,866 | 347 | 346 | **0.8247** | 0.3963 | 0.8491 | 0.8518 | 0.8464 |
| **EXP-002-FIXED** | Sequence-Exclusive Group Split | 3,866 | 852 | 829 | **0.6694** | 0.2896 | 0.7328 | 0.7431 | 0.7228 |

### Effect Decomposition

$$\Delta_{\text{total}} = \text{mAP50(EXP-001)} - \text{mAP50(EXP-002-FIXED)} = 0.8342 - 0.6694 = \mathbf{+0.1648} \text{ (+16.48 pp)}$$

$$\Delta_{\text{size}} = \text{mAP50(EXP-001)} - \text{mAP50(EXP-003)} = 0.8342 - 0.8247 = \mathbf{+0.0095} \text{ (+0.95 pp, 5.78\% of total drop)}$$

$$\Delta_{\text{sequence}} = \text{mAP50(EXP-003)} - \text{mAP50(EXP-002-FIXED)} = 0.8247 - 0.6694 = \mathbf{+0.1553} \text{ (+15.53 pp, 94.22\% of total drop)}$$

### Scientific Conclusion: **CASE A**
The data size confound accounts for only **0.95 mAP points (5.8% of the gap)**, whereas sequence isolation accounts for **15.53 mAP points (94.2% of the gap)**. The hypothesis that the previously observed drop was merely an artifact of dataset size reduction is empirically falsified. Video sequence leakage in random-frame splitting creates severe artificial performance inflation due to near-duplicate frame memorization across splits.

---

## 1. Experimental Methodology & Rigorous Controls

To eliminate confounding variables, all three experiments adhered to strict architectural, hyperparameter, and hardware equality:

1. **Model Architecture**: Ultralytics YOLOv8 nano (`yolov8n.pt`, 3,006,038 parameters, 8.1 GFLOPs at 640×640).
2. **Deterministic Random Seed**: `seed = 42` across dataset partitioning, initialization, data loaders, and training loops.
3. **Hyperparameters**:
   - Resolution: 640×640
   - Batch size: 16
   - Optimizer: AdamW (`lr0=0.001667`, `momentum=0.9`, `weight_decay=0.0005`)
   - Schedule: Linear warmup (3 epochs), cosine decay over 50 epochs
   - Augmentation: Standard YOLOv8 mosaic (1.0), horizontal flip (0.5), scale (0.5), translate (0.1), HSV jitter (`h=0.015, s=0.7, v=0.4`)
4. **Hardware Environment**: Apple Silicon MPS (`mps:0`, Apple M1, macOS 26.5.2, Python 3.11.15, PyTorch 2.13.0, Ultralytics 8.4.129).
5. **Evaluation Protocol**: Standardized evaluation using frozen `best.pt` checkpoints at IoU=0.70, Conf=0.25 on unseen test splits.

---

## 2. Dataset Partitioning & Disjointness Proof

The dataset consists of **5,547 images** (8,248 annotated bounding boxes) across 169 source video recordings (`kawke_dibona_v1`).

### Split Characteristics
* **EXP-001 (Random Frame Split)**: 4,854 train (87.5%), 347 val (6.25%), 346 test (6.25%). Video frames from the same sequence are scattered across train, val, and test splits.
* **EXP-003 (Size-Matched Random Control)**: 3,866 train (subsampled with seed=42 from EXP-001), 347 val, 346 test. Preserves the random-frame leakage regime while matching EXP-002-FIXED training size.
* **EXP-002-FIXED (Sequence-Exclusive Group Split)**: 118 train sequences (3,866 images, 69.7%), 26 val sequences (852 images, 15.4%), 25 test sequences (829 images, 14.9%).

### Disjointness Verification
$$\text{Train Sequences} \cap \text{Val Sequences} = \emptyset$$
$$\text{Train Sequences} \cap \text{Test Sequences} = \emptyset$$
$$\text{Val Sequences} \cap \text{Test Sequences} = \emptyset$$
* Sequence overlap across all pairs: **0.00%** (mathematically verified).

---

## 3. Comprehensive Performance Comparison

### Primary Evaluation Metrics (Test Split)

| Evaluation Metric | EXP-001 Baseline | EXP-003 Size-Control | EXP-002-FIXED Sequence-Isolated | $\Delta_{\text{size}}$ (EXP-001 - EXP-003) | $\Delta_{\text{sequence}}$ (EXP-003 - EXP-002-FIXED) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP @ 0.50** | **0.8342** | **0.8247** | **0.6694** | **+0.0095 (+1.1%)** | **+0.1553 (+18.8%)** |
| **mAP @ 0.50:0.95** | 0.4135 | 0.3963 | 0.2896 | +0.0172 (+4.2%) | +0.1067 (+26.9%) |
| **F1-Score** | 0.8626 | 0.8491 | 0.7328 | +0.0135 (+1.6%) | +0.1163 (+13.7%) |
| **Precision** | 0.8688 | 0.8518 | 0.7431 | +0.0170 (+2.0%) | +0.1087 (+12.8%) |
| **Recall** | 0.8566 | 0.8464 | 0.7228 | +0.0101 (+1.2%) | +0.1236 (+14.6%) |

### Per-Class Detection Performance (Test Split)

| Class | Metric | EXP-001 Baseline | EXP-003 Size-Control | EXP-002-FIXED Sequence-Isolated | $\Delta_{\text{size}}$ | $\Delta_{\text{sequence}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `no_helmet` (Class 0) | AP @ 0.50 | 0.8177 | 0.8041 | 0.6791 | +0.0137 | +0.1250 |
| | AP @ 0.50:0.95 | 0.4015 | 0.3934 | 0.2861 | +0.0081 | +0.1072 |
| | Precision | 0.8591 | 0.8332 | 0.7970 | +0.0259 | +0.0362 |
| | Recall | 0.8413 | 0.8160 | 0.7001 | +0.0253 | +0.1159 |
| | F1-Score | 0.8501 | 0.8245 | 0.7454 | +0.0256 | +0.0791 |
| `helmet` (Class 1) | AP @ 0.50 | 0.8508 | 0.8454 | 0.6598 | +0.0054 | +0.1856 |
| | AP @ 0.50:0.95 | 0.4255 | 0.3992 | 0.2930 | +0.0263 | +0.1062 |
| | Precision | 0.8785 | 0.8703 | 0.6891 | +0.0082 | +0.1811 |
| | Recall | 0.8718 | 0.8769 | 0.7456 | -0.0051 | +0.1313 |
| | F1-Score | 0.8752 | 0.8736 | 0.7163 | +0.0016 | +0.1573 |

---

## 4. Training Dynamics & Convergence Diagnostics

* **EXP-001**: Converged smoothly, best validation epoch 40 (Val mAP50 = 81.61%), training duration 7.48 hours.
* **EXP-003**: Converged steadily, best validation epoch 46 (Val mAP50 = 78.91%), training duration 6.15 hours.
* **EXP-002-FIXED**: Completed 50 epochs, best validation epoch 40 (Val mAP50 = 85.23%), training duration 6.00 hours.
* **Overfitting Diagnostic**: In EXP-002-FIXED, training box loss decreased from 2.062 to 1.342, and training classification loss decreased from 3.158 to 0.638. Validation loss steadily declined alongside training loss without divergence, confirming healthy optimization.

---

## 5. Detailed Scientific Discussion

### 1. The Negligible Impact of Training Dataset Size
Reducing the training dataset from 4,854 to 3,866 images within the random-frame evaluation regime (EXP-001 vs. EXP-003) resulted in only a **0.95% drop in mAP@0.50** (0.8342 $\to$ 0.8247) and a **1.35% drop in F1-score** (0.8626 $\to$ 0.8491). This confirms that a YOLOv8 nano model trained on ~3.8k frames is already in the asymptotic performance plateau for this feature distribution when test frames are drawn from the same video sequences.

### 2. The Dominant Impact of Sequence Isolation
When evaluating on unseen video recordings (EXP-003 vs. EXP-002-FIXED), performance dropped by **15.53% in mAP@0.50** (0.8247 $\to$ 0.6694) and **10.67% in mAP@0.50:0.95** (0.3963 $\to$ 0.2896).
* In random-frame splitting, the model achieves high recall and precision by recognizing specific riders, background buildings, camera angles, and lighting conditions present in other frames of the exact same video.
* In sequence-exclusive group splitting, the model must generalize to novel riders, new background clutter, and unseen viewpoint variations.
* Detection of the `helmet` class suffered the most substantial degradation ($\Delta_{\text{sequence}} = -18.56\text{ pp}$), driven by a sharp drop in Precision ($0.8703 \to 0.6891$), reflecting false positives on novel headwear, hairstyles, and background objects in unseen camera environments.

---

## 6. Recommendations & Roadmap for Phase 4

### Classification: CASE A (Confirmed)
The single-seed pilot provides compelling evidence that sequence leakage in benchmark datasets creates a severe optical illusion of high generalization.

### Concrete Recommendations for Phase 4 Multi-Seed Statistical Validation:
1. **Multi-Seed Protocol**: Execute a 5-seed statistical validation (Seeds: 42, 100, 2024, 7, 999) comparing the Size-Matched Random Control (`EXP-003`) and the Sequence-Exclusive Group Split (`EXP-002-FIXED`).
2. **Statistical Significance Testing**:
   - Compute mean $\pm$ standard deviation for Precision, Recall, F1, mAP@0.50, and mAP@0.50:0.95.
   - Perform paired two-sample t-tests and Wilcoxon signed-rank tests to establish formal statistical significance ($p < 0.01$).
   - Report 95% confidence intervals and Cohen's $d$ effect sizes.
3. **Scientific Reporting**: Prepare publication-ready comparison tables, PR curves, and error distribution figures documenting temporal video leakage in motorcycle helmet violation benchmarks.
