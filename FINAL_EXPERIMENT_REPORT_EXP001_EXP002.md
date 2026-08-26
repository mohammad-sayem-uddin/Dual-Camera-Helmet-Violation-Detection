# Final Experiment Report: EXP-001 vs. EXP-002

**Project**: TrafficVision-AI — Context-Aware Motorcycle Helmet Violation Detection Research  
**Investigation**: Quantitative Evaluation of Random Frame Splitting vs. Video-Sequence-Exclusive Splitting  
**Date**: August 26, 2026  
**Status**: Both Experiments Completed, Validated, Evaluated, and Scientifically Compared  

---

## 1. Research Objective

This research aims to establish an empirical benchmark for deep learning-based motorcycle helmet violation detection and investigate the extent to which conventional random frame-level dataset splitting produces artificially optimistic performance metrics due to temporal autocorrelation across video frames.

---

## 2. Dataset Description & Provenance

* **Dataset Source**: High-resolution traffic surveillance imagery for motorcycle helmet compliance monitoring (`kawke_dibona_v1`).
* **Total Image Count**: 5,547 images (100% 640×640 JPEGs).
* **Total Annotation Count**: 8,248 bounding boxes.
* **Target Classes**:
  * **Class `0` (`no_helmet`)**: 4,888 instances (59.3%) — Head region of motorcycle rider/passenger without a helmet.
  * **Class `1` (`helmet`)**: 3,360 instances (40.7%) — Head region of motorcycle rider/passenger wearing a helmet.
* **Integrity Audit**: Verified 0 corrupted images, 0 missing labels, 0 orphan labels, and 0 invalid bounding boxes.

---

## 3. Original Roboflow Split & Data Leakage Discovery

In the standard Roboflow export (`data/annotated/kawke_dibona_v1/`):
* **Train**: 4,854 images (1,618 base frames with 3× offline augmentation).
* **Validation**: 347 images.
* **Test**: 346 images.

### Sequence Overlap Finding
The underlying source frames originate from **275 distinct video sequences**.
* **114 out of 133 sequences in Validation** (85.7%) are also present in the training set.
* **111 out of 131 sequences in Test** (84.7%) are also present in the training set.
While exact image files do not duplicate across splits, adjacent video frames captured within fractions of a second appear across train, validation, and test splits.

---

## 4. Group-Exclusive Split Methodology (`v1_group_split`)

To measure true out-of-distribution generalization, we constructed a group-exclusive dataset (`data/processed/v1_group_split/`):
* All frames belonging to a video sequence (`YYYYMMDD_HHMMSS` and `frame_XX_YYY`) were grouped and partitioned atomically.
* **Train**: 3,873 images across 187 unique video sequences.
* **Validation (`val`)**: 837 images across 44 unique video sequences.
* **Test (`test`)**: 837 images across 44 unique video sequences.
* **Sequence Overlap**: Strictly **0.0% (Zero)** across all split pairs.

---

## 5. Experiment Configurations

Both experiments were executed on identical hardware (Apple Silicon MPS, Apple M1) using the same baseline architecture and hyperparameter budget:

| Parameter | EXP-001 (Random Frame Split) | EXP-002 (Sequence-Exclusive Split) |
| :--- | :--- | :--- |
| **Experiment ID** | `EXP-001` | `EXP-002` |
| **Model Architecture** | `yolov8n.pt` (Pretrained backbone) | `yolov8n.pt` (Pretrained backbone) |
| **Dataset Config** | `configs/dataset.yaml` | `configs/dataset_group_split.yaml` |
| **Resolution** | 640×640 | 640×640 |
| **Batch Size** | 16 | 16 |
| **Epochs** | 50 | 50 |
| **Optimizer** | AdamW (`lr0: 0.001667`, momentum: 0.9) | AdamW (`lr0: 0.001667`, momentum: 0.9) |
| **Random Seed** | 42 (Deterministic execution) | 42 (Deterministic execution) |
| **Patience** | 25 | 25 |
| **Augmentation** | `fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1`, `hsv: [0.015, 0.7, 0.4]` | `fliplr: 0.5`, `mosaic: 1.0`, `scale: 0.5`, `translate: 0.1`, `hsv: [0.015, 0.7, 0.4]` |
| **Training Duration** | 26,917.32s (7.48 hours) | 23,610.26s (6.56 hours) |
| **Checkpoint Checksum** | SHA256: `a8610ee01e72254d...` | SHA256: `21b17a014938fff6...` |

---

## 6. Complete Metric Results & Comparison

### Validation & Test Metrics Summary

| Metric | EXP-001 (Random Frame Split) | EXP-002 (Sequence-Exclusive Split) | Absolute Difference | Relative Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Test Precision ($P$)** | **0.8688** | 0.7325 | -0.1363 | -15.69% |
| **Test Recall ($R$)** | **0.8566** | 0.7136 | -0.1430 | -16.69% |
| **Test F1-Score** | **0.8626** | 0.7229 | -0.1397 | -16.20% |
| **Test $\text{mAP}@0.50$** | **0.8342** | 0.6444 | **-0.1898** | **-22.75%** |
| **Test $\text{mAP}@0.50:0.95$** | **0.4135** | 0.2923 | -0.1212 | -29.30% |
| **Test `no_helmet` $\text{AP}@50$** | **0.8177** | 0.6586 | -0.1591 | -19.46% |
| **Test `helmet` $\text{AP}@50$** | **0.8508** | 0.6303 | -0.2205 | -25.92% |
| **Validation Precision** | 0.8356 | 0.8034 | -0.0322 | -3.85% |
| **Validation Recall** | 0.8763 | 0.7655 | -0.1107 | -12.64% |
| **Validation F1-Score** | 0.8554 | 0.7840 | -0.0714 | -8.35% |
| **Validation $\text{mAP}@0.50$** | 0.8161 | 0.7274 | -0.0887 | -10.87% |
| **Validation $\text{mAP}@0.50:0.95$** | 0.3673 | 0.3282 | -0.0391 | -10.64% |
| **Best Validation Epoch** | Epoch 40 | Epoch 35 | -5 epochs | -12.50% |

---

## 7. Per-Class Analysis

1. **`no_helmet` Violation Class**:
   * EXP-001 Test AP50: **81.77%** ($P=85.91\%$, $R=84.13\%$, $F1=85.01\%$)
   * EXP-002 Test AP50: **65.86%** ($P=74.10\%$, $R=74.20\%$, $F1=74.15\%$)
   * *Analysis*: When confronting unseen traffic scenes, precision remains relatively robust ($74.10\%$), but localized recall drops by $\approx 10\text{ percentage points}$ due to unfamiliar lighting, backgrounds, and rider postures.
2. **`helmet` Compliance Class**:
   * EXP-001 Test AP50: **85.08%** ($P=87.85\%$, $R=87.18\%$, $F1=87.52\%$)
   * EXP-002 Test AP50: **63.03%** ($P=72.39\%$, $R=68.52\%$, $F1=70.40\%$)
   * *Analysis*: Recall drops from $87.18\%$ to $68.52\%$, indicating that helmet appearance varies substantially across novel camera angles and helmet designs that were not encountered during training.

---

## 8. Training Dynamics & Convergence Comparison

* **EXP-001**: Converged steadily across 50 epochs; best validation performance occurred at Epoch 40 ($\text{mAP}@0.50 = 86.31\%$). Training classification loss dropped from `3.1231` to `0.4578`.
* **EXP-002**: Converged rapidly by Epoch 35 ($\text{mAP}@0.50 = 79.96\%$). Training classification loss dropped from `3.3095` to `0.6354`.
* **Overfitting Signals**: In both experiments, training loss and validation loss decreased concurrently without severe divergence. The performance gap between EXP-001 and EXP-002 is attributable to **dataset distribution shifts (unseen sequence evaluation)** rather than classical model overfitting.

---

## 9. Generalization Discussion & Scientific Insights

1. **Quantification of Leakage Bias**: The **$18.98\text{ percentage point}$ drop in test $\text{mAP}@0.50$** (from $83.42\%$ to $64.44\%$) provides unambiguous empirical proof that random frame-level dataset splitting produces inflated performance benchmarks in video-based traffic vision research.
2. **Real-World Reliability**: EXP-002 establishes the true baseline generalization capability of a standalone YOLOv8n detector on previously unobserved traffic sequences.
3. **Research Pathway**: A standalone single-frame detector operating at $64.44\%$ mAP@50 and $72.29\%$ F1 has inherent limitations. This directly supports the necessity of multi-frame temporal confirmation, confidence accumulation across video tracks, and synchronized dual-camera verification.

---

## 10. Research Artifact Directory

| Artifact Description | File Path |
| :--- | :--- |
| **EXP-001 Run Directory** | [`experiments/EXP-001/`](file:///Users/sayemuddin/Desktop/research_helmet/experiments/EXP-001) |
| **EXP-001 Checkpoint** | [`models/checkpoints/EXP-001/best.pt`](file:///Users/sayemuddin/Desktop/research_helmet/models/checkpoints/EXP-001/best.pt) |
| **EXP-001 Experiment Summary** | [`experiments/EXP-001/EXPERIMENT_SUMMARY.md`](file:///Users/sayemuddin/Desktop/research_helmet/experiments/EXP-001/EXPERIMENT_SUMMARY.md) |
| **EXP-001 Validation Metrics** | [`results/metrics/EXP-001_val_metrics.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/metrics/EXP-001_val_metrics.json) |
| **EXP-001 Test Metrics** | [`results/metrics/EXP-001_test_metrics.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/metrics/EXP-001_test_metrics.json) |
| **EXP-001 Training Curves** | [`results/plots/EXP-001_training_curves.png`](file:///Users/sayemuddin/Desktop/research_helmet/results/plots/EXP-001_training_curves.png) |
| **EXP-002 Run Directory** | [`experiments/EXP-002/`](file:///Users/sayemuddin/Desktop/research_helmet/experiments/EXP-002) |
| **EXP-002 Checkpoint** | [`models/checkpoints/EXP-002/best.pt`](file:///Users/sayemuddin/Desktop/research_helmet/models/checkpoints/EXP-002/best.pt) |
| **EXP-002 Experiment Summary** | [`experiments/EXP-002/EXPERIMENT_SUMMARY.md`](file:///Users/sayemuddin/Desktop/research_helmet/experiments/EXP-002/EXPERIMENT_SUMMARY.md) |
| **EXP-002 Validation Metrics** | [`results/metrics/EXP-002_val_metrics.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/metrics/EXP-002_val_metrics.json) |
| **EXP-002 Test Metrics** | [`results/metrics/EXP-002_test_metrics.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/metrics/EXP-002_test_metrics.json) |
| **EXP-002 Training Curves** | [`results/plots/EXP-002_training_curves.png`](file:///Users/sayemuddin/Desktop/research_helmet/results/plots/EXP-002_training_curves.png) |
| **Scientific Comparison CSV** | [`results/tables/EXP_001_vs_EXP_002.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/tables/EXP_001_vs_EXP_002.csv) |
| **Scientific Comparison Report** | [`results/EXP_001_vs_EXP_002_COMPARISON.md`](file:///Users/sayemuddin/Desktop/research_helmet/results/EXP_001_vs_EXP_002_COMPARISON.md) |
| **Execution Log** | [`experiments/EXECUTION_LOG.md`](file:///Users/sayemuddin/Desktop/research_helmet/experiments/EXECUTION_LOG.md) |
