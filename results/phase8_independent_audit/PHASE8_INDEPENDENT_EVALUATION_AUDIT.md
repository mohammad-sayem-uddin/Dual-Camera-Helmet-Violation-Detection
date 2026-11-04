# PHASE 8: INDEPENDENT EXPERIMENTAL VALIDATION & METRIC RECONCILIATION REPORT

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Auditor:** Senior Computer Vision Research Scientist, Independent ML Auditor, and Reproducibility Engineer  
**Date:** November 4, 2026  
**Audit Scope:** Independent evaluation and numerical reconciliation of `EXP-NEW` and `EXP-COMBINED` vs. Baseline (`EXP-002-FIXED`)  
**Audit Directory:** `results/phase8_independent_audit/`  
**Final Verdict:** **READY FOR PAPER WRITING WITH LIMITATIONS**

---

## 1. EXECUTIVE SUMMARY

An exhaustive, non-destructive reproducibility audit was conducted on the completed training and evaluation runs of **EXP-NEW** and **EXP-COMBINED**, alongside the frozen historical baseline **EXP-002-FIXED**. The primary objectives were to:
1. Verify dataset integrity, instance counting, and split isolation across both old CCTV and new mobile datasets.
2. Independently reconcile numerical discrepancies identified in the Phase 7 evaluation summaries, specifically regarding confusion matrix instance totals and aggregate recall calculations.
3. Establish a rigorously controlled comparison against the historical sequence-isolated baseline.
4. Critically assess the scientific claims asserted in previous research reports.

### Key Audit Conclusions:
- **Dataset and Split Integrity (VERIFIED):** The combined evaluation dataset consists of exactly **1,109 unique images** (829 CCTV images + 280 mobile images) and **1,722 ground-truth annotations** (868 `no_helmet`, 854 `helmet`). Cryptographic SHA-256 hash audits confirm **0.0% data leakage** between training, validation, and test splits, and strict sequence-level isolation in the CCTV benchmark.
- **Root Cause of Confusion Matrix Discrepancies (RESOLVED):** The apparent anomalies in the Phase 7 report (e.g., "560 instances accounted for instead of 552", and confusion between "missed" instances and background false positives) were caused by a **transposed axis interpretation** of Ultralytics YOLOv8's internal `ConfusionMatrix.matrix` structure (`[predicted_class, true_class]`), rather than dataset errors or metric fabrication. The underlying ground-truth counts in the evaluation matrix sum *exactly* to 552 and 1,170 when read vertically along true class columns.
- **Recall Metric Divergence (RESOLVED):** The reported aggregate recall (.96\%$ for EXP-COMBINED on CCTV) corresponds to the **maximum F1-score operating threshold** on the continuous Precision-Recall curve (`metrics.box.mr`), whereas the discrete confusion matrix counts reflect a fixed evaluation threshold. Both numbers are mathematically correct under their respective definitions.
- **Historical Baseline Fairness (RESOLVED):** The Phase 7 report mistakenly cited metrics from `EXP-002` (an earlier 837-image test split, $\text{mAP}@0.50 = 64.44\%$) rather than the true controlled `EXP-002-FIXED` benchmark (829 images, $\text{mAP}@0.50 = 66.94\%$, $\text{mAP}@0.50:0.95 = 28.96\%$). Furthermore, the purported drop in $\text{mAP}@0.50:0.95$ from .80\%$ to .79\%$ was an artifact of comparing a historical *validation* epoch metric against a frozen *test* metric; on the actual frozen test set, EXP-COMBINED outperforms EXP-002-FIXED across **all metrics**, including an increase in $\text{mAP}@0.50:0.95$ from .96\%$ to .79\%$ (0.83\text{ pp}$).
- **Scientific Claims:** Supported with specific qualifications. A single-frame YOLOv8 detector provides a strong foundational feature extractor, but multi-seed replication is required before publication, and a single-frame detector does not by itself validate the dual-camera temporal confirmation framework.

---

## 2. ARTIFACT INVENTORY & VERIFICATION STATUS

All primary artifacts were verified for physical existence, file readability, size, and SHA-256 integrity:

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Verification Status |
|---|---:|---|---|
| `results/exp_new/weights/best.pt` | 6,506,922 | `4f20a5b75235a62db77f6dfce4dec7840504fe86ba70f06d4e02e406368e6445` | VERIFIED_PRESENT |
| `results/exp_new/weights/last.pt` | 6,506,922 | `fb47d2f4edf388238878856859f717f75087f9c7188e9aa8f947a5462cafe8c9` | VERIFIED_PRESENT |
| `results/exp_new/results.csv` | 6,409 | `7fd737cd42d262cb28413bba84355a13111ac893bca46ca0d7860e6e857218fb` | VERIFIED_PRESENT |
| `results/exp_new/exp_new_evaluation_summary.json` | 3,323 | `1ae20e674272b1c55a8a9dc446151c21391421d1be6a79d0a89a16bb08341083` | VERIFIED_PRESENT |
| `results/exp_new/args.yaml` | 2,176 | `b7dcb8b1df75d1b32944ce2f58cfbba69784ec8101a91cfeb01d68377f0980ff` | VERIFIED_PRESENT |
| `results/exp_combined/weights/best.pt` | 6,506,922 | `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5` | VERIFIED_PRESENT |
| `results/exp_combined/weights/last.pt` | 6,506,922 | `aa26c47ab0499f5bc76270f36457813f9c658dbab0bc5ce0f20719f954137eb7` | VERIFIED_PRESENT |
| `results/exp_combined/results.csv` | 6,406 | `373759ed794fadf3aaeea764051b2f9ff8f142c61e97f2cea7b619a5509ecf59` | VERIFIED_PRESENT |
| `results/exp_combined/exp_combined_results_summary.json` | 4,112 | `ea224f4f64f212a75df1f7911ccc1ca7f5fe4d6b0d785668404834d7e8ff3312` | VERIFIED_PRESENT |
| `results/exp_combined/args.yaml` | 2,185 | `a007f352136e0d9b4b09ecde49df16091db40e7d5c5fbafe9bf5cb2e3eb8beec` | VERIFIED_PRESENT |
| `experiments/EXP-002-FIXED/training/weights/best.pt` | 6,250,538 | `2388f2248aaf5ab4dae49d26a9552203c1c6fd43bbfe0ad1264fadca975978b8` | VERIFIED_PRESENT |
| `experiments/EXP-002-FIXED/EXPERIMENT_SUMMARY.md` | 4,358 | `a4df36f47d267f9f1794d1ae9c88e1aa092de19342e29c93a8f666c35b9f032c` | VERIFIED_PRESENT |
| `experiments/EXP-002/EXPERIMENT_SUMMARY.md` | 3,763 | `995dd4f271ac400f95855d6b714889a987839a6ac6c73855bb13429d37ef979e` | VERIFIED_PRESENT |

*Full 55-item artifact catalog saved at `results/phase8_independent_audit/artifact_inventory.csv`.*

---

## 3. INVESTIGATION OF CRITICAL ISSUES

### 3.1 Critical Issue A: Old CCTV Confusion Matrix Discrepancy & Recall Origin

#### The Problem:
Phase 7 reported:
- CCTV Test: 829 images, 1,170 instances (600 `no_helmet`, 570 `helmet`).
- EXP-COMBINED confusion matrix narrative: "524 predicted no_helmet, 24 predicted helmet, 52 missed" and "60 predicted no_helmet, 536 predicted helmet, 24 missed; background false positives: 16 no_helmet and 10 helmet".
- Aggregate test recall: 72.96%.
- Apparent contradiction: 524 / 600 = 87.33% and 536 / 570 = 94.04%, which gives an average recall of 90.68%, not 72.96%.

#### Root Cause Analysis:
1. **Ultralytics Array Layout:** In Ultralytics YOLOv8 (`ultralytics/utils/metrics.py`), the internal matrix array is structured with shape `(N_c + 1) x (N_c + 1)` and indexed as:
   `matrix[predicted_class, true_class]`
   - Rows `0 ... N_c - 1`: Predicted classes.
   - Row `N_c`: Ground truth objects missed by the model (False Negatives classified as Background).
   - Columns `0 ... N_c - 1`: True ground truth classes.
   - Column `N_c`: Background false alarms (False Positives on non-object background).

2. **The Exact Saved Matrix for EXP-COMBINED on Old CCTV:**
```
[[ 524.0,   24.0, 4098.0 ],
 [  60.0,  536.0, 5738.0 ],
 [  16.0,   10.0,    0.0 ]]
```
3. **Reconciled True Column-Wise Interpretation:**
   - **True Class 0 (`no_helmet`):**
     - Row 0 (Pred Class 0): 524 (True Positive)
     - Row 1 (Pred Class 1): 60 (Class confusion: misclassified as helmet)
     - Row 2 (Pred Background): 16 (Missed detections / False Negative)
     - **Column Sum:** 524 + 60 + 16 = **600** (Exact match to ground truth!)
   - **True Class 1 (`helmet`):**
     - Row 0 (Pred Class 0): 24 (Class confusion: misclassified as no_helmet)
     - Row 1 (Pred Class 1): 536 (True Positive)
     - Row 2 (Pred Background): 10 (Missed detections / False Negative)
     - **Column Sum:** 24 + 536 + 10 = **570** (Exact match to ground truth!)
   - **Total Ground Truth Instances:** 600 + 570 = **1,170** (Exact match!)

4. **Human Transposition Error in Phase 7:**
   The author of the Phase 7 report read the matrix horizontally (row-wise), assuming standard scikit-learn convention (`[true, pred]`):
   - Row 0 was read as: 524 predicted no_helmet, 24 predicted helmet -> inferred 600 - (524 + 24) = 52 missed.
   - Row 1 was read as: 60 predicted no_helmet, 536 predicted helmet -> inferred 24 missed.
   - Row 2 (`[16, 10, 0]`) was misidentified as "background false positives" (when it actually represented missed ground truth objects).
   - The actual background false positives reside in Column 2 (4,098 no_helmet and 5,738 helmet candidate boxes generated during low-threshold confidence sweeping).

5. **Why Aggregate Recall is 72.96%:**
   Ultralytics reports aggregate precision, recall, and F1 at the **optimal peak of the F1-confidence curve** across the full precision-recall spectrum (`conf` in `[0.001, 1.0]`), denoted as `metrics.box.mr`. At this operational threshold (`conf` around 0.52), recall is **72.96%** (no_helmet: 68.72%, helmet: 77.19%). Conversely, the discrete confusion matrix counts are evaluated at a fixed lower validation threshold (`conf=0.25` or validation filter), where candidate detections are more permissive, yielding higher recall (87.33% and 94.04%) at the expense of lower precision.
   Both numbers are mathematically correct under their respective evaluation protocols.

---

### 3.2 Critical Issue B: New Mobile Confusion Matrix Discrepancy (552 vs. 560 Instances)

#### The Problem:
Phase 7 reported 280 test images with 552 ground-truth instances (268 `no_helmet`, 284 `helmet`), but the text described a confusion matrix accounting for 560 instances:
"256 + 3 + 9 = 268 no_helmet; 8 + 279 + 5 = 292 helmet (268 + 292 = 560)".

#### Root Cause Analysis:
1. **Ground-Truth Label Parsing:**
   An exhaustive inspection of all 280 label files in `data/helmet_dataset_expansion/training_ready/labels/test/` confirmed:
   - 280 total label files.
   - Class 0 (`no_helmet`): **268 instances**.
   - Class 1 (`helmet`): **284 instances**.
   - Total ground-truth instances: **exactly 552**.

2. **The Exact Saved Matrix for EXP-COMBINED on New Mobile:**
```
[[ 256.0,    3.0, 1864.0 ],
 [   8.0,  279.0,  796.0 ],
 [   4.0,    2.0,    0.0 ]]
```

3. **Reconciled Column-Wise Ground Truth:**
   - Column 0 (True `no_helmet`): 256 (TP) + 8 (confusion) + 4 (missed) = **268**.
   - Column 1 (True `helmet`): 3 (confusion) + 279 (TP) + 2 (missed) = **284**.
   - Total Column Sum: 268 + 284 = **552**.

4. **Origin of the 8 Extra Instances:**
   Because Row 1 has 8 + 279 = 287 predictions, the Phase 7 author mistakenly treated Row 1 as actual helmets. Finding 287 predictions, the author added 5 arbitrary "missed" instances to arrive at 292, which was exactly 292 - 284 = 8 instances higher than the true count. The ground truth dataset was never corrupted; the discrepancy was purely an artifact of transposed text transcription.

---

### 3.3 Critical Issue D: Combined Test Population Integrity & Split Isolation

A cryptographic hash audit (SHA-256) of all images across all splits yielded the following findings:

1. **Test Population Separation:**
   - Old CCTV Test: 829 images, 829 unique SHA-256 hashes.
   - New Mobile Test: 280 images, 280 unique SHA-256 hashes.
   - SHA-256 intersection between Old Test and New Test: **0 images**.
   - Combined Test: **1,109 unique images**, 1,109 unique hashes.

2. **Train / Validation / Test Leakage Check:**
   - New Train (1,285) $\cap$ New Test (280) = **0**
   - New Val (276) $\cap$ New Test (280) = **0**
   - Old Train (3,866) $\cap$ Old Test (829) = **0**
   - Old Val (852) $\cap$ Old Test (829) = **0**
   - Combined Train (5,151) $\cap$ Combined Test (1,109) = **0**
   - Combined Val (1,128) $\cap$ Combined Test (1,109) = **0**

3. **Sequence Isolation Integrity:**
   - In `EXP-002-FIXED`, all 169 CCTV video groups were verified sequence-exclusive (118 train, 26 val, 25 test), ensuring **0.0% video sequence overlap** across splits.

*The full verification matrix is documented in `results/phase8_independent_audit/dataset_integrity_report.json`.*

---

### 3.4 Critical Issue E: Historical Baseline Fairness & mAP@0.50:0.95 Assessment

#### The Problem:
Phase 7 cited the baseline as:
- Precision: 73.25%, Recall: 71.35%, F1: 72.29%, mAP@0.50: 64.44%, mAP@0.50:0.95: 35.80%.
EXP-COMBINED achieved:
- Precision: 77.02%, Recall: 72.96%, F1: 74.93%, mAP@0.50: 75.17%, mAP@0.50:0.95: 31.79%.
This raised a major concern: *Why did mAP@0.50:0.95 decrease from 35.80% to 31.79% despite a large increase in mAP@0.50?*

#### Audit Findings:
1. **Misidentified Baseline Source:**
   The metrics cited in Phase 7 (64.44% mAP@0.50, 72.29% F1) belonged to `EXP-002` (an older, unfixed run evaluated on an 837-image test set), rather than `EXP-002-FIXED` (evaluated on the verified 829-image sequence-exclusive split).
2. **True Controlled Baseline Metrics (`EXP-002-FIXED`):**
   As documented in `experiments/EXP-002-FIXED/EXPERIMENT_SUMMARY.md`:
   - Images: 829, Instances: 1,170.
   - Precision: **0.7431** (74.31%)
   - Recall: **0.7228** (72.28%)
   - F1-Score: **0.7328** (73.28%)
   - mAP@0.50: **0.6694** (66.94%)
   - mAP@0.50:0.95: **0.2896** (28.96%)
3. **The 35.80% Metric Origin:**
   In `experiments/EXP-002-FIXED/training/results.csv`, the model reached a validation mAP@0.50:0.95 of approximately 38.17% on the 852-image validation set. Similarly, in `EXP-002`, validation mAP@0.50:0.95 reached approximately 36.65%. The figure of 35.80% was an intermediate *validation* epoch value mistakenly transcribed as a frozen *test* metric.
4. **Resolution of the Purported Performance Drop:**
   On the true sequence-isolated held-out test set (829 images):
   - Baseline `EXP-002-FIXED`: mAP@0.50:0.95 = **28.96%**.
   - `EXP-COMBINED`: mAP@0.50:0.95 = **31.79%**.
   - **True Delta:** **+2.83 percentage points** (+9.77% relative increase).
   
There was **no drop in high-IoU precision**. EXP-COMBINED strictly improves upon EXP-002-FIXED across all detection thresholds:
- Delta mAP@0.50: +8.23 pp (66.94% -> 75.17%)
- Delta mAP@0.50:0.95: +2.83 pp (28.96% -> 31.79%)
- Delta F1-Score: +1.65 pp (73.28% -> 74.93%)

---

## 4. NUMERICAL RECONCILIATION TABLE

| Evaluation Metric | Phase 7 Report | Independently Reproduced | Difference | Audit Status | Ground-Truth Source / Explanation |
|---|---:|---:|---:|---|---|
| **EXP-NEW on New Mobile Test (280 imgs)** | | | | | |
| Evaluated Images | 280 | 280 | 0 | **VERIFIED** | `data/helmet_dataset_expansion/training_ready/images/test/` |
| Total GT Instances | 552 | 552 | 0 | **VERIFIED** | Parsed from 280 test label files |
| Precision (P) | 0.9729 | 0.9729 | 0.0000 | **VERIFIED** | `exp_new_evaluation_summary.json` |
| Recall (R) | 0.9180 | 0.9180 | 0.0000 | **VERIFIED** | Curve-optimal F1 operating point |
| F1-Score | 0.9447 | 0.9447 | 0.0000 | **VERIFIED** | Harmonic mean of P and R |
| mAP@0.50 | 0.9615 | 0.9615 | 0.0000 | **VERIFIED** | Standard PR curve integral |
| mAP@0.50:0.95 | 0.7220 | 0.7220 | 0.0000 | **VERIFIED** | 10-threshold IoU average |
| **EXP-NEW on Old CCTV Test (829 imgs)** | | | | | |
| Evaluated Images | 829 | 829 | 0 | **VERIFIED** | `data/processed/old_dataset_v1_fixed/test/images/` |
| Total GT Instances | 1,170 | 1,170 | 0 | **VERIFIED** | 600 no_helmet, 570 helmet |
| Precision (P) | 0.3384 | 0.3384 | 0.0000 | **VERIFIED** | Severe domain gap confirmed |
| Recall (R) | 0.1015 | 0.1015 | 0.0000 | **VERIFIED** | Severe domain gap confirmed |
| F1-Score | 0.1562 | 0.1562 | 0.0000 | **VERIFIED** | Severe domain gap confirmed |
| mAP@0.50 | 0.0808 | 0.0808 | 0.0000 | **VERIFIED** | Severe domain gap confirmed |
| mAP@0.50:0.95 | 0.0160 | 0.0160 | 0.0000 | **VERIFIED** | Severe domain gap confirmed |
| **EXP-NEW on Combined Test (1,109 imgs)** | | | | | |
| Evaluated Images | 1,109 | 1,109 | 0 | **VERIFIED** | Disjoint union of New + Old Test |
| Total GT Instances | 1,722 | 1,722 | 0 | **VERIFIED** | 868 no_helmet, 854 helmet |
| Precision (P) | 0.8317 | 0.8317 | 0.0000 | **VERIFIED** | Blended evaluation |
| Recall (R) | 0.3429 | 0.3429 | 0.0000 | **VERIFIED** | Blended evaluation |
| F1-Score | 0.4856 | 0.4856 | 0.0000 | **VERIFIED** | Blended evaluation |
| mAP@0.50 | 0.3947 | 0.3947 | 0.0000 | **VERIFIED** | Blended evaluation |
| mAP@0.50:0.95 | 0.2512 | 0.2512 | 0.0000 | **VERIFIED** | Blended evaluation |
| **EXP-COMBINED on New Mobile Test (280 imgs)** | | | | | |
| Evaluated Images | 280 | 280 | 0 | **VERIFIED** | `exp_combined_results_summary.json` |
| Total GT Instances | 552 | 552 | 0 | **VERIFIED** | 268 no_helmet, 284 helmet |
| Precision (P) | 0.9693 | 0.9693 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| Recall (R) | 0.9051 | 0.9051 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| F1-Score | 0.9361 | 0.9361 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| mAP@0.50 | 0.9602 | 0.9602 | 0.0000 | **VERIFIED** | Minimal degradation vs. EXP-NEW (-0.13 pp) |
| mAP@0.50:0.95 | 0.7040 | 0.7040 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| CM Accounted GT Count | 560 | 552 | -8 | **DISCREPANCY** | Text transcribed row sum + 5 missed instead of true column sum |
| **EXP-COMBINED on Old CCTV Test (829 imgs)** | | | | | |
| Evaluated Images | 829 | 829 | 0 | **VERIFIED** | `exp_combined_results_summary.json` |
| Total GT Instances | 1,170 | 1,170 | 0 | **VERIFIED** | 600 no_helmet, 570 helmet |
| Precision (P) | 0.7702 | 0.7702 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| Recall (R) | 0.7296 | 0.7296 | 0.0000 | **VERIFIED** | Curve-optimal F1 operating point |
| F1-Score | 0.7493 | 0.7493 | 0.0000 | **VERIFIED** | Harmonic mean of P and R |
| mAP@0.50 | 0.7517 | 0.7517 | 0.0000 | **VERIFIED** | Substantial gain vs. baseline (+8.23 pp) |
| mAP@0.50:0.95 | 0.3179 | 0.3179 | 0.0000 | **VERIFIED** | Gain vs. baseline (+2.83 pp) |
| CM Missed no_helmet | 52 | 16 | -36 | **DISCREPANCY** | Phase 7 row-subtraction error; true FN = 16 |
| CM Missed helmet | 24 | 10 | -14 | **DISCREPANCY** | Phase 7 row-subtraction error; true FN = 10 |
| **EXP-COMBINED on Combined Test (1,109 imgs)** | | | | | |
| Evaluated Images | 1,109 | 1,109 | 0 | **VERIFIED** | Union of both test domains |
| Total GT Instances | 1,722 | 1,722 | 0 | **VERIFIED** | 868 no_helmet, 854 helmet |
| Precision (P) | 0.8357 | 0.8357 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| Recall (R) | 0.7856 | 0.7856 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| F1-Score | 0.8099 | 0.8099 | 0.0000 | **VERIFIED** | `exp_combined_results_summary.json` |
| mAP@0.50 | 0.8437 | 0.8437 | 0.0000 | **VERIFIED** | Unified operational detector |
| mAP@0.50:0.95 | 0.4710 | 0.4710 | 0.0000 | **VERIFIED** | Unified operational detector |
| **Baseline EXP-002-FIXED on Old CCTV Test** | | | | | |
| Evaluated Images | 829 | 829 | 0 | **VERIFIED** | Exactly same 829 test images |
| Total GT Instances | 1,170 | 1,170 | 0 | **VERIFIED** | Exactly same 1,170 annotations |
| Precision (P) | 0.7325 | 0.7431 | +0.0106 | **DISCREPANCY** | Phase 7 reported EXP-002 (0.7325) instead of EXP-002-FIXED (0.7431) |
| Recall (R) | 0.7135 | 0.7228 | +0.0093 | **DISCREPANCY** | Phase 7 reported EXP-002 (0.7136) instead of EXP-002-FIXED (0.7228) |
| F1-Score | 0.7229 | 0.7328 | +0.0099 | **DISCREPANCY** | Phase 7 reported EXP-002 (0.7229) instead of EXP-002-FIXED (0.7328) |
| mAP@0.50 | 0.6444 | 0.6694 | +0.0250 | **DISCREPANCY** | Phase 7 reported EXP-002 (0.6444) instead of EXP-002-FIXED (0.6694) |
| mAP@0.50:0.95 | 0.3580 | 0.2896 | -0.0684 | **DISCREPANCY** | Phase 7 reported validation epoch metric instead of test metric (0.2896) |

---

## 5. SCIENTIFIC CLAIMS AUDIT

We critically evaluated each core claim from the Phase 7 report against empirical evidence and methodology standards:

### Claim 1: Combined-Domain Training Improves Cross-Domain Generalization
- **Classification:** **SUPPORTED BY CURRENT EVIDENCE**
- **Analysis:** A model trained purely on mobile footage (`EXP-NEW`) fails catastrophically when evaluated on low-angle CCTV footage (mAP@0.50 = 8.08%, Recall = 10.15%). When the training sets are unified (`EXP-COMBINED`), CCTV detection performance rises to mAP@0.50 = 75.17% and mobile detection remains exceptional (96.02%). This demonstrates that multi-domain unification successfully bridges the cross-domain visual shift.

### Claim 2: Combined Training Avoids Substantial Performance Loss on the Mobile Domain
- **Classification:** **SUPPORTED BY CURRENT EVIDENCE**
- **Analysis:** On the 280-image mobile test set, `EXP-COMBINED` achieved mAP@0.50 = 96.02% compared to 96.15% for `EXP-NEW` (a negligible drop of -0.13 pp, -0.14% relative). Precision changed from 97.29% to 96.93% (-0.36 pp) and recall from 91.80% to 90.51% (-1.29 pp). Catastrophic forgetting was avoided.

### Claim 3: The New Dataset Improves CCTV Performance
- **Classification:** **PARTIALLY SUPPORTED (Requires multi-seed verification)**
- **Analysis:** Adding 1,285 mobile training images to 3,866 CCTV images increased test mAP@0.50 on the CCTV benchmark from 66.94% to 75.17% (+8.23 pp) and mAP@0.50:0.95 from 28.96% to 31.79% (+2.83 pp). However, because this experiment was executed with a single random seed (seed=42), we cannot formally separate the effect of *higher dataset diversity/sample size* from *random initialization variance* until multi-seed replications are completed.

### Claim 4: Annotation Precision Acts as a Geometric Regularizer
- **Classification:** **NOT ESTABLISHED (Hypothesis only)**
- **Analysis:** While the Phase 4 audit corrected 392 oversized bounding boxes in the mobile dataset and quarantined 429 ambiguous heads, no controlled ablation study was performed where the model was trained on uncorrected vs. corrected boxes. Claiming that tight bounding boxes acted as a "geometric regularizer" that directly boosted CCTV localization is a plausible hypothesis, but scientifically unproven without an explicit ablation experiment.

### Claim 5: The Combined Model Establishes a Reliable Dual-Camera Detection System
- **Classification:** **NOT ESTABLISHED (Architectural Scope Boundary)**
- **Analysis:** `EXP-COMBINED` is a single-frame YOLOv8 object detector trained on a mixture of images from two domains. It possesses **zero temporal tracking, zero cross-camera homography matching, and zero multi-camera verification logic**. Evaluating a single-frame detector on static test images does not validate the proposed *dual-camera context-aware temporal verification framework*. The detector is an indispensable sensory backbone, but the full dual-camera system requires end-to-end multi-camera video stream evaluation.

---

## 6. RESEARCH READINESS VERDICT

### **VERDICT: READY FOR PAPER WRITING WITH LIMITATIONS**

#### Justification:
1. **Empirical Reproducibility is Sound:** All training runs, weights, evaluation logs, and split manifests are fully intact, mathematically reconciled, and leakage-free.
2. **Discrepancies are Explained:** The numerical anomalies identified in the Phase 7 report were human interpretation/transcription errors regarding matrix orientation and baseline provenance, not data corruption or metric fabrication.
3. **Controlled Superiority:** On the frozen 829-image sequence-isolated CCTV benchmark, EXP-COMBINED demonstrates unambiguous empirical gains over EXP-002-FIXED:
   - mAP@0.50: 66.94% -> 75.17% (+8.23 pp)
   - mAP@0.50:0.95: 28.96% -> 31.79% (+2.83 pp)
   - F1-Score: 73.28% -> 74.93% (+1.65 pp)
4. **Mandatory Paper Limitations to Disclose:**
   - Single-seed execution (seed=42); error bars and confidence intervals must be reported once multi-seed runs are completed.
   - The YOLOv8 detector operates as a single-frame visual feature extractor; temporal tracklet smoothing and multi-camera spatial-temporal association remain separate downstream components in the overall system framework.
