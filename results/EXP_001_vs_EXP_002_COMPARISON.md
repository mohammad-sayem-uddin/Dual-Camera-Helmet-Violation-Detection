# Scientific Comparison: EXP-001 vs. EXP-002

**Research Study**: Impact of Video Sequence Data Leakage on Motorcycle Helmet Violation Detection Performance  
**Date**: August 26, 2026  
**Investigator**: Experiment Execution & Research Validation Operator  

---

## 1. Research Question & Experimental Hypothesis

> **Research Question**: *How does conventional random frame-level dataset splitting compare with strict video-sequence-exclusive splitting when evaluating a YOLOv8 detector for motorcycle helmet violation detection?*

### Experimental Design (Controlled Pair)
* Both experiments utilize the identical model architecture (`yolov8n.pt`, 3.0M parameters).
* Both experiments employ identical training hyperparameters (640×640 resolution, batch size 16, 50 epochs, AdamW optimizer, random seed 42, deterministic execution on Apple Silicon MPS).
* Both experiments use the identical online augmentation policy.
* **Controlled Variable**:
  * **EXP-001**: Conventional Roboflow random frame split (85.7% validation and 84.7% test sequence overlap with train).
  * **EXP-002**: Strict sequence-isolated group split (**0.0% sequence overlap** across all splits).

---

## 2. Quantitative Performance Comparison

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
| **Training Duration** | 7.48 hours | 6.56 hours | -0.92 hours | -12.30% |

---

## 3. Scientific Interpretation & Discussion

The empirical findings fall decisively under **Case A**: EXP-001 achieves significantly higher test performance ($\text{mAP}@0.50 = 83.42\%$ vs. $64.44\%$, a relative difference of **$-22.75\%$**).

### Interpretation:
1. **Evidence of Temporal Autocorrelation Inflation**:
   * In EXP-001, test evaluation occurred on individual video frames whose preceding or succeeding frames (often captured milliseconds apart during the same motorcycle transit) were present in the training set. This allowed the detector to benefit from familiar background scenery, identical rider apparel, vehicle color, and camera angle.
   * The inflated metric in EXP-001 represents an optimistic performance estimate rather than genuine generalization to novel traffic scenes.
2. **True Out-of-Sequence Generalization (EXP-002)**:
   * In EXP-002, the model was tested exclusively on 44 video sequences from completely unseen camera vantage points and distinct traffic periods.
   * The resulting performance ($\text{mAP}@0.50 = 64.44\%$, $\text{F1} = 72.29\%$) represents the true baseline capability of a single-frame YOLOv8n detector on novel traffic footage.
3. **Implications for Violation Detection Research**:
   * The $\approx 19.0\text{ percentage point}$ drop in $\text{mAP}@0.50$ highlights why single-camera, single-frame detectors often underperform when deployed in real-world surveillance infrastructure.
   * This provides strong empirical motivation for the broader research project: context-aware rider-vehicle association, temporal multi-frame confidence accumulation, and dual-camera synchronized verification are necessary to bridge this generalization gap.
