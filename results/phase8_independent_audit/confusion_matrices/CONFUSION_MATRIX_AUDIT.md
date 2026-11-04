# Independent Confusion Matrix Audit & Dissection

In Ultralytics YOLOv8, confusion matrices follow the indexing convention `matrix[predicted_class, true_class]`:
- **Row 0**: Predicted `no_helmet` (Class 0)
- **Row 1**: Predicted `helmet` (Class 1)
- **Row 2**: Predicted Background (Missed True Positives / False Negatives)
- **Column 0**: True `no_helmet` (Class 0)
- **Column 1**: True `helmet` (Class 1)
- **Column 2**: True Background (False Positives on background areas)

## EXP-COMBINED on New Test (280 images, 552 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **256** (TP) | 3 (Confusion) | 1864 (FP Background) | 2123 |
| **Pred: helmet (1)** | 8 (Confusion) | **279** (TP) | 796 (FP Background) | 1083 |
| **Pred: Background (Missed)** | **4** (Missed FN) | **2** (Missed FN) | 0 | 6 |
| **Total Ground Truth** | **268** | **284** | 2660 | 3212 |

- **Ground Truth Verification**: Class 0 = 268 (expected 268), Class 1 = 284 (expected 284). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 95.52%, Class 1 = 98.24%, Macro Recall = 96.88%.

## EXP-COMBINED on Old Test (829 images, 1170 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **524** (TP) | 24 (Confusion) | 4098 (FP Background) | 4646 |
| **Pred: helmet (1)** | 60 (Confusion) | **536** (TP) | 5738 (FP Background) | 6334 |
| **Pred: Background (Missed)** | **16** (Missed FN) | **10** (Missed FN) | 0 | 26 |
| **Total Ground Truth** | **600** | **570** | 9836 | 11006 |

- **Ground Truth Verification**: Class 0 = 600 (expected 600), Class 1 = 570 (expected 570). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 87.33%, Class 1 = 94.04%, Macro Recall = 90.68%.

## EXP-COMBINED on Combined Test (1109 images, 1722 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **780** (TP) | 27 (Confusion) | 5962 (FP Background) | 6769 |
| **Pred: helmet (1)** | 68 (Confusion) | **815** (TP) | 6534 (FP Background) | 7417 |
| **Pred: Background (Missed)** | **20** (Missed FN) | **12** (Missed FN) | 0 | 32 |
| **Total Ground Truth** | **868** | **854** | 12496 | 14218 |

- **Ground Truth Verification**: Class 0 = 868 (expected 868), Class 1 = 854 (expected 854). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 89.86%, Class 1 = 95.43%, Macro Recall = 92.65%.

## EXP-NEW on New Test (280 images, 552 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **254** (TP) | 3 (Confusion) | 814 (FP Background) | 1071 |
| **Pred: helmet (1)** | 8 (Confusion) | **278** (TP) | 364 (FP Background) | 650 |
| **Pred: Background (Missed)** | **6** (Missed FN) | **3** (Missed FN) | 0 | 9 |
| **Total Ground Truth** | **268** | **284** | 1178 | 1730 |

- **Ground Truth Verification**: Class 0 = 268 (expected 268), Class 1 = 284 (expected 284). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 94.78%, Class 1 = 97.89%, Macro Recall = 96.33%.

## EXP-NEW on Old Test (829 images, 1170 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **102** (TP) | 42 (Confusion) | 1094 (FP Background) | 1238 |
| **Pred: helmet (1)** | 20 (Confusion) | **175** (TP) | 897 (FP Background) | 1092 |
| **Pred: Background (Missed)** | **478** (Missed FN) | **353** (Missed FN) | 0 | 831 |
| **Total Ground Truth** | **600** | **570** | 1991 | 3161 |

- **Ground Truth Verification**: Class 0 = 600 (expected 600), Class 1 = 570 (expected 570). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 17.00%, Class 1 = 30.70%, Macro Recall = 23.85%.

## EXP-NEW on Combined Test (1109 images, 1722 instances)

| Axis (Pred \ True) | True: no_helmet (0) | True: helmet (1) | True: background (FP) | Total Predictions |
|---|---:|---:|---:|---:|
| **Pred: no_helmet (0)** | **356** (TP) | 45 (Confusion) | 1908 (FP Background) | 2309 |
| **Pred: helmet (1)** | 28 (Confusion) | **453** (TP) | 1261 (FP Background) | 1742 |
| **Pred: Background (Missed)** | **484** (Missed FN) | **356** (Missed FN) | 0 | 840 |
| **Total Ground Truth** | **868** | **854** | 3169 | 4891 |

- **Ground Truth Verification**: Class 0 = 868 (expected 868), Class 1 = 854 (expected 854). Exact match: **True**.
- **Fixed-Threshold Recall (conf=0.25/0.001)**: Class 0 = 41.01%, Class 1 = 53.04%, Macro Recall = 47.03%.
