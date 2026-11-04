# Dual-Camera Helmet Violation Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducibility: Deterministic](https://img.shields.io/badge/Reproducibility-Multi--Seed%20(42,%20123,%202026)-green.svg)]()

An integrated, context-aware computer vision framework investigating **motorcycle helmet violation detection across heterogeneous camera domains** (fixed-angle overhead CCTV vs. mobile street-level views) with **downstream confidence-aware temporal verification**.

---

## 1. Research Motivation & Core Problems

Automated vision-based motorcycle helmet violation enforcement is a critical Intelligent Transportation System (ITS) intervention for reducing urban traffic fatalities. However, deploying computer vision models in real roadside surveillance reveals two fundamental scientific bottlenecks:

### 1.1 Catastrophic Cross-Domain Collapse
Detectors trained exclusively on high-resolution, multi-angle mobile street photography experience catastrophic failure when evaluated on low-resolution, overhead CCTV surveillance ($\text{mAP}@0.50$ plummets from $96.15\%$ down to $8.08\%$, an $88\text{ pp}$ drop). Conversely, models trained solely on fixed-angle CCTV fail to generalize across diverse viewing angles and complex urban lighting.

### 1.2 Temporal Classification Jitter & False Alert Cascades
Single-frame object detectors evaluate each video frame independently. A single misclassified frame—caused by temporary motion blur, visor reflections, or severe head angles—triggers immediate false roadside violation citations in naïve systems. Falsely penalizing compliant citizens destroys public trust and creates severe administrative liabilities.

---

## 2. System Architecture & Methodology

```mermaid
flowchart TD
    subgraph Data["1. Multi-Domain Sequence-Isolated Corpus"]
        D1["Fixed-Angle CCTV (5,547 imgs / 169 seqs)"]
        D2["Audited Mobile Street (1,841 imgs / 3 audit passes)"]
        D1 & D2 --> D3["Unified Combined Corpus (5,151 train / 1,128 val / 1,109 test)"]
    end

    subgraph Detector["2. Deep Object Detector (YOLOv8n)"]
        D3 --> Y["YOLOv8n (AdamW, imgsz=640, lr=0.001667)"]
        Y --> CK["Frozen Checkpoints (Seeds 42, 123, 2026)"]
    end

    subgraph Temporal["3. Downstream Temporal Verification Framework"]
        CK --> TR["Spatial IoU RiderTracker"]
        TR --> AG["Confidence Aggregator (Voting / Weighted / EMA)"]
        AG --> SM["6-State Finite State Machine (UNKNOWN, SUSPECTED, CONFIRMED)"]
        SM --> HY["Dual-Threshold Hysteresis (tau_confirm=0.70, tau_clear=0.40)"]
        HY --> AL["Authoritative Violation Alert Dispatcher"]
    end
```

---

## 3. Dataset Integrity & Zero-Leakage Protocol

To ensure publication-grade scientific validity, the framework utilizes two strictly partitioned datasets:

| Population | Source Domain | Total Images | Train | Validation | Test | Quality Audit & Isolation Protocol |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Old CCTV** | Overhead Infrastructure CCTV | 5,547 | 3,866 | 852 | 829 | 169 atomic video sequences isolated; **0.0% sequence leakage** |
| **New Mobile** | Handheld & Street Photography | 1,841 | 1,285 | 276 | 280 | 3-pass forensic audit; 392 box corrections, 28 label inversions fixed |
| **Combined** | Multi-Domain Union | 7,388 | 5,151 | 1,128 | 1,109 | Unified leak-free cross-domain benchmark |

---

## 4. Benchmark Performance & Evaluation

All models are independently evaluated across three separate held-out test populations (audited values from Phase 8):

| Experiment | Training Population | New Mobile Test (280) mAP@0.50 | Old CCTV Test (829) mAP@0.50 | Combined Test (1,109) mAP@0.50 | Overall F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-002-FIXED** | CCTV Only | — | 66.94% | — | 73.28% | Sequence-Isolated Baseline |
| **EXP-NEW** | Mobile Only | **96.15%** | 8.08% *(Collapse)* | 39.47% | 48.56% | Severe Domain Collapse |
| **EXP-COMBINED** | Multi-Domain Union | **96.02%** | **75.17%** | **84.37%** | **80.99%** | **Superior Across All Domains** |

### Key Research Insights:
1. **Domain Collapse Resolution:** EXP-COMBINED recovers cross-domain CCTV performance from $8.08\%$ to **$75.17\%$** ($+67.09\text{ pp}$), while strictly outperforming the CCTV-only baseline by **$+8.23\text{ pp}$** ($75.17\%$ vs $66.94\%$).
2. **Zero In-Domain Degradation:** EXP-COMBINED preserves mobile street accuracy at **$96.02\%$** (within $0.13\text{ pp}$ of the specialized model).

---

## 5. Confidence-Aware Temporal Verification Framework

The Phase 10 temporal verification pipeline resolves single-frame classification jitter:
- **6-State FSM:** `UNKNOWN` $\to$ `HELMET_CONFIRMED` $\to$ `NO_HELMET_SUSPECTED` $\to$ `NO_HELMET_CONFIRMED` $\to$ `TEMPORARILY_OCCLUDED` $\to$ `TRACK_LOST`.
- **Hysteresis Thresholds:** Confirmation requires $\ge 0.70$ confidence sustained across $N_{confirm} \ge 5$ frames ($167\text{ ms}$). De-escalation strictly requires confidence $< 0.40$.
- **Aggregators:** Evaluates Simple Voting, Confidence-Weighted Aggregation, and Exponential Moving Average (EMA).
- **False Alert Suppression:** Demonstrated **100% suppression** of spurious 2-frame false positive violation bursts on compliant helmeted riders.

---

## 6. Installation & Reproduction

### 6.1 Setup Local Environment
```bash
git clone https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection.git
cd Dual-Camera-Helmet-Violation-Detection

python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision pyyaml pandas numpy pillow matplotlib seaborn
```

### 6.2 Run Automated Test Suite
```bash
# Execute all temporal verification unit tests
python3 -m unittest discover -s tests/temporal -p "test_*.py" -v
```

### 6.3 Run Interactive Demonstration
```bash
# Demonstrate violation detection and hysteresis gating
python3 scripts/demo_temporal_verification.py --scenario stable_violation

# Demonstrate false positive noise burst suppression
python3 scripts/demo_temporal_verification.py --scenario fp_burst_on_compliant

# Run full 21-experiment benchmark across all aggregators
python3 scripts/demo_temporal_verification.py --export-benchmark
```

---

## 7. Project Documentation Index

- [`docs/phase10/PHASE10_SYSTEM_DESIGN.md`](docs/phase10/PHASE10_SYSTEM_DESIGN.md): Architectural specification of the temporal state machine, aggregators, and tracker.
- [`docs/phase10/NEXT_EXPERIMENTS_AND_ABLATIONS.md`](docs/phase10/NEXT_EXPERIMENTS_AND_ABLATIONS.md): Future controlled ablation studies and sensitivity protocols.
- [`docs/REPRODUCIBILITY_GUIDE.md`](docs/REPRODUCIBILITY_GUIDE.md): End-to-end instructions for dataset preparation, Kaggle execution, and evaluation.
- [`docs/research/PAPER_OUTLINE.md`](docs/research/PAPER_OUTLINE.md): Preliminary research manuscript outline for IEEE submission.
- [`results/phase8_independent_audit/PHASE8_INDEPENDENT_EVALUATION_AUDIT.md`](results/phase8_independent_audit/PHASE8_INDEPENDENT_EVALUATION_AUDIT.md): Bit-exact independent evaluation and metric reconciliation report.
- [`results/phase10_temporal/PHASE10_TEMPORAL_VERIFICATION_REPORT.md`](results/phase10_temporal/PHASE10_TEMPORAL_VERIFICATION_REPORT.md): Comprehensive validation report for temporal verification.

---

## 8. License & Research Attribution

This project is licensed under the MIT License. If you utilize this dataset, splitting methodology, or temporal verification framework in your research, please cite:

```bibtex
@article{uddin2026dualcamera,
  title={An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification},
  author={Uddin, Mohammad Sayem and collaborators},
  journal={Working Paper},
  year={2026}
}
```
