# Research Reproducibility Guide

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection.git`  
**Target Environment:** Local macOS / Linux (Development & Temporal Pipeline) and Kaggle NVIDIA Tesla T4 (Model Training & Benchmark Evaluation)  

---

## 1. Repository Structure & Prerequisites

### 1.1 Local Environment Setup
Clone the repository and install core dependencies:

```bash
git clone https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection.git
cd Dual-Camera-Helmet-Violation-Detection

# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate

# Install essential computer vision and machine learning packages
pip install ultralytics torch torchvision pyyaml pandas numpy pillow matplotlib seaborn
```

---

## 2. Dataset Overview & Organization

The research utilizes two distinct dataset populations:
1. **Historical CCTV Dataset (`old-dataset-v1-fixed`):**
   - 5,547 total fixed-angle surveillance images from roadside infrastructure.
   - Sequence-isolated split: Train (3,866 images), Val (852 images), Test (829 images).
   - Strict 0.0% video sequence leakage across splits.
2. **Mobile Street Expansion Dataset (`helmet-dataset-expansion-v2`):**
   - 1,841 multi-angle street-level handheld/dashcam images.
   - Tri-fold audited split: Train (1,285 images), Val (276 images), Test (280 images).
   - Class-balanced and audit-verified (0 label inversions, 0 corrupt bounding boxes).
3. **Combined Multi-Domain Dataset:**
   - Unified union: Train (5,151 images), Val (1,128 images), Held-Out Test (1,109 images).

### 2.1 Dataset Configuration Files
Standard YOLO dataset YAML files are located in `configs/`:
- `configs/dataset_new.yaml`: Points to mobile expansion splits.
- `configs/dataset_old.yaml`: Points to historical CCTV splits.
- `configs/dataset_combined.yaml`: Points to merged cross-domain splits.

---

## 3. Kaggle Training Protocol

All models are trained with official COCO-pretrained `yolov8n.pt` initialization on NVIDIA Tesla T4 GPUs (16 GB VRAM).

### 3.1 Hyperparameter Specifications
Every experiment adheres to the recovered Phase 7 configuration:
- **Architecture:** YOLOv8n (`yolov8n.pt`)
- **Resolution (`imgsz`):** 640x640
- **Batch Size:** 32
- **Optimizer:** AdamW
- **Initial Learning Rate (`lr0`):** 0.001667
- **Weight Decay:** 0.0005
- **Momentum:** 0.9
- **Max Epochs:** 50
- **Early Stopping Patience:** 20 epochs
- **Random Seeds:** 42, 123, 2026
- **Deterministic:** Enabled (`deterministic=True`)

### 3.2 Pushing Notebooks to Kaggle
Notebooks and kernel metadata are located in `notebooks/`:
```bash
# Push EXP-NEW multi-seed kernel
kaggle kernels push -p notebooks/phase9_multiseed/exp_new

# Push EXP-COMBINED multi-seed kernel
kaggle kernels push -p notebooks/phase9_multiseed/exp_comb
```

---

## 4. Multi-Population Evaluation Protocol

To prevent evaluation bias and measure cross-domain generalization, every trained checkpoint is independently evaluated across three held-out test populations:
1. **New Mobile Test (280 images, 552 instances):** In-domain for EXP-NEW; cross-domain test for CCTV.
2. **Old CCTV Test (829 images, 1,170 instances):** In-domain for historical CCTV; out-of-domain cross-camera test for EXP-NEW.
3. **Combined Held-Out Test (1,109 images, 1,722 instances):** Global benchmark evaluating total real-world traffic coverage.

Run local evaluation using:
```bash
python3 scripts/evaluation/evaluate_yolo.py \
    --weights results/exp_combined/weights/best.pt \
    --data configs/dataset_combined.yaml \
    --split test \
    --imgsz 640 \
    --batch 32 \
    --device auto \
    --name EXP-COMBINED-TEST
```

---

## 5. Temporal Verification Pipeline Execution

The confidence-aware temporal verification engine operates downstream of detection:

### 5.1 Configuration
Inspect and edit `configs/temporal_config.yaml`:
```yaml
detection:
  confidence_threshold: 0.25
  nms_iou_threshold: 0.45

tracking:
  iou_match_threshold: 0.30
  max_track_age_frames: 15

aggregation:
  method: "confidence_weighted" # options: simple_voting, confidence_weighted, ema
  sliding_window_size: 10

state_machine:
  confirm_frames: 5
  confirm_threshold: 0.70
  clear_threshold: 0.40
  helmet_threshold: 0.65
```

### 5.2 Running the Test Suite
Verify all state machine transitions, aggregators, tracking, and hysteresis rules:
```bash
python3 -m unittest discover -s tests/temporal -p "test_*.py" -v
```

### 5.3 Running the Interactive Demonstration
Execute the command-line telemetry demonstration across synthetic traffic scenarios:
```bash
# Run stable violation scenario
python3 scripts/demo_temporal_verification.py --scenario stable_violation

# Run false positive noise burst suppression scenario
python3 scripts/demo_temporal_verification.py --scenario fp_burst_on_compliant

# Run full 21-experiment benchmark across all aggregators
python3 scripts/demo_temporal_verification.py --export-benchmark
```
