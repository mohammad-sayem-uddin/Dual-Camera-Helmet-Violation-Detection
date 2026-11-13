# Research Reproducibility Guide

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection` (`main`)

---

## 1. Environment & Python Runtime Notes

### Tested Local Environment (Apple Silicon macOS)
- **Active Working Interpreter:** `/opt/homebrew/opt/python@3.11/bin/python3.11` (`Python 3.11.15`, `PyTorch 2.14.1`, `Ultralytics 8.4.201`, `OpenCV 5.0.0`, `NumPy 2.4.6`, `Pandas 3.1.0`, `SciPy`, `Matplotlib 3.11.2`, `PyYAML 6.0.3`, `pytest 9.1.1`, `MPS Available: True`).
- **Node.js (Frontend JS Syntax Verification):** `node v26.9.0`.
- **Known Broken Virtualenv:** A legacy `.venv/` directory (if present locally) fails PyTorch/pytest imports (`ModuleNotFoundError: No module named 'torch.torch_version'`). Always run scripts with Python 3.11 (`/opt/homebrew/opt/python@3.11/bin/python3.11` or a fresh Python 3.11 venv).

### Verify Python Environment
```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/utils/check_env.py
```

---

## 2. One-Command CPU-Safe Full Research Reproduction

The unified reproduction runner executes all non-destructive CPU verification steps into an isolated temporary/specified output directory (preserving all frozen artifacts in `data/` and `results/`):

```bash
bash scripts/reproducibility/run_full_research_reproduction.sh /tmp/research_repro_run
```

What [`run_full_research_reproduction.sh`](../scripts/reproducibility/run_full_research_reproduction.sh) verifies end-to-end:
1. Environment & package imports (`torch`, `ultralytics`, `cv2`, `numpy`, `pandas`, `scipy`, `yaml`, `pytest`).
2. Cryptographic SHA-256 verification:
   - `39 / 39` base annotation files (`data/dual_camera_test_v1/annotations/checksums/annotation_checksums.sha256`).
   - `43 / 43` Stage E `v1.0` release files (`release_checksums.sha256`).
   - `8 / 8` frozen Stage K outputs (`results/phase12_integrated/checksums.sha256`).
   - Model checkpoints (`yolov8n.pt`, `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt`, `results/exp_combined/weights/best.pt`).
   - Integrity of unexecuted `v1.1` blind workqueue (`executed_human_units == 0`, `PENDING_HUMAN_REVIEW`).
3. Stage K evaluation replay (`evaluate.py --tiou-mode legacy` & `--tiou-mode fixed`).
4. GAP-B offline replay (`evaluate_gapb_redesign.py` via `GAPB_EVAL_OUT_DIR`).
5. GAP-C tracking & consensus ablation (`evaluate_gapc_tracking.py`).
6. TASK-10 bounded real-video Cue C 720p/4K verification (`evaluate_task10_real_video_cue_c.py`).
7. TASK-11 10-arm dual-camera Bayesian fusion & single-vs-dual ablation (`evaluate_task11_fusion.py`).
8. TASK-12 tIoU boundary & `v1.1` non-destruction audit (`evaluate_task12_audit.py`).
9. Frontend JavaScript check (`node --check src/review_dashboard/static/app.js`).
10. Full test suite (`pytest -q tests/` — including synthetic pipeline, temporal FSM, review dashboard, 24 real-video integration tests, and D1–D6 defect integration checks).

### Intentionally Excluded Stages
- **Kaggle GPU Retraining (Phases 7 & 9):** `EXP-NEW` and `EXP-COMBINED` (seeds `42, 123, 2026`) checkpoints are frozen and SHA-256 verified (`results/phase9_multiseed/*_checkpoints_sha256.json`). Training configs match `imgsz=640`, `batch=32`, `AdamW`, `lr0=0.001667`, `max_epochs=50`, `patience=20`.
- **Stage J 50-Stream Full Inference Regeneration:** Offline Phase 12 scripts consume frozen Stage J JSONLs (`results/phase12_stagej/`) plus bounded decoded video streams (`TASK-10`, `test_phase12_pipeline_real_video.py`).
- **Blind Human Annotation (`v1.1`):** Stopped at `0 / 92` units per PI directive; `dual_camera_gt_v1.0_ai_assisted_audited` is the accepted evaluation reference.

---

## 3. Running Individual Subsystems & CLI Tools

### 3.1 Single Pair End-to-End Dual-Camera Pipeline
```bash
PYTHONPATH=. /opt/homebrew/opt/python@3.11/bin/python3.11 src/pipeline/dual_camera_pipeline.py \
  --pairs pair_019 \
  --fusion-mode dual_fused \
  --device mps \
  --output-root /tmp/pair_019_dual_fused
```
Supported `--fusion-mode` values: `legacy` *(default)*, `dual_fused`, `front_only`, `back_only`.

### 3.2 Phase 10 Temporal Verification Benchmark
```bash
PYTHONPATH=. /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/demo_temporal_verification.py --scenario fp_burst_on_compliant
```

### 3.3 Blind Human Review Dashboard (Preserved Infrastructure Template)
```bash
./scripts/start_gt_review_dashboard.sh
# Serves http://127.0.0.1:8765/ with strict role isolation for v1.1 continuation
```

### 3.4 Individual Test Execution
```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -q tests/
```
