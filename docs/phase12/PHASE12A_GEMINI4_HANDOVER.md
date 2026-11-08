# PHASE 12A — HANDOVER BRIEF FOR GEMINI 4 ARGON

**To**: Gemini 4 Argon (Lead Research Architect & Implementation Engineer)  
**From**: Gemini 3.8 Flash (ML Infrastructure & Environment Engineer)  
**Project**: An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Handover Status**: **VERIFIED & OPERATIONAL (VERDICT: READY FOR GEMINI 4 ARGON)**  
**Target Hardware**: Apple MacBook Air M1 (8 GB RAM, 256 GB SSD, macOS arm64)  
**Python Runtime**: `/opt/homebrew/bin/python3.11`  
**Test Suite**: 28 / 28 tests passing (`/opt/homebrew/bin/python3.11 -m pytest tests/ -v`)  

---

## 1. Executive Briefing

The development environment and modular model infrastructure for the dual-camera research evaluation have been fully prepared and validated on the researcher's M1 Mac. 

All technical blockers, dependency conflicts, and environment ambiguities have been resolved. You can begin research architecture planning and implementation immediately without addressing system setup issues.

---

## 2. Verified System Architecture & Module Map

```text
research_helmet/
├── configs/
│   ├── environment.yaml            # Machine specs, Python 3.11, hardware acceleration (MPS)
│   ├── model_registry.yaml         # Formal registry of all models, weights, licenses, & devices
│   ├── ocr_config.yaml             # OCR thresholds, preprocessing, and text normalization regex
│   └── temporal_config.yaml        # Phase 10 temporal verification parameters (W=10, N_confirm=5)
├── src/
│   ├── models/                     # Standardized Model Adapters
│   │   ├── helmet_detector_adapter.py   # YOLOv8n detector wrapper (MPS & CPU, 21ms latency)
│   │   ├── plate_detector_adapter.py    # BasePlateDetectorAdapter & HeuristicPlateDetector
│   │   └── ocr_adapter.py               # EasyOCRAdapter (primary) & TesseractOCRAdapter (backup)
│   ├── video/                      # Stream Decoding & Frame Utilities
│   │   ├── video_reader.py         # OpenCV VideoReader with random access seeking
│   │   ├── frame_timestamp.py      # Frame-to-seconds & overlap calculation utilities
│   │   ├── frame_processor.py      # Bounding box & temporal state visualizer
│   │   └── video_pipeline.py       # Single-stream verification pipeline
│   ├── synchronization/            # Dual-Camera Synchronized Access
│   │   └── synchronized_reader.py  # SynchronizedVideoReader for the 25 benchmark pairs
│   └── temporal/                   # Preserved Phase 10 Temporal Engine (16 unit tests passing)
│       ├── confidence_aggregator.py
│       ├── rider_tracker.py
│       ├── state_machine.py
│       ├── temporal_engine.py
│       └── violation_decision.py
├── data/
│   └── dual_camera_test_v1/        # Frozen, audited 4K 60fps benchmark dataset
│       ├── front/                  # 25 front videos (pair_001_front.mp4 to pair_025_front.mp4)
│       ├── back/                   # 25 back videos (pair_001_back.mp4 to pair_025_back.mp4)
│       ├── metadata/               # Manifests & synchronization_metadata.json
│       ├── previews/               # 25 preview directories
│       └── unmatched_quarantine/   # 2 quarantined videos (unmatched_01_front_11, unmatched_02_front_16)
└── tests/                          # 28 passing unit tests
    ├── test_environment_readiness.py
    ├── test_model_adapters.py
    ├── test_synchronized_reader.py
    └── temporal/                   # 16 temporal verification unit tests
```

---

## 3. Available Checkpoints & Model Registry Summary

1. **Helmet Detector**:
   - **File**: `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt`
   - **SHA-256**: `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5`
   - **Classes**: `0 = no_helmet`, `1 = helmet`.
   - **Inference**: Verified functional via `HelmetDetectorAdapter`. Steady-state 4K inference takes **20.98 ms on MPS** (Apple Silicon GPU).
2. **OCR Engine (Primary)**:
   - **Engine**: `EasyOCR` (CRAFT detection + CRNN recognition).
   - **Weights**: Pretrained official weights cached in `~/.EasyOCR/model/` (`craft_mlt_25k.pth` [79MB], `english_g2.pth` [14MB]).
   - **Inference**: Verified functional via `EasyOCRAdapter`. CPU inference on plate patch takes ~150 ms.
3. **Plate Detector**:
   - **Active Implementation**: `HeuristicPlateDetector` (extracts lower central ROI of rear motorcycle track).
   - **Extensibility**: `BasePlateDetectorAdapter` contract is ready if you choose to plug in a pretrained YOLO plate detector.

---

## 4. Key Researcher Decision: Zero Custom OCR Training

The researcher has explicitly decided:
> **We will NOT develop, train, or fine-tune a custom OCR or number plate recognition model.**

### Architectural Implication:
- Rely strictly on existing open-source pretrained solutions.
- Do NOT search the entire 4K frame with OCR. Use a **two-stage pipeline**:
  1. Detect motorcycle / rider track (Back camera view).
  2. Crop license plate ROI (`HeuristicPlateDetector` or pretrained plate model).
  3. Run `EasyOCRAdapter` on the small crop.

---

## 5. Synchronized Benchmark Video Access

Use `src/synchronization/synchronized_reader.py`:

```python
from src.synchronization.synchronized_reader import SynchronizedVideoReader

with SynchronizedVideoReader("pair_001") as reader:
    info = reader.sync_info
    print(f"Offset: {info.global_offset_sec}s | Overlap: {info.overlap_duration_sec}s")
    
    # Extract synchronized frame pair at time t_back
    frame_back, frame_front, status = reader.get_synchronized_pair(t_back_sec=1.55)
    
    # Or iterate across the entire common overlap interval
    for fb, ff, status in reader.iter_synchronized_frames(step_frames=5):
        # Process synchronized frames
        pass
```

### Audited Timing Rules (from Phase 11.6):
- Reference timeline is **Back Camera** ($t_{\text{back}}$).
- Front camera timestamp is:
  $$t_{\text{front}} = t_{\text{back}} - \Delta t_{\text{offset}}$$
- 21 pairs have sub-frame residual uncertainty ($\le \pm 10.0\text{ ms}$).
- 4 pairs (`pair_004`, `pair_005`, `pair_008`, `pair_015`) have near-frame residual uncertainty ($\pm 18.0\text{--}38.0\text{ ms}$, $1.08\text{--}2.28\text{ frames}$).

---

## 6. Outstanding Architectural Decisions for Gemini 4 Argon

You have full authority to plan and execute the research methodology. Key decisions for your phase:

1. **Cross-Camera Track Association Strategy**:
   - How to associate the front-camera track (oncoming rider face/chest) with the back-camera track (receding rider back/plate).
   - Options: Temporal transit window matching, color histogram / appearance embedding matching, motion trajectory extrapolation, or sequence-level pass-by indexing.
2. **License Plate Localization Choice**:
   - Whether to use `HeuristicPlateDetector` (which requires zero additional weights and runs instantly) or download a public pretrained YOLOv8n license-plate checkpoint.
3. **Dual-Camera Evidence Fusion Logic**:
   - How evidence from front camera (e.g. front helmet view) and back camera (e.g. rear helmet view + license plate) are mathematically combined in the temporal verification engine.
4. **Evaluation Protocol on Benchmark**:
   - Frame sampling rate (e.g. processing every 1st, 3rd, or 5th frame for real-time throughput on M1).
   - Metrics to compute: Dual-camera precision, recall, citation latching accuracy, plate recognition accuracy, and end-to-end processing latency.

---

## 7. Execution Commands for Your Work

```bash
# Run the complete test suite
/opt/homebrew/bin/python3.11 -m pytest tests/ -v

# Run Python scripts using the verified environment
/opt/homebrew/bin/python3.11 <script_path>.py
```

The codebase is clean, tested, documented, and ready for your architectural leadership. Good luck!
