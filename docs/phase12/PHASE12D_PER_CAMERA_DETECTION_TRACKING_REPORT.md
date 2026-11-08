# PHASE 12D — PER-CAMERA HIERARCHICAL DETECTION, RIDER–MOTORCYCLE COUPLING & TRACKING REPORT

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection` (`main` branch)  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Scope:** Implementation, unit/synthetic geometry verification, and controlled real-video smoke benchmarking of the independent Front and Back per-camera detection, rider–motorcycle coupling, and hierarchical tracking pipelines.

---

## 1. Pre-Implementation Audit & Model Verification

Before writing Phase 12D code, both model checkpoints and the runtime environment were forensically verified:

| Model / Component | Path | Size (Bytes) | Verified SHA-256 | Loaded Class Mapping |
| :--- | :--- | :---: | :--- | :--- |
| **Vehicle & Person Detector** | [`yolov8n.pt`](file:///Users/sayemuddin/Desktop/research_helmet/yolov8n.pt) | `6,549,796` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | 80 COCO classes; restricted to `{0: 'person', 3: 'motorcycle'}` |
| **Phase 9 Helmet Detector** | [`results/phase9_multiseed/exp_combined/seed_42/weights/best.pt`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase9_multiseed/exp_combined/seed_42/weights/best.pt) | `6,506,922` | `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5` | Strictly 2 classes: `{0: 'no_helmet', 1: 'helmet'}` |
| **Runtime Environment** | `/opt/homebrew/bin/python3.11` | — | PyTorch `2.14.1` (`mps` enabled), Ultralytics `8.4.201`, OpenCV `5.0.0`, SciPy `1.17.1` | Apple M1 (8 GB Unified RAM) |

---

## 2. Implemented Architecture & Module Inventory

| Module Path | Responsibility |
| :--- | :--- |
| [`configs/tracking_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/tracking_config.yaml) | Externalized configuration of all detection thresholds, `MotoRiderCoupler` geometric parameters/weights, ByteTrack thresholds, and Phase 10 temporal state-machine parameters. |
| [`src/models/vehicle_detector_adapter.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/models/vehicle_detector_adapter.py) | [`VehiclePersonDetectorAdapter`](file:///Users/sayemuddin/Desktop/research_helmet/src/models/vehicle_detector_adapter.py#L22-L135): Wraps `yolov8n.pt`, verifies `{0: 'person', 3: 'motorcycle'}` against `model.names` on load, and filters predictions to classes `[0, 3]`. |
| [`src/detection/camera_detector.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py) | [`StructuredDetection`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py#L21-L82), [`FrameDetectionBundle`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py#L85-L116), [`SharedDetectorRegistry`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py#L119-L148), and [`PerCameraDetector`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py#L151-L246). Enforces singleton model sharing across cameras and strict separation between `motorcycle`, `person`, and `head` detections. |
| [`src/tracking/moto_rider_coupler.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py) | [`MotoRiderCoupler`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L168-L450): Links rider heads and persons to parent motorcycles across 1–3 rider slots (`driver`, `pillion_1`, `pillion_2`), abstains on ambiguous adjacent motorcycles (`AMBIGUOUS_ADJACENT_MOTORCYCLES`), and outputs [`OrphanRiderObservation`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L121-L147) when no reliable motorcycle association exists. |
| [`src/tracking/kalman_bytetrack.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/kalman_bytetrack.py) | [`KalmanBoxFilter`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/kalman_bytetrack.py#L46-L142), [`TrackLifecycleState`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/kalman_bytetrack.py#L17-L23) (`TENTATIVE`, `CONFIRMED`, `LOST`, `RECOVERED`, `TERMINATED`), and [`two_stage_bytetrack_assign`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/kalman_bytetrack.py#L176-L240). |
| [`src/tracking/camera_tracker.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py) | [`RiderSubTrack`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py#L46-L157), [`MotoTrack`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py#L160-L368), [`OrphanRiderTrack`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py#L371-L473), and [`CameraTrackingPipeline`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py#L476-L699). Connects each rider slot to the untouched Phase 10 `Track`, `BaseConfidenceAggregator`, `TemporalStateMachine`, and `ViolationDecisionEngine`, with disjoint `front_*` and `back_*` track ID namespaces. |
| [`tests/test_phase12d_tracking.py`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12d_tracking.py) | 9 comprehensive unit, synthetic-geometry, and real-video PTS-gate integration tests covering all 13 mandatory test categories. |
| [`scripts/phase12d_smoke_benchmark.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/phase12d_smoke_benchmark.py) | Controlled real-video smoke test and per-camera latency/memory benchmark across `pair_001`, `pair_013` (VFR), `pair_015` (30 fps Back), and `pair_019`. |

---

## 3. Separation of Synthetic Test Verification vs. Real-Video Smoke Observations

> [!IMPORTANT]
> In accordance with Section 7 of the Phase 12D specification, **synthetic unit/geometry test results are strictly separated from unannotated real-video smoke-test observations**. Because `data/dual_camera_test_v1/` does not yet have human-verified bounding-box/track ground truth (scheduled for Phase 12H), no detection precision, recall, mAP, coupling accuracy, or MOTA/IDF1 claims are made on the real videos.

### 3.1 Synthetic Geometry & Tracking Verification (`tests/test_phase12d_tracking.py`)
All 9 test functions in [`tests/test_phase12d_tracking.py`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12d_tracking.py) (`44/44` total repository tests) passed in `31.46 s`:
1. **Model Metadata & SHA-256 Verification** (`test_model_class_mappings_and_sha256_verification` & `test_per_camera_detector_smoke_on_blank_frame`): Verified exact class tables and singleton sharing between `front` and `back` detectors.
2. **Synthetic 1-, 2-, and 3-Rider Coupling & Slot Ordering** (`test_synthetic_driver_plus_pillion_and_three_rider_coupling`): Verified `driver`, `pillion_1`, and `pillion_2` ordering when vertical head separation $\ge 8.0\text{ px}$, and verified automatic fallback to `ordering_confident = False` (`"ambiguous_slot_0"`, `"ambiguous_slot_1"`) when two heads on the same motorcycle have vertical separation $< 8.0\text{ px}$.
3. **Synthetic Adjacent Motorcycle Separation** (`test_synthetic_adjacent_motorcycle_rider_separation`): Verified two side-by-side motorcycles retain exclusive 1-to-1 rider assignments without cross-linking.
4. **Synthetic Orphan Fallback & Ambiguous Coupling Abstention** (`test_synthetic_orphan_rider_fallback_and_ambiguous_abstention`): Verified `NO_MOTORCYCLE_IN_FRAME` when no motorcycle is present, and verified `AMBIGUOUS_ADJACENT_MOTORCYCLES` abstention when a rider head sits equidistant between two overlapping motorcycles with margin $< 0.08$.
5. **Synthetic Kalman Velocity & Two-Stage ByteTrack Recovery** (`test_kalman_filter_prediction_and_two_stage_bytetrack_matching` & `test_temporary_occlusion_and_track_lifecycle_recovery`): Verified constant-velocity projection and full 5-state lifecycle transitions (`TENTATIVE -> CONFIRMED -> LOST -> RECOVERED` via a `0.25` low-confidence Stage 2 detection `-> TERMINATED`).
6. **Synthetic Multi-Rider Independent Phase 10 States & Cooldown** (`test_multiple_riders_independent_helmet_states_and_phase10_cooldown`): Verified that on a single motorcycle (`back_moto_0001`), Slot 0 (`driver` wearing helmet) reaches `HELMET_CONFIRMED` (`0` alerts) while Slot 1 (`pillion_1` without helmet) independently reaches `NO_HELMET_CONFIRMED` (`1` alert, with cooldown preventing duplicate alerts and hysteresis resisting a single-frame false helmet flip).

### 3.2 Real-Video Smoke Benchmark Observations (`results/phase12d/`)
Executed via [`scripts/phase12d_smoke_benchmark.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/phase12d_smoke_benchmark.py) over a 30-tick (`0.50 s` at `60 Hz`) window near the overlap midpoint of `pair_001`, `pair_013` (VFR), `pair_015` (30 fps Back), and `pair_019` on Apple M1 MPS (`device = "mps"`):

| Pair ID | Camera | 60Hz Ticks | Unique Frames Updated | Dup Ticks Skipped | Vehicle YOLO Mean (ms) | Helmet YOLO Mean (ms) | Combined Mean / P95 (ms) | Moto Tracks | Rider SubTracks | Orphan Tracks | Stage-2 Low-Conf Recoveries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pair_001` | `front` | 30 | `30` | `0` | `20.97` | `13.46` | `34.43 / 36.36` | `1` | `1` | `0` | `0` |
| `pair_001` | `back` | 30 | `30` | `0` | `13.55` | `11.39` | `24.94 / 28.56` | `0` | `0` | `0` | `0` |
| `pair_013` | `front` | 30 | `29` | `1` | `21.05` | `14.12` | `35.17 / 44.48` | `2` | `1` | `0` | **`12`** |
| `pair_013` | `back` *(VFR)* | 30 | **`15`** | **`15`** | `32.45` | `12.19` | `44.64 / 120.06` | `0` | `0` | `0` | `0` |
| `pair_015` | `front` | 30 | `30` | `0` | `19.23` | `12.82` | `32.05 / 32.54` | `5` | `2` | `0` | **`5`** |
| `pair_015` | `back` *(30fps)* | 30 | **`15`** | **`15`** | `12.12` | `10.77` | `22.90 / 27.80` | `0` | `0` | `0` | `0` |
| `pair_019` | `front` | 30 | `30` | `0` | `20.55` | `13.38` | `33.92 / 63.98` | `1` | `1` | `0` | `0` |
| `pair_019` | `back` | 30 | `30` | `0` | `13.61` | `12.17` | `25.78 / 32.27` | `2` | `1` | **`3`** | `0` |

#### Key Real-Video Qualitative Findings:
1. **Verified VFR and 30 fps Duplicate-Frame Suppression**: On `pair_013_back` (VFR 30 fps segment) and `pair_015_back` (30 fps CFR), exactly **`15` of the `30` canonical 60 Hz ticks were executed** and **`15` duplicate ticks were skipped** (`duplicate_ticks_skipped = 15`), cutting redundant Back-camera GPU compute by `50%` and preserving accurate Phase 10 frame counts.
2. **Stage-2 Low-Confidence ByteTrack Recovery in Action**: On `pair_013_front` (`front_moto_0002`, `12` Stage-2 matches) and `pair_015_front` (`front_moto_0005`, `5` Stage-2 matches), partially occluded/distant motorcycles whose COCO confidence dipped between `0.18` and `0.45` maintained unbroken `CONFIRMED` trajectories instead of fragmenting.
3. **OrphanRiderTrack Rescue on Distant Receding Motorcycles (`pair_019_back`)**: In `pair_019_back`, as a receding motorcycle traveled far down the corridor, COCO `yolov8n.pt` ceased detecting the small motorcycle chassis while Phase 9 `best.pt` continued detecting the rider's head across `23` frames. `MotoRiderCoupler` routed those detections to `back_orphan_0001` (`hits = 23`, `aggregated_confidence = {'helmet': 0.6989, 'no_helmet': 0.3011}`), preventing data loss.
4. **Annotated Visual Previews**: Saved in [`results/phase12d/previews/`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12d/previews/) (`pair_001_phase12d_preview.jpg`, `pair_013_phase12d_preview.jpg`, `pair_015_phase12d_preview.jpg`, `pair_019_phase12d_preview.jpg`).

---

## 4. Apple M1 Hardware Resource Measurements

- **Device Used**: Apple Silicon MPS (`device = "mps"`), with shared singleton detectors via `SharedDetectorRegistry`.
- **Mean Per-Camera Inference Latency**:
  - `yolov8n.pt` (COCO Vehicle/Person): `12.12–32.45 ms` per unique frame.
  - `best.pt` (Phase 9 Helmet/No-Helmet): `10.77–14.12 ms` per unique frame.
  - Combined Dual-Model Per-Camera Latency: **`22.90–44.64 ms`** per unique frame.
- **Peak Resident Memory (RSS)**: **`974.69 MB`** across the 4-pair 4K benchmark (`47.6%` of the `2,048 MB` ceiling, leaving `> 6 GB` free on the 8 GB Apple M1 machine).

---

## 5. Known Limitations, Failure Cases & Unresolved Research Questions

1. **Temporal Transit Lag Between Front and Back Views**: In the 0.5-second midpoint slice of `pair_001`, `pair_013`, and `pair_015`, the motorcycle is actively passing through the oncoming Front camera (`front_moto_0001`), whereas it enters or exits the receding Back camera at a displaced timestamp ($t_{\text{back}} = t_{\text{front}} + \Delta \tau_{\text{transit}}$). This confirms that cross-camera matching in Phase 12E must match *trajectories across a sliding transit window* rather than matching simultaneous single-tick frames.
2. **Distant Motorcycle Chassis Missed by Generic COCO `yolov8n.pt`**: When a motorcycle is very distant ($< 40\text{ px}$ wide), `yolov8n.pt` sometimes misses the `motorcycle` class even when the domain-trained Phase 9 `best.pt` detects the head. While `OrphanRiderTrack` preserves the rider's temporal helmet state within the camera, an orphan head track lacks a parent motorcycle bounding box for chassis color extraction or rear license plate cropping until the motorcycle approaches closer.
3. **Pillion Head Occlusion Along the Camera Line of Sight**: In `pair_015_front`, `front_moto_0001_rider_s1` (`pillion_1`) is intermittently occluded behind the driver's helmet (`hits = 1`, `time_since_update = 9`, `TEMPORARILY_OCCLUDED`), whereas the receding Back camera has a clear view of the pillion's back/head—directly motivating Phase 12E–12G's opposing-viewpoint evidence fusion.
