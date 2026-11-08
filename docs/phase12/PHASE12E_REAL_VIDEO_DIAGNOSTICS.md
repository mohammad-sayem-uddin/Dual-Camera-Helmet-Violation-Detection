# PHASE 12E — REAL-VIDEO OPERATIONAL DIAGNOSTICS REPORT

**Project:** *An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification*  
**Phase:** Phase 12E — Cross-Camera Tracklet Association  
**Diagnostic Script:** [`scripts/phase12e_operational_diagnostics.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/phase12e_operational_diagnostics.py)  
**Result Artifacts:**
- [`results/phase12e/phase12e_diagnostics_summary.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12e/phase12e_diagnostics_summary.json)
- [`results/phase12e/tracklet_association_audit.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12e/tracklet_association_audit.csv)
- [`results/phase12e/cue_score_distributions.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12e/cue_score_distributions.csv)
- [`results/phase12e/association_runtime_benchmark.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12e/association_runtime_benchmark.csv)

---

## 1. Scientific Caveat

These diagnostics evaluate operational behavior, candidate generation, multi-cue availability, weight renormalization, conservative abstention, and computational overhead on four real synchronized video pairs (`pair_001`, `pair_013`, `pair_015`, `pair_019`). **Because no human cross-camera identity ground truth exists yet, these diagnostics do not measure true cross-camera association accuracy (IDF1, HOTA, precision, or recall).**

---

## 2. Evaluated Video Pairs & Synchronization Provenance

Each pair was evaluated over a `1.60 s` (`96` canonical 60 Hz ticks) active transit window inside the common overlap interval using [`SynchronizedVideoReader`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py), [`PerCameraDetector`](file:///Users/sayemuddin/Desktop/research_helmet/src/detection/camera_detector.py), [`CameraTrackingPipeline`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py), and [`CrossCameraAssociationEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/cross_camera_association.py):

| Pair ID | Regime | Canonical Ticks | Front Updates | Back Updates | Back Repeated Skipped | $\epsilon_{\text{sync}}$ (ms) | Mean $\delta_{\text{PTS,front}}$ (ms) | Mean $\delta_{\text{PTS,back}}$ (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pair_001` | 60 fps Front + 60 fps Back | 96 | 96 | 96 | 0 | 3.0 | 5.5929 | 2.6736 |
| `pair_013` | Confirmed VFR pair | 96 | 96 | 48 | 48 | 3.0 | 1.5787 | 8.3333 |
| `pair_015` | 60 fps Front + 30 fps Back | 96 | 96 | 48 | 48 | 37.0 | 4.6593 | 8.3333 |
| `pair_019` | 60 fps Front + 60 fps Back | 96 | 96 | 96 | 0 | 8.0 | 1.9611 | 3.5764 |

---

## 3. Per-Pair Tracklet Extraction & Association Outcomes

### 3.1 Motorcycle-Level Outcomes

| Pair ID | Front Moto Tracklets | Back Moto Tracklets | Candidate Pairs | `LINKED` | `UNMATCHED` | `AMBIGUOUS_ABSTAIN` | `INSUFFICIENT_EVIDENCE` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pair_001` | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `pair_013` | 2 | 1 | 2 | 0 | 0 | 2 | 0 |
| `pair_015` | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| `pair_019` | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| **Total** | **5** | **3** | **4** | **2** | **1** | **2** | **0** |

### 3.2 Detailed Decision Analysis by Pair

1. **`pair_001` (Single-Camera Motorcycle + Orphan Rider Ambiguity Abstention):**
   - `front_moto_0001` was tracked in Front (`1` motorcycle tracklet), while in Back the distant rider head was detected without a coupled motorcycle box (`back_orphan_0001`).
   - `front_moto_0001` was classified as `UNMATCHED` (`NO_ELIGIBLE_OPPOSITE_CAMERA_TRACKLETS`).
   - At the orphan level, `front_moto_0001_rider_s1` vs. `back_orphan_0001` achieved score `0.8419`, but `front_orphan_0002` vs. `back_orphan_0001` achieved `0.8406` (margin `0.0013 < 0.14`), triggering conservative `AMBIGUOUS_ABSTAIN` (`ORPHAN_AMBIGUOUS_MARGIN_0.001_BELOW_0.14`). Three 1-frame Front orphans (`front_orphan_0003..0005`) were classified as `INSUFFICIENT_EVIDENCE`.
2. **`pair_013` (VFR Pair — Multi-Candidate Ambiguity Abstention):**
   - Two Front tracklets (`front_moto_0001`, `front_moto_0002`) competed for a single Back tracklet (`back_moto_0001`).
   - `(front_moto_0001, back_moto_0001)` scored `0.9219` (`temporal_transit = 0.9975`, `rider_cardinality = 1.0000`, `appearance_hsv = 0.5647`, $\Delta t_{\text{transit}} = +0.4333\text{ s}$).
   - `(front_moto_0002, back_moto_0001)` scored `0.8727` (`temporal_transit = 0.9964`, `rider_cardinality = 0.6500`, $\Delta t_{\text{transit}} = -0.5250\text{ s}$).
   - Because the uniqueness margin `0.9219 - 0.8727 = 0.0492 < 0.10` (`ambiguous_margin_threshold`), [`GlobalAssignmentSolver`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/global_assignment.py) abstained (`AMBIGUOUS_ABSTAIN`) on both rather than forcing a potentially false link.
3. **`pair_015` (Mixed 60/30 fps Pair — Confirmed Motorcycle Link + Insufficient Rider Evidence):**
   - `(front_moto_0001, back_moto_0001)` was `LINKED` as `global_moto_0001` with composite score `0.9045` (`temporal_transit = 0.9791`, `rider_cardinality = 1.0000`, `appearance_hsv = 0.5218`, $\Delta t_{\text{transit}} = +1.0667\text{ s}$, $\epsilon_{\text{sync}} = 37.0\text{ ms}$).
   - Within `global_moto_0001`, `front_moto_0001_rider_s0` (`driver`) and `back_moto_0001_rider_s0` (`driver`) had only `1` observed frame on the Back subtrack (`< min_observed_frames_rider = 2`), so [`RiderAssociationEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/rider_association.py) classified the rider link as `INSUFFICIENT_EVIDENCE` while preserving the parent motorcycle link.
4. **`pair_019` (Multi-Rider Partial Occlusion — Confirmed Motorcycle & Driver Link):**
   - `(front_moto_0001, back_moto_0001)` was `LINKED` as `global_moto_0001` with composite score `0.8298` (`temporal_transit = 1.0000`, `rider_cardinality = 0.6500` with partial pillion occlusion tolerance, `appearance_hsv = 0.6191`, $\Delta t_{\text{transit}} = -0.0083\text{ s}$).
   - Within `global_moto_0001`, `front_moto_0001_rider_s0` (`driver`) and `back_moto_0001_rider_s0` (`driver`) were `LINKED` with score `0.9176` (margin `0.6676`), while `back_moto_0001_rider_s1` (`pillion_1`, visible only in Back) was recorded as `UNMATCHED` (`SINGLE_CAMERA_ONLY_RIDER_UNMATCHED`).

---

## 4. Cue Availability & Weight Renormalization Verification

Across all evaluated candidate pairs in [`results/phase12e/cue_score_distributions.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12e/cue_score_distributions.csv):
- **Cue D (`lane_spatial_context`)** was `available = False` (`UNCALIBRATED_CAMERA_EXTRINSICS`) on 100% of pairs and received `effective_weight = 0.0`.
- **Cue E (`convoy_order`)** was `available = False` (`SINGLE_MOTORCYCLE_NO_CONVOY`) when fewer than 2 motorcycles existed in Back and received `effective_weight = 0.0`.
- When Cues A, B, and C were active (`pair_013` pair 1, `pair_015`, `pair_019`), nominal weights `(0.45, 0.25, 0.15)` were renormalized to `(0.5294, 0.2941, 0.1765)` (summing to `1.0000`).
- When only Cues A and B were active (`pair_013` pair 2), nominal weights `(0.45, 0.25)` were renormalized to `(0.6429, 0.3571)` (summing to `1.0000`).

---

## 5. Runtime & Memory Benchmark (`results/phase12e/association_runtime_benchmark.csv`)

| Pair ID | Tracklet Build (ms) | Candidate Gen (ms) | Cue Scoring (ms) | Global + Rider Solve (ms) | Orphan Solve (ms) | Total Assoc (ms) | Total Wall Time (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_001` | 118.8279 | 0.0043 | 0.0068 | 0.0353 | 1.0230 | 119.8973 | 8.479 | 1329.58 |
| `pair_013` | 67.8001 | 0.1676 | 0.2057 | 0.1238 | 0.0106 | 68.3077 | 11.503 | 1400.95 |
| `pair_015` | 66.2328 | 0.0555 | 0.0946 | 0.1133 | 0.0008 | 66.4970 | 7.057 | 1458.30 |
| `pair_019` | 146.7734 | 0.0848 | 0.1172 | 0.2026 | 0.0337 | 147.2118 | 10.113 | 1458.81 |
