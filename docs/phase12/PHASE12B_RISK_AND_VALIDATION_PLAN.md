# PHASE 12B — TECHNICAL RISK ANALYSIS, ABLATION DESIGN & VALIDATION PLAN

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Purpose:** Comprehensive failure-mode register, mitigation architecture, mandatory per-phase validation gates (Phases 12C–12J), and formal 5-way ablation study design.

---

## 1. Technical & Scientific Risk Register

| Risk ID | Risk Category & Description | Empirical Severity | Root Cause / Trigger | Architectural Mitigation Strategy |
| :--- | :--- | :---: | :--- | :--- |
| **R-01** | **Variable Frame Rate (VFR) Desynchronization in `pair_013_back` & `pair_014_back`** | **CRITICAL** (Measured `2,527 ms` and `826 ms` error under nominal FPS) | iOS camera dynamically switches between 60 fps ($\Delta t=16.7\text{ ms}$) and 30 fps ($\Delta t=33.3\text{ ms}$) mid-clip. | **Phase 12C PTS Lookup Table**: Index every frame's true `CAP_PROP_POS_MSEC` timestamp at stream initialization and retrieve via binary search (`np.searchsorted`). Reduces VFR error to $0.0\text{ ms}$ (bounded only by half-frame exposure step $\le 16.7\text{ ms}$). |
| **R-02** | **Frame-Rate Asymmetry (60 fps Front vs. 30 fps Back in `pair_015`–`pair_018`)** | **HIGH** | Back camera recorded at 30 fps while Front recorded at 60 fps; naive tick stepping feeds duplicate Back frames into Phase 10 EMA/state machine, artificially halving confirmation time. | **Phase 12C `is_new_back_frame` Gate**: `SynchronizedVideoReader` tracks `prev_back_frame_idx` and sets `is_new_back_frame = False` when a 30 fps frame is held across two 60 fps ticks. Per-camera detector and state machine update **only** when `is_new_frame == True`. |
| **R-03** | **Near-Frame Synchronization Uncertainty in 4 Pairs (`pair_004`, `005`, `008`, `015`)** | **MEDIUM** ($\epsilon_{\text{sync}} \in [18, 38]\text{ ms}$) | Residual uncertainty from Phase 11.6 acoustic/visual alignment (~1 to 2.3 frame intervals). | **Phase 12E/12G $\epsilon_{\text{sync}}$ Propagation**: Widen cross-camera transit variance to $\sigma_{\tau}^2 + \epsilon_{\text{sync}}^2$, apply sync discount $\phi_{\text{sync}}(\epsilon_{\text{sync}})$, and report evaluation metrics stratified by sync tier (21 Sub-Frame vs. 4 Near-Frame). |
| **R-04** | **$180^\circ$ Viewpoint Inversion Appearance Mismatch** | **HIGH** | Front view captures headlight/windshield/face; Back view captures taillight/plate/occiput. Standard CNN Re-ID features fail. | **Phase 12E View-Symmetric Multi-Cue Matcher**: Combine transit kinematics ($S_{\text{time}}$), rider occupancy count ($S_{\text{occ}}$), and $360^\circ$-symmetric helmet/shoulder HSV histograms ($S_{\text{color}}$) with convoy ordering ($S_{\text{order}}$). |
| **R-05** | **Wrong Cross-Camera Identity Link in Dense Convoys (False Plate Attachment)** | **CRITICAL** | Two similar motorcycles pass simultaneously; forced Hungarian matching could attach Motorcycle B's license plate to Motorcycle A's violation. | **Phase 12E Ambiguity Margin Gate ($\Delta S_{\text{margin}} \ge 0.10$)**: If best match score minus second-best match score is $< 0.10$, transition both tracks to `AMBIGUOUS_ASSOC_ABSTAIN` and prohibit cross-camera plate transfer. |
| **R-06** | **Multi-Rider Confusion (Driver Wearing Helmet + Pillion Without Helmet)** | **HIGH** | In South Asian traffic, drivers often wear helmets while pillion passengers do not. Head-only tracking cannot bind pillion to the same motorcycle/plate. | **Phase 12D `MotoRiderCoupler`**: Explicitly couples $K \in \{1,2,3\}$ rider heads to their parent `motorcycle` bounding box (`yolov8n.pt` class 3) and maintains ordered `RiderSubTrack` slots (`driver`, `pillion_1`, `pillion_2`). |
| **R-07** | **Rear License Plate Motion Blur, Low Resolution & Non-Latin Scripts** | **HIGH** | Handheld smartphone motion blur on fast receding motorcycles; Bangladeshi plates mix Bengali and Latin characters or suffer mud occlusion. | **Phase 12F Top-$K$ Sharpness Filter & Evidentiary Decoupling**: Score crops by Laplacian variance + pixel area, run EasyOCR (`['en', 'bn']`) on top $K=5$ crops with ROVER voting, and decouple violation confirmation from OCR legibility (`UNREADABLE_BLUR_OR_OCCLUSION`). |
| **R-08** | **Anterior-Posterior Semantic Contradiction (Caps / Turbans / Hoods)** | **MEDIUM** | A cap or hood may look like a helmet from behind ($P_B(\text{helmet}) = 0.85$) while the front camera clearly shows an bare face ($P_F(\text{no\_helmet}) = 0.90$). | **Phase 12G Contradiction Gate ($K_{\text{conflict}} \ge 0.35$)**: Detect strong opposing posteriors and route to `EVIDENCE_CONFLICT_ABSTAIN` for human review rather than averaging to $0.50$. |
| **R-09** | **Apple M1 Unified Memory Exhaustion (8 GB RAM Limit)** | **MEDIUM** | Simultaneous loading of 3 YOLO models + EasyOCR + full video frame buffers can exceed 8 GB RAM. | **Strict Streaming & Lazy Singleton Budget (< 1.0 GB Peak)**: Stream frames one tick at a time via sequential `grab()`, batch Front+Back YOLO inference on MPS, and invoke EasyOCR only on $\le 5$ tiny plate crops at track completion. |

---

## 2. Formal 5-Way Ablation Study Matrix (Phase 12I)

To prove the scientific contribution of each architectural component for thesis defense and publication, Phase 12I will evaluate five controlled system configurations across all 25 benchmark pairs:

| Ablation ID | Configuration Name | Active Cameras | Temporal Verification (Phase 10) | Cross-Camera Matching & Fusion | Rear Plate & OCR | Primary Research Question Answered |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ABL-1** | `FRONT_ONLY_TEMPORAL` | Front ($C_F$) only | Enabled (EMA + Hysteresis) | Disabled | Unavailable ($0\%$ rear plate visibility in oncoming view) | Baseline oncoming single-camera performance: high anterior facial clarity for helmet detection, but zero rear license plate capture. |
| **ABL-2** | `BACK_ONLY_TEMPORAL` | Back ($C_B$) only | Enabled (EMA + Hysteresis) | Disabled | Enabled on Back tracks | Baseline receding single-camera performance: captures rear plates, but suffers anterior head occlusion (driver head blocked by pillion in receding view). |
| **ABL-3** | `DUAL_INSTANTANEOUS_NO_TEMPORAL` | Front + Back ($C_F + C_B$) | **Disabled** (Raw single-frame max/avg without EMA or state machine) | Enabled | Enabled | Quantifies the necessity of **Confidence-Aware Temporal Verification** in suppressing frame-to-frame label flips and transient false positives. |
| **ABL-4** | `DUAL_FUSED_NO_OCR` | Front + Back ($C_F + C_B$) | Enabled | Enabled (5-Cue Matcher + Log-Odds Fusion + Conflict Gate) | Disabled | Isolates the pure **Dual-Camera Helmet Violation Detection & Cross-Camera Association** gain (Precision, Recall, $F_1$, Flip-Rate reduction, pillion occlusion recovery) over `ABL-1` and `ABL-2`. |
| **ABL-5** | `DUAL_FULL_FRAMEWORK` *(Proposed)* | Front + Back ($C_F + C_B$) | Enabled | Enabled | Enabled (Pretrained Plate + Top-5 EasyOCR ROVER) | Evaluates the complete end-to-end framework: corroborated dual-view helmet violation detection linked to rear license plate evidence with calibrated abstention. |

### Secondary Component Ablations (Within Phase 12E & 12G)
In addition to the 5 primary system ablations, Phase 12I will report a compact **Association Cue Leave-One-Out Table**:
1. *Time-Only Matching* ($w_t = 1.0$, all other cues $0$)
2. *Time + Occupancy Cardinality* ($S_{\text{time}} + S_{\text{occ}}$)
3. *Time + Occupancy + Symmetric HSV Color* ($S_{\text{time}} + S_{\text{occ}} + S_{\text{color}}$)
4. *Full 5-Cue Matcher without Margin Abstention* ($\delta_{\text{margin}} = 0$) vs. *with Margin Abstention* ($\delta_{\text{margin}} = 0.10$)

---

## 3. Sequential Validation Gates & Test Strategy (Phases 12C–12J)

No subphase may advance to the next until its automated `pytest` suite and quantitative validation criteria pass 100%:

### Gate 12C: Synchronized Video Reader & PTS Alignment Verification
- **Unit & Forensic Tests (`tests/test_phase12c_pts_sync.py`)**:
  1. Verify that `VideoReader` builds monotonically increasing PTS tables across all 50 video files (`25 front + 25 back`).
  2. **VFR Regression Test on `pair_013_back.mp4` and `pair_014_back.mp4`**: Prove that `get_frame_at_pts(t)` retrieves frames within $\delta_{\text{pts}} \le 18.0\text{ ms}$ of true `CAP_PROP_POS_MSEC` across the entire clip (eliminating the $2,527\text{ ms}$ nominal-FPS bug).
  3. **30/60 FPS Deduplication Test on `pair_015`–`pair_018`**: Verify `is_new_back_frame` toggles `True` ~30 times/sec while `is_new_front_frame` toggles `True` ~60 times/sec.
  4. **Sign Convention Sanity Check**: Confirm that at $t_{\text{ref}}$, $t_{\text{front}} = t_{\text{ref}} - \Delta t_{\text{rec}}$ matches the Phase 11.5 visual contact sheet frames within $\le 1$ frame across all 25 pairs.

### Gate 12D: Per-Camera Detection, Rider-Motorcycle Coupling & Tracking
- **Unit & Integration Tests (`tests/test_phase12d_tracking.py`)**:
  1. Synthetic multi-rider geometry test: Verify `MotoRiderCoupler` accurately binds a driver head and pillion head to the same motorcycle box while keeping an adjacent lane motorcycle's rider separate.
  2. Verify `OrphanRiderTrack` fallback when a motorcycle bbox is artificially masked out.
  3. Verify `RiderSubTrack` preserves exact Phase 10 `TemporalStateMachine` hysteresis behavior (`min_confirm_frames=5`, cooldown suppression).
  4. Smoke test on 3 real pairs (`pair_001`, `pair_013`, `pair_019`) verifying RSS memory stays $< 1.2\text{ GB}$.

### Gate 12E: Cross-Camera Identity Association
- **Unit & Integration Tests (`tests/test_phase12e_association.py`)**:
  1. Verify $S_{\text{time}}, S_{\text{occ}}, S_{\text{color}}, S_{\text{lane}}, S_{\text{order}} \in [0, 1]$ and deterministic Hungarian matching.
  2. **Ambiguity Abstention Test**: Feed two synthetic motorcycles with identical timestamps and colors ($\Delta S_{\text{margin}} = 0.02 < 0.10$) and assert that `assoc_status == "AMBIGUOUS_ABSTAIN"`.
  3. **Sync Uncertainty Widening Test**: Verify that increasing $\epsilon_{\text{sync}}$ from $1.0\text{ ms}$ to $38.0\text{ ms}$ widens the transit tolerance kernel without throwing false rejections.

### Gate 12F: Pretrained License Plate Localization & Top-$K$ OCR
- **Unit & Integration Tests (`tests/test_phase12f_ocr_pipeline.py`)**:
  1. Verify `PlateQualitySelector` rejects crops smaller than $28 \times 12\text{ px}$ and ranks sharp synthetic crops above Gaussian-blurred crops via Laplacian variance.
  2. Verify ROVER character consensus merges noisy multi-frame reads (`["DHAKA-1234", "DHAKA-123A", "DHAKA-1234"] -> "DHAKA-1234"`).
  3. Verify that an unreadable plate (`C_ocr < 0.40`) outputs `plate_status = "UNREADABLE_BLUR_OR_OCCLUSION"` without suppressing the parent helmet violation.

### Gate 12G: Confidence-Aware Evidence Fusion & Conflict Gate
- **Unit & Integration Tests (`tests/test_phase12g_fusion.py`)**:
  1. **Corroboration Boost Test**: Verify that concordant Front ($P_F = 0.75$) and Back ($P_B = 0.72$) violation posteriors fuse to $P_{\text{fused}} > 0.75$.
  2. **Contradiction Gate Test**: Verify that strongly conflicting views ($P_F = 0.88$ `no_helmet`, $P_B = 0.12$ `helmet`, $K_{\text{conflict}} = 0.76 \ge 0.35$) trigger `EVIDENCE_CONFLICT_ABSTAIN`.
  3. **Unmatched Fallback Test**: Verify that `UNMATCHED_FRONT_ONLY` tracks with $P_F \ge 0.65$ output `SINGLE_VIEW_VIOLATION_FRONT` with `plate_status = "UNAVAILABLE_FRONT_ONLY"`.

### Gate 12H–12J: Ground-Truth Annotation Integrity & Full Benchmark Evaluation
- **Validation Checks**:
  1. Schema validator confirming `dual_camera_gt_v1.json` covers all 25 pairs (`pair_001`–`pair_025`) with zero missing required fields and zero model-generated placeholder flags.
  2. Reproducibility check confirming all 5 ablation runs (`ABL-1` to `ABL-5`) execute deterministically from a single CLI command and export complete CSV/JSON/Markdown reports.
