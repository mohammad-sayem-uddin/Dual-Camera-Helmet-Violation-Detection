# PHASE 12B — ALGORITHMIC COMPARISON & SELECTION MATRIX

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Purpose:** Rigorous trade-off analysis comparing candidate algorithms across all five core subsystems: (1) Synchronized Timestamp Alignment, (2) Per-Camera Object Detection & Tracking, (3) Opposing-View Cross-Camera Identity Association, (4) Pretrained License Plate Localization & OCR, and (5) Confidence-Aware Dual-Camera Evidence Fusion.

> [!IMPORTANT]
> Throughout this document, we explicitly distinguish between **Established Literature / Empirical Facts** (verified by prior published work or direct forensic measurement on our 25-pair dataset) and **Engineering Hypotheses** (design expectations to be empirically validated in Phase 12I ablations).

---

## 1. Subsystem A: Video Timestamping & Cross-Camera Frame Alignment

### 1.1 Empirical Baseline Facts (Measured on `data/dual_camera_test_v1/`)
- **Fact A1**: 25/25 Front videos (`pair_001_front`–`pair_025_front`) are Constant Frame Rate (CFR) at `59.72–60.02` fps.
- **Fact A2**: 19/25 Back videos are CFR at `59.92–59.94` fps, 4/25 (`pair_015_back`–`pair_018_back`) are CFR at `29.96–30.04` fps, and 2/25 (`pair_013_back` at `40.17` avg fps, `pair_014_back` at `48.51` avg fps) are **Variable Frame Rate (VFR)** where the iOS encoder dynamically alternates between $\Delta t = 16.67\text{ ms}$ (60 fps) and $\Delta t = 33.33\text{ ms}$ (30 fps).
- **Fact A3**: On `pair_013_back.mp4`, computing frame timestamps via nominal `frame_idx / avg_fps` produces a measured maximum error of **`2,527.07 ms` (2.53 seconds)** against container Presentation Timestamps (`CAP_PROP_POS_MSEC`). On `pair_014_back.mp4`, the measured error is **`826.01 ms`**.
- **Fact A4**: On Android Front recordings (`pair_004_front`, `pair_008_front`, `pair_015_front`), raw `ffprobe` packet `pkt_pts_time` begins at `+31 ms` to `+72 ms` due to an audio-stream lead, whereas OpenCV `CAP_PROP_POS_MSEC` anchors Frame 0 to `0.000 s` (matching the timeline used when `synchronization_metadata.json` was generated in Phase 11).

### 1.2 Candidate Comparison Matrix

| Candidate Method | VFR Robustness (`pair_013`, `014`) | 30/60 FPS Asymmetry (`pair_015`–`018`) | Decode Speed on Apple M1 | Temporal Precision | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1. Nominal Frame-Index Mapping** (`idx = round(t * avg_fps)` + `CAP_PROP_POS_FRAMES`) | **Fails catastrophically** (up to `2,527 ms` drift on `pair_013_back`) | Duplicates 30 fps frames without tracking unique frame transitions | Slow ($\sim 12\text{ ms/seek}$ due to H.264 GOP keyframe re-decoding) | Poor on VFR ($\pm 2.53\text{ s}$) | **Rejected** (Phase 12A placeholder only) |
| **A2. Pre- Re-encoding All 25 Pairs to 60 fps CFR via FFmpeg** | Eliminates VFR by synthesizing duplicate frames | Upsamples 30 fps to 60 fps via frame duplication | Fast sequential read | High ($\le 8.3\text{ ms}$) | **Rejected** (Violates strict rule against modifying/overwriting dataset files & wastes disk) |
| **A3. Stream-Normalized PTS Lookup Table + Sequential Forward Buffer (`CAP_PROP_POS_MSEC` + `np.searchsorted`)** | **Exact** (`0.0 ms` VFR drift; tracks every 60 $\leftrightarrow$ 30 fps transition natively) | Emits `is_new_back_frame` flag so 30 fps streams run detector only once per unique physical frame | **Fastest** ($\sim 0.8\text{ ms/frame}$ via sequential `grab()`/`retrieve()`) | **Optimal** ($\delta_{\text{pts}} \le \frac{1}{2}\Delta t_{\text{frame}}$) | **SELECTED FOR PHASE 12C** |

---

## 2. Subsystem B: Per-Camera Object Detection & Multi-Object Tracking

### 2.1 Literature & Empirical Grounding
- **Established Fact**: The Phase 9 `best.pt` model (`exp_combined` seed 42) achieves strong cranial detection (`mAP@50 = 0.9025` on combined test set) for `0: no_helmet` and `1: helmet`, but has no `motorcycle` or `person` class.
- **Established Fact**: ByteTrack (Zhang et al., ECCV 2022) outperforms single-threshold SORT/IoU trackers under partial occlusion by retaining low-confidence detections ($0.20 \le p < 0.45$) in a second association stage matched against Kalman filter velocity predictions.

### 2.2 Candidate Comparison Matrix

| Candidate Architecture | Multi-Rider Grouping (Driver + Pillion) | Rear Plate Search ROI Anchoring | Chassis Color / Velocity Extraction | Apple M1 Latency (per pair tick) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1. Phase 10 Greedy Head-Only IoU Tracker (`RiderTracker` + `best.pt` only)** | **Impossible** (cannot tell if 2 heads ride 1 motorcycle or 2 motorcycles) | Weak (must guess plate location far below head box) | None (head box contains no motorcycle chassis) | `~26 ms` | **Retained as Baseline Reference Only** |
| **B2. DeepSORT (`best.pt` + CNN Re-ID Appearance Extractor per frame)** | Impossible without motorcycle box | Weak | High GPU overhead running CNN on every crop at 60 fps | `~95 ms` | **Rejected** (Unnecessary compute within a single continuous 60 fps view) |
| **B3. Hierarchical Dual-YOLO (`yolov8n.pt` COCO + `best.pt` Helmet) + Spatial `MotoRiderCoupler` + 2-Stage Kalman ByteTrack** | **Explicitly groups** $K \in \{1,2,3\}$ rider heads onto parent `MotoTrack` via $\text{IoA}_{\text{head}}$ cone | **Exact** (rear plate sits directly inside lower `MotoTrack` box) | Extracts chassis + torso + cranial cues + Kalman velocity | `~52 ms` (batched MPS inference) | **SELECTED FOR PHASE 12D** |

---

## 3. Subsystem C: Cross-Camera Identity Association Across Opposing Views ($180^\circ$ Inversion)

### 3.1 Literature Grounding vs. Engineering Hypothesis
- **Established Literature Fact**: Pretrained pedestrian/vehicle Re-ID embeddings (e.g., OSNet, ResNet50-IBN) rely heavily on viewpoint-consistent surface textures. Under $180^\circ$ viewpoint inversion (front headlight/windshield/chest vs. rear taillight/mudguard/back), cosine similarity of generic CNN embeddings degrades sharply, frequently scoring a different motorcycle of similar scale higher than the true opposite view.
- **Engineering Hypothesis (to be tested in Phase 12I)**: Because our two cameras are synchronized to within $\epsilon_{\text{sync}} \le 10\text{ ms}$ (21 pairs) or $\le 38\text{ ms}$ (4 pairs) and observe a constrained road corridor, **spatio-temporal pass-through kinematics ($S_{\text{time}}$) + topological rider count ($S_{\text{occ}}$) + view-symmetric HSV color histograms ($S_{\text{color}}$) + convoy ordering ($S_{\text{order}}$)** will achieve higher association precision and better calibrated abstention than appearance-only Re-ID.

### 3.2 Candidate Comparison Matrix

| Candidate Method | Robustness to $180^\circ$ Viewpoint Inversion | Handling of Multi-Motorcycle Convoys | Adaptation to Sync Residual ($\epsilon_{\text{sync}}$) | Abstention on Ambiguous Twins | M1 Compute Cost | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1. Temporal Nearest-Neighbor Only** ($\arg\min |t_{\text{exit}}^F - t_{\text{enter}}^B|$) | View-invariant | **Fails when $\ge 2$ motorcycles pass side-by-side** | None (fixed threshold) | None | `< 0.1 ms` | **Rejected** (High swap risk in dense traffic) |
| **C2. Pretrained Deep Re-ID Embedding Cosine Matching** (ResNet50 / MobileNetV3) | **Poor** (anterior headlight/face $\neq$ posterior plate/back texture) | Moderate | None | Weak (uncalibrated cosine scores) | `~35 ms` | **Rejected as Primary** (Viewpoint domain gap) |
| **C3. Calibrated 3D Homography / Epipolar Projection** | High if cameras are static & overlapping | High | High | Moderate | `< 1 ms` | **Impossible** (Cameras are handheld smartphones without static intrinsics/extrinsics) |
| **C4. Multi-Cue Spatio-Temporal + View-Symmetric Anatomical HSV + Hungarian Bipartite Matcher with Margin Abstention** | **High** (uses transit kinematics, rider cardinality $\hat{K}$, and $360^\circ$-symmetric helmet/shoulder HSV) | **High** (Hungarian assignment + convoy rank order + lane geometry) | **Explicit** (widens transit kernel by $\sigma_{\tau}^2 + \epsilon_{\text{sync}}^2$) | **Explicit** (`AMBIGUOUS_ABSTAIN` if margin $\Delta S < 0.10$) | `< 1.5 ms` per track pair | **SELECTED FOR PHASE 12E** |

---

## 4. Subsystem D: License Plate Localization & Optical Character Recognition (OCR)

### 4.1 Strict Constraint & Trade-off Analysis
- **Mandatory Constraint**: No custom OCR or license plate detector training is allowed; only existing pretrained models may be used.
- **Problem with Per-Frame OCR**: Running EasyOCR on every Back camera frame (60 fps) costs $\sim 80\text{ ms/frame}$ on M1, bottlenecking throughput and flooding the aggregator with motion-blurred distant crops.

### 4.2 Candidate Comparison Matrix

| Candidate Pipeline | Localization Accuracy on Rear Motorcycles | OCR Legibility & Multi-Frame Stability | Compute Cost per Video Pair | Compliance with Pretrained Rule | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1. Phase 12A Heuristic Crop Only + Single-Frame EasyOCR** | Low (fixed percentage crop misses pitched or off-center plates) | Poor (single frame often suffers motion blur) | Fast (`~80 ms` once) | Compliant | **Insufficient as Primary** (Kept as fallback only) |
| **D2. Full-Frame EasyOCR on Every Back Frame** | Moderate (CRAFT text detector finds scene text/shop signs too) | High noise from roadside billboards & shop signs | **Unacceptable** (`> 45 sec` per 600-frame clip) | Compliant | **Rejected** |
| **D3. Pretrained YOLOv8n License Plate Detector (inside `MotoTrack_B` bbox) + Heuristic Fallback + Sharpness-Guided Top-$K$ ($K=5$) Crop Selection + CLAHE + EasyOCR ROVER Consensus** | **High** (pretrained plate detector searches strictly inside rear motorcycle bbox; heuristic fallback if missed) | **High** (selects 5 sharpest/largest crops via Laplacian variance; aligns character strings via ROVER voting) | **Fast** (`~3 ms` per crop for plate bbox + `~0.4 s` total EasyOCR per motorcycle track) | **100% Compliant** (uses open pretrained weights only) | **SELECTED FOR PHASE 12F** |

---

## 5. Subsystem E: Confidence-Aware Dual-Camera Evidence Fusion

### 5.1 Candidate Comparison Matrix

| Candidate Fusion Rule | Viewpoint Reliability Asymmetry ($C_F$ vs. $C_B$) | Handling of Distant / Blurred Boxes | Handling of Front-vs-Back Contradiction | Integration with Sync Uncertainty ($\epsilon_{\text{sync}}$) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1. Logical OR Rule** (`Violation = Front_Viol OR Back_Viol`) | Ignores | Amplifies false positives from either camera | Forces `Violation` even when better view strongly says `Helmet` | None | **Rejected** (Inflates False Positive Rate) |
| **E2. Logical AND Rule** (`Violation = Front_Viol AND Back_Viol`) | Ignores | Collapses recall when rider is occluded in one camera | Defaults to `Compliant` on conflict | None | **Rejected** (Destroys recall when one view is occluded) |
| **E3. Unweighted Probability Average** ($P = 0.5 P_F + 0.5 P_B$) | Treats occipital view equally with facial view | Treats tiny $15 \times 15\text{ px}$ box equally with $80 \times 80\text{ px}$ box | Averages $0.90$ and $0.10$ to $0.50$ without flagging conflict | None | **Rejected** (Naively masks strong conflicts) |
| **E4. Quality-Weighted Log-Odds Fusion with Viewpoint Prior ($\gamma_F, \gamma_B$), Sync Discount ($\phi_{\text{sync}}$), Corroboration Boost ($\beta_{\text{corrob}}$), and Contradiction Gate ($K_{\text{conflict}}$)** | Weights anterior facial clarity ($\gamma_F=1.10$) above posterior ($\gamma_B=0.95$) | Weights observations by scale $A_{\text{head}}$, truncation, and sharpness | Explicitly triggers `EVIDENCE_CONFLICT_ABSTAIN` when $K_{\text{conflict}} \ge 0.35$ | Discounts cross-camera link weight via $\phi_{\text{sync}}(\epsilon_{\text{sync}}, \delta_{\text{pts}})$ | **SELECTED FOR PHASE 12G** |

---

## 6. Summary of Selected Architecture Stack

| Stage | Selected Algorithm | Key Hyperparameters (Default Config) |
| :--- | :--- | :--- |
| **12C. Sync Reader** | Stream-Normalized PTS Lookup (`CAP_PROP_POS_MSEC`) + Binary Search + Sequential Forward Grab | $\Delta t_{\text{tick}} = 16.67\text{ ms}$, VFR-exact, `is_new_frame` deduplication |
| **12D. Per-Camera Tracking** | Dual YOLOv8n (`yolov8n.pt` + `best.pt`) + `MotoRiderCoupler` ($\text{IoA}_{\text{head}}$) + 2-Stage Kalman ByteTrack + Phase 10 EMA/Hysteresis | `high_thresh=0.45`, `low_thresh=0.20`, `ema_alpha=0.40`, `min_confirm=5` |
| **12E. Cross-Camera Matching** | 5-Cue Affinity ($S_{\text{time}}, S_{\text{occ}}, S_{\text{color}}, S_{\text{lane}}, S_{\text{order}}$) + Hungarian Assignment + Margin Abstention | $(w_t, w_o, w_c, w_l, w_r) = (0.35, 0.20, 0.25, 0.10, 0.10)$, $\tau_{\text{match}} = 0.55$, $\delta_{\text{margin}} = 0.10$ |
| **12F. Rear Plate & OCR** | Pretrained YOLOv8n Plate Detector + Heuristic Fallback + Top-$K$ Sharpness Filter + CLAHE + EasyOCR ROVER Voting | $K=5$ crops/track, $\min\text{size} = 28 \times 12\text{ px}$, $\tau_{\text{ocr}} = 0.40$ |
| **12G. Evidence Fusion** | Quality-Weighted Log-Odds Fusion + Sync Discount $\phi_{\text{sync}}$ + Conflict Abstention Gate $K_{\text{conflict}}$ | $\gamma_F = 1.10, \gamma_B = 0.95, \beta_{\text{corrob}} = 0.20, \tau_{\text{conflict}} = 0.35$ |
