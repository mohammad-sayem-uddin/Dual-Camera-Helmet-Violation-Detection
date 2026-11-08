# PHASE 12E — ALGORITHM AND PARAMETER DECISIONS

**Project:** *An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification*  
**Phase:** Phase 12E — Cross-Camera Tracklet Association  
**Configuration File:** [`configs/cross_camera_association_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/cross_camera_association_config.yaml)  
**Schema Version:** `1.0.0`

---

## 1. Scientific Provenance & Engineering Hypothesis Policy

Every numerical threshold, transit-time window, quality gate, and multi-cue weight in Phase 12E is explicitly labeled with `engineering_hypothesis: true` in [`configs/cross_camera_association_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/cross_camera_association_config.yaml). None of these parameters is presented as an empirically calibrated optimum until evaluated against a human-annotated cross-camera identity ground-truth dataset.

---

## 2. Core Algorithmic Decisions

### 2.1 Observed-Only Tracklet Extraction vs. Full-Kalman Trajectory Summarization
- **Decision:** [`TrackletBuilder`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_builder.py) separates matched physical observations ([`ObservedFrameRecord`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py)) from purely Kalman-predicted (`LOST`) frames (`predicted_only_frame_count`, `occlusion_intervals`).
- **Rationale:** During occlusion, constant-velocity Kalman predictions extrapolate bounding boxes without visual evidence. Including predicted frames in appearance histograms or entry/exit bounding box geometry would contaminate cross-camera descriptors with background pixels and extrapolated positions.

### 2.2 Canonical Reference Clock Alignment & Uncertainty Separation
- **Decision:** [`TrackletBuilder`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_builder.py) maps Front and Back camera stream-normalized PTS onto the Phase 12C canonical reference timeline ($t_{\text{ref}} = t_{\text{back}}$ for Back; $t_{\text{ref}} = t_{\text{front}} + \Delta t_{\text{rec}}$ where $\Delta t_{\text{rec}} = \text{global\_offset\_sec}$ for Front), while [`TransitTimeModel`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/transit_time_model.py) and [`CandidatePairScore`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py) preserve five separate uncertainty/delay terms:
  1. `epsilon_sync_ms`: Phase 11.6 residual synchronization error (`3.0 ms` to `38.0 ms`).
  2. `delta_pts_front_ms`: Front container PTS discretization residual.
  3. `delta_pts_back_ms`: Back container PTS discretization residual.
  4. `delta_transit_sec`: Signed physical cross-camera transit delay ($t_{\text{back,centroid}} - t_{\text{front,centroid}}$).
  5. `transit_uncertainty_sec`: Physical vehicle speed and corridor transit uncertainty (`0.35 s`).

### 2.3 Multi-Cue Architecture & Missing-Cue Weight Renormalization
- **Decision:** [`AssociationScorer`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/association_scorer.py) evaluates up to five cues:
  - **Cue A (`temporal_transit`, nominal weight `0.45`):** Gaussian transit-time compatibility within the feasible corridor bounds, gated by `min_temporal_transit_score_gate = 0.20`.
  - **Cue B (`rider_cardinality`, nominal weight `0.25`):** Compares modal and subtrack rider counts across views, granting partial occlusion tolerance (`0.65`) when rider counts differ by `1` (`2 vs 1`) and one camera has partial rider visibility (`< 0.75` observation ratio or occlusion intervals), while strongly penalizing `diff >= 2` (`0.08`).
  - **Cue C (`appearance_hsv`, nominal weight `0.15`):** Optional viewpoint-symmetric joint chromatic (`12 x 4` H-S) + achromatic (`8` V) histogram Bhattacharyya similarity across three zones (`cranial_cap = 0.30`, `upper_torso = 0.45`, `moto_chassis = 0.25`). Strictly gated by crop resolution (`>= 20x20 px`), Laplacian sharpness (`>= 12.0`), and composite quality (`>= 0.35`). **Never uses helmet vs. no-helmet classification labels as an identity cue.**
  - **Cue D (`lane_spatial_context`, nominal weight `0.05`):** Explicitly disabled by default (`enabled: false`, `unavailable_reason = "UNCALIBRATED_CAMERA_EXTRINSICS"`) because Front and Back recordings were captured by handheld smartphones without calibrated extrinsics or homography.
  - **Cue E (`convoy_order`, nominal weight `0.10`):** Evaluates ordinal arrival rank preservation when $\ge 2$ motorcycles co-occur within `convoy_window_sec = 5.0 s` in both cameras with minimum intra-camera anchor separation `>= 0.25 s`. Marked `available: false` (`SINGLE_MOTORCYCLE_NO_CONVOY`) when only a single motorcycle is present.
- **Missing-Cue Renormalization:** Unavailable cues (`available = False`) receive `effective_weight = 0.0`, and all available cues have their weights renormalized to sum to `1.0`:
  $$w_k^{\text{eff}} = \frac{w_k^{\text{nom}}}{\sum_{m \in \mathcal{A}} w_m^{\text{nom}}}, \quad S(F_i, B_j) = \sum_{k \in \mathcal{A}} w_k^{\text{eff}} s_k(F_i, B_j)$$
  This guarantees that an unavailable optional cue never acts as a silent zero penalty.

### 2.4 Global 1-to-1 Bipartite Assignment with 4-State Conservative Outcomes
- **Decision:** [`GlobalAssignmentSolver`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/global_assignment.py) solves global 1-to-1 matching via `scipy.optimize.linear_sum_assignment` and enforces four explicit outcomes (`AssociationOutcome`):
  1. `INSUFFICIENT_EVIDENCE`: Tracklet has `< 2` observed frames, `FRAGMENTED_TENTATIVE` completeness, or `< 2` active cues.
  2. `AMBIGUOUS_ABSTAIN`: Candidate pair passes `min_association_score = 0.62`, but either its row/column second-best margin $S_{i,j} - \max(S_{i,j'}, S_{i',j}) < 0.10$ or its $2 \times 2$ assignment swap margin $(S_{i,j} + S_{i',j'}) - (S_{i,j'} + S_{i',j}) < 0.10$.
  3. `LINKED`: Unambiguous 1-to-1 match passing all gates and margins, assigned a deterministic global ID (`global_moto_0001`, ...).
  4. `UNMATCHED`: Single-camera tracklet with no viable opposite-camera counterpart.

### 2.5 Separate Rider-Level & First-Class Orphan Association
- **Rider-Level Association ([`RiderAssociationEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/rider_association.py)):** Operates within a `LINKED` motorcycle pair using slot-role compatibility (`0.55`), relative temporal coverage (`0.25`), and optional rider appearance (`0.20`). Preserves per-camera `subtrack_id`s, `slot_role`s, and Phase 10 verification states independently as read-only snapshots without merging or mutating them. Abstains (`AMBIGUOUS_ABSTAIN`) at the rider level when multi-rider ordering is not confident (`ordering_confident = False`) or when the rider swap margin is `< 0.12`, while keeping the parent motorcycle `LINKED`.
- **Orphan-Level Association ([`OrphanAssociationEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/orphan_association.py)):** Treats [`OrphanRiderTracklet`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py) as a first-class input, requiring stricter score (`>= 0.68`), stricter uniqueness margin (`>= 0.14`), and either valid appearance similarity or strong temporal transit compatibility (`>= 0.82`). Every orphan match is explicitly labeled `association_level = "ORPHAN_LEVEL_LINK"` with `implies_confirmed_motorcycle_link = False`.

---

## 3. Complete Parameter Inventory (`configs/cross_camera_association_config.yaml`)

| Section | Parameter | Default Value | Engineering Hypothesis | Purpose |
|---|---|---|---|---|
| `tracklet_builder` | `min_observed_frames_moto` | `2` | `true` | Minimum observed frames for motorcycle tracklet eligibility |
| `tracklet_builder` | `min_observed_frames_rider` | `2` | `true` | Minimum observed frames for rider subtrack eligibility |
| `tracklet_builder` | `min_observed_frames_orphan` | `2` | `true` | Minimum observed frames for orphan tracklet eligibility |
| `tracklet_builder` | `min_observation_density` | `0.30` | `true` | Threshold below which a tracklet is flagged `HEAVILY_OCCLUDED` |
| `tracklet_builder` | `border_margin_px` | `45.0` | `true` | Frame border margin for near-border entry/exit classification |
| `transit_time_model` | `corridor_mode` | `"bidirectional_corridor"` | `true` | Supports bidirectional roadside traffic corridor |
| `transit_time_model` | `min_transit_delay_sec` | `-3.50` | `true` | Lower signed canonical transit time bound ($t_{\text{back}} - t_{\text{front}}$) |
| `transit_time_model` | `max_transit_delay_sec` | `6.00` | `true` | Upper signed canonical transit time bound ($t_{\text{back}} - t_{\text{front}}$) |
| `transit_time_model` | `expected_transit_delay_sec` | `0.75` | `true` | Nominal expected transit delay magnitude |
| `transit_time_model` | `transit_sigma_sec` | `1.50` | `true` | Gaussian transit compatibility scale parameter |
| `transit_time_model` | `transit_uncertainty_sec` | `0.35` | `true` | Physical transit time uncertainty (separate from clock sync error) |
| `transit_time_model` | `max_temporal_search_window_sec` | `8.00` | `true` | Hard candidate generation temporal search cutoff |
| `rider_cardinality` | `exact_count_match_score` | `1.00` | `true` | Score when modal rider counts match |
| `rider_cardinality` | `partial_pillion_occlusion_diff1_score` | `0.65` | `true` | Score when rider count differs by 1 with occlusion evidence |
| `rider_cardinality` | `unexplained_diff1_score` | `0.35` | `true` | Score when rider count differs by 1 without occlusion evidence |
| `rider_cardinality` | `diff2_or_more_score` | `0.08` | `true` | Penalty when rider count differs by $\ge 2$ (`1 vs 3`) |
| `appearance_features` | `min_crop_width_px` / `height_px` | `20` | `true` | Minimum bounding box crop resolution for Cue C |
| `appearance_features` | `min_laplacian_sharpness` | `12.0` | `true` | Minimum grayscale Laplacian variance to reject motion blur |
| `appearance_features` | `min_valid_crop_quality` | `0.35` | `true` | Minimum composite crop quality in $[0, 1]$ |
| `lane_spatial_context` | `enabled` | `false` | `true` | Disabled because cameras lack calibrated extrinsics/homography |
| `convoy_order` | `convoy_window_sec` | `5.00` | `true` | Co-occurrence window for multi-motorcycle convoy ordering |
| `association_scoring` | `cue_weights` | `0.45 / 0.25 / 0.15 / 0.05 / 0.10` | `true` | Nominal weights for Cues A, B, C, D, E before renormalization |
| `global_assignment` | `min_association_score` | `0.62` | `true` | Minimum composite score required for `LINKED` |
| `global_assignment` | `ambiguous_margin_threshold` | `0.10` | `true` | Minimum uniqueness and $2 \times 2$ swap margin before `AMBIGUOUS_ABSTAIN` |
| `rider_association` | `min_rider_association_score` | `0.60` | `true` | Minimum rider-level score within a linked motorcycle |
| `rider_association` | `rider_ambiguous_margin_threshold` | `0.12` | `true` | Minimum rider-level uniqueness/swap margin |
| `orphan_association` | `min_orphan_association_score` | `0.68` | `true` | Stricter threshold for `ORPHAN_LEVEL_LINK` |
| `orphan_association` | `orphan_ambiguous_margin_threshold` | `0.14` | `true` | Stricter uniqueness margin for orphan rider links |
