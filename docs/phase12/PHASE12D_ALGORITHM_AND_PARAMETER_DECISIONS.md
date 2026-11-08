# PHASE 12D — ALGORITHM & PARAMETER DECISIONS

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection` (`main` branch)  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Configuration Source:** [`configs/tracking_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/tracking_config.yaml)

> [!IMPORTANT]
> **Scientific Status of Parameters:**  
> All geometric thresholds, affinity weights, and association gates documented below are **initial engineering parameters (hypotheses)** rather than established scientific constants. They are exposed in [`configs/tracking_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/tracking_config.yaml) for controlled sensitivity and ablation analysis in Phase 12I.

---

## 1. Dual-Model Per-Camera Detection Decisions

### 1.1 Model Selection & Verified Provenance
| Model Role | Checkpoint Path | SHA-256 Hash | Verified Class Mapping |
| :--- | :--- | :--- | :--- |
| **Vehicle & Person Detector** | [`yolov8n.pt`](file:///Users/sayemuddin/Desktop/research_helmet/yolov8n.pt) (6,549,796 bytes) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | Filtered strictly to `0: person`, `3: motorcycle` (from 80 COCO classes) |
| **Cranial Helmet Detector** | [`results/phase9_multiseed/exp_combined/seed_42/weights/best.pt`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase9_multiseed/exp_combined/seed_42/weights/best.pt) (6,506,922 bytes) | `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5` | Strictly 2 classes: `0: no_helmet`, `1: helmet` |

### 1.2 Architectural Rationale
- **Why Two Specialized YOLOv8n Models**: The Phase 9 `best.pt` model is a domain-adapted cranial detector (`no_helmet` / `helmet`) and does not output full-body `person` or `motorcycle` boxes. Conversely, official `yolov8n.pt` detects `motorcycle` and `person` boxes reliably on roadside traffic without retraining.
- **Strict Semantic Separation**: A cranial `helmet`/`no_helmet` box is never treated as a `person` box. `PerCameraDetector` tags every `StructuredDetection` with `category in {"motorcycle", "person", "head"}` and `source_model in {"yolov8n_coco", "phase9_best_helmet"}`.
- **Singleton Sharing on Apple M1 (8 GB Unified RAM)**: `SharedDetectorRegistry` loads a single instance of `yolov8n.pt` and `best.pt` on Apple Silicon `mps` (with automatic `cpu` fallback), shared across the independent `front` and `back` camera pipelines.

---

## 2. Spatial Rider-to-Motorcycle Coupler (`MotoRiderCoupler`) Decisions

### 2.1 Geometric Formulation
Given a candidate head detection $h$ and candidate motorcycle detection $m$ (with width $W_m$, height $H_m$, top edge $y_{1,m}$, and horizontal center $c_{x,m}$), [`MotoRiderCoupler`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L168-L450) evaluates four geometric compatibility features:

1. **Plausible Upper-Saddle Cranial Cone Inclusion ($\text{IoA}_{\text{cone}} \in [0, 1]$)**:
   $$\mathcal{C}_m = \left[x_{1,m} - \alpha_x W_m,\, x_{2,m} + \alpha_x W_m\right] \times \left[y_{1,m} - \alpha_y H_m,\, y_{1,m} + \beta_y H_m\right]$$
   where $\alpha_x = 0.25$, $\alpha_y = 1.25$, and $\beta_y = 0.55$. $\text{IoA}_{\text{cone}}(h, m) = \frac{|B_h \cap \mathcal{C}_m|}{|B_h|}$ measures the fraction of the head box inside the motorcycle's upper rider envelope.
2. **Mounted-Person Bridge Score ($S_{\text{person}} \in [0, 1]$) with Gated 1-to-1 Hungarian Assignment**:
   - A `person` box $p$ is considered mounted on motorcycle $m$ if its horizontal center lies within $\mathcal{C}_m$ and $\text{IoU}(p, m) \ge 0.12$ or $\text{IoA}(p, m) \ge 0.25$.
   - A head $h$ matches person $p$ if at least $35\%$ of $B_h$ lies inside the upper $45\%$ cranial region of $B_p$ (`head_in_person_min_ioa = 0.35`). The cranial affinity $A_{\text{head,person}}(h, p) = 0.55\,\text{IoA}_{\text{upper}} + 0.25\,S_{\text{lat}} + 0.20\,S_{\text{apex}}$ combines upper-region inclusion, horizontal center alignment, and vertical proximity to the person box apex $y_{1,p}$.
   - **Strict One-to-One Bipartite Matching (`assign_heads_to_persons_bipartite`)**: [`MotoRiderCoupler.assign_heads_to_persons_bipartite`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L236-L337) solves minimum-cost bipartite matching (`scipy.optimize.linear_sum_assignment`) over $1 - A_{\text{head,person}}(h_i, p_j)$ across all detected heads and persons in the frame:
     - **One person cannot be assigned to multiple heads**, and **one head cannot be assigned to multiple persons**.
     - **Competitive Ambiguity Abstention (`head_person_ambiguity_margin = 0.08`, engineering hypothesis)**: If an unmatched head competes for the same person within $0.08$, or a head sits ambiguously between two unmatched persons within $0.08$, or a $2\times 2$ cross-assignment swap changes total affinity by $< 0.08$, the head-to-person link **abstains** (`matched_person_idx = None`, `head_person_abstained = True`) rather than forcing an arbitrary association.
     - **Support for Unmatched Heads & Multi-Rider Motorcycles**: Any head without an assigned person box (either because the pillion torso is occluded behind the driver or because the person bridge abstained) remains fully eligible for direct geometric motorcycle coupling via the `w_no_person_*` weights (`0.55` cone IoA, `0.30` lateral alignment, `0.15` vertical scale), or becomes an [`OrphanRiderObservation`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L127-L153) if no motorcycle couples.
   - When a mounted person uniquely bridges $h$ and $m$, $S_{\text{person}} = \sqrt{A_{\text{person,moto}} \cdot A_{\text{head,person}}}$.
3. **Lateral Center Alignment ($S_{\text{lat}} \in [0, 1]$)**:
   $$d_x = \frac{|c_{x,h} - c_{x,m}|}{(0.5 + \alpha_x) W_m}, \quad S_{\text{lat}} = \max\!\left(0,\, 1 - d_x^2\right)$$
   Hard rejection gate: if $d_x > 1.15$, coupling is rejected (`LATERAL_OUT_OF_BOUNDS`).
4. **Vertical Position & Scale Plausibility ($S_{\text{vert}} \in [0, 1]$)**:
   Hard rejection gates reject heads outside $[y_{1,m} - \alpha_y H_m,\, y_{1,m} + \beta_y H_m]$ (`VERTICAL_OUT_OF_BOUNDS`) or with implausible head-to-motorcycle width ratio $W_h / W_m > 1.15$ (`HEAD_SCALE_IMPLAUSIBLE`).

### 2.2 Composite Score, Abstention Gates & Slot Ordering
- **Composite Coupling Score**:
  - With mounted `person` bridge: $S_{\text{couple}} = 0.40\,\text{IoA}_{\text{cone}} + 0.30\,S_{\text{person}} + 0.20\,S_{\text{lat}} + 0.10\,S_{\text{vert}}$.
  - Without `person` bridge (when rider body is occluded): $S_{\text{couple}} = 0.55\,\text{IoA}_{\text{cone}} + 0.30\,S_{\text{lat}} + 0.15\,S_{\text{vert}}$.
- **Three Safety & Abstention Gates**:
  1. **Minimum Coupling Threshold ($\tau_{\text{min}} = 0.35$)**: Heads scoring below $0.35$ (or in frames with zero detected motorcycles) become explicit [`OrphanRiderObservation`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/moto_rider_coupler.py#L121-L147) records (`BELOW_MIN_COUPLING_SCORE` or `NO_MOTORCYCLE_IN_FRAME`).
  2. **Adjacent-Motorcycle Ambiguity Margin ($\delta_{\text{ambig}} = 0.08$)**: When a rider head sits between two adjacent motorcycles such that both score $\ge 0.35$ and $S_{\text{best}} - S_{\text{2nd}} < 0.08$, the coupler **abstains** from forced assignment and emits an `OrphanRiderObservation` with `rejection_reason = "AMBIGUOUS_ADJACENT_MOTORCYCLES"`.
  3. **Capacity Cap (`max_riders_per_motorcycle = 3`)**: Supports 1, 2, or 3 riders per motorcycle (`driver`, `pillion_1`, `pillion_2`); any 4th candidate is demoted to `EXCEEDED_MAX_RIDERS_PER_MOTO`.
- **Evidence-Gated Slot Ordering**:
  - For solo riders ($K=1$): assigned `slot_role = "driver"`, `ordering_confident = True`.
  - For multi-rider motorcycles ($K \in \{2, 3\}$): ordered along the longitudinal camera projection axis (descending $c_y$ in oncoming Front view; ascending $c_y$ in receding Back view) **if and only if** pairwise vertical or scale separation is $\ge 8.0\text{ px}$ (`slot_ordering_min_separation_px`). Otherwise, `ordering_confident = False` and roles are labeled `"ambiguous_slot_0"`, `"ambiguous_slot_1"`.

---

## 3. Motorcycle Kalman Filter, Two-Stage ByteTrack, Rider Identity Protection & Phase 10 Integration

### 3.1 Parameter Table (All Thresholds Are Engineering Hypotheses)

| Parameter | Default Value | Engineering Rationale |
| :--- | :---: | :--- |
| `high_conf_threshold` | `0.45` | Stage 1 ByteTrack pool for primary Hungarian IoU association. |
| `low_conf_threshold` | `0.18` | Stage 2 ByteTrack pool for recovering occluded or motion-blurred motorcycles. |
| `new_track_min_conf` | `0.45` | Prevents low-confidence background noise from spawning spurious `TENTATIVE` tracks. |
| `stage1_iou_threshold` | `0.25` | Minimum IoU between Kalman prediction and high-confidence detection. |
| `stage2_iou_threshold` | `0.20` | Slightly relaxed IoU gate for Stage 2 recovery of confirmed/lost tracks. |
| `min_confirm_hits` | `2` | Frames required to transition `TENTATIVE -> CONFIRMED`. |
| `max_lost_frames` | `10` | Maximum consecutive missed frames in `LOST` before `TERMINATED`. |
| `kalman_std_weight_pos` | `0.05` (`1/20`) | Externalized Kalman position process/measurement noise scale relative to box height $h$. |
| `kalman_std_weight_vel` | `0.00625` (`1/160`) | Externalized Kalman velocity process noise scale relative to box height $h$. |
| `rider_match_min_iou` | `0.15` | Minimum IoU between predicted `RiderSubTrack` box and incoming head box to accept a slot-consistent or partial-visibility match. |
| `rider_spatial_override_min_iou` | `0.35` | Minimum disjoint spatial IoU required to override a conflicting geometric `slot_index` when two riders exchange apparent vertical ordering. |
| `rider_ambiguity_iou_margin` | `0.15` | Minimum IoU margin $\text{IoU}_{\text{best}} - \text{IoU}_{\text{2nd}}$ between competing `RiderSubTrack`s; below this margin, the tracker abstains to prevent Phase 10 history contamination. |
| `rider_max_norm_center_dist` | `0.65` | Maximum center distance normalized by predicted head diagonal when `0.05 <= IoU < 0.15`. |
| `Phase 10 Aggregator` | `ema` ($\alpha=0.35$, $W=10$) | Preserves exact Phase 10 [`EMAAggregator`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/confidence_aggregator.py#L87-L133) per `RiderSubTrack` and `OrphanRiderTrack`. |
| `Phase 10 StateMachine` | `confirm_frames=5`, `confirm_thresh=0.70` | Preserves exact Phase 10 [`TemporalStateMachine`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/state_machine.py#L42-L225) hysteresis per individual rider. |
| `Phase 10 Cooldown` | `cooldown_frames=90` | Preserves [`ViolationDecisionEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/violation_decision.py#L46-L131) duplicate-alert suppression per rider. |

### 3.2 Conservative `RiderSubTrack` Identity Protection Policy (`MotoTrack._sync_riders_on_match`)

To prevent transferring a rider's accumulated Phase 10 helmet history to another rider when geometric slot ordering flips or when one rider is temporarily occluded, [`MotoTrack._sync_riders_on_match`](file:///Users/sayemuddin/Desktop/research_helmet/src/tracking/camera_tracker.py#L262-L476) enforces a 6-step spatial-first identity protection protocol:

1. **Spatial Prediction Comparison**: Every incoming coupled rider observation $o_j$ is compared against the Kalman-predicted bounding boxes $\hat{B}_{s_i}$ of all existing `RiderSubTrack`s on the parent motorcycle, computing $\text{IoU}(s_i, o_j)$ and diagonal-normalized center distance $d_{\text{norm}}(s_i, o_j)$.
2. **Cross-Subtrack Spatial Ambiguity Abstention**: If an observation overlaps or lies close to two existing `RiderSubTrack`s such that $\text{IoU}_{\text{best}} - \text{IoU}_{\text{2nd}} < \text{rider\_ambiguity\_iou\_margin}$ (`0.15`), or two observations compete within `0.15` for the same `RiderSubTrack`, the tracker **abstains** (`rider_abstained_updates += 1`, `abstained_updates_count += 1`), calling `update_unmatched` on the affected subtracks rather than risking cross-rider contamination.
3. **Bipartite Spatial Matching Before Slot Acceptance**: Candidate matches are solved via `linear_sum_assignment` on the gated spatial affinity matrix. Even when `obs.slot_index == subtrack.slot_index` and `ordering_confident == True`, the match is accepted **only if** it passes the spatial consistency gate ($\text{IoU} \ge 0.15$, or $\text{IoU} \ge 0.05$ with $d_{\text{norm}} \le 0.65$).
4. **Conflict Detection & Conservative Resolution (`obs.slot_index != subtrack.slot_index`)**:
   - **Partial Visibility (`K < M`, e.g., Driver occluded while Pillion remains visible)**: When fewer riders are detected than existing subtracks, `MotoRiderCoupler` assigns `slot_index = 0` to the single visible head. If that head has an unambiguous spatial match ($\text{IoU} \ge 0.15$, competing subtrack $\text{IoU} < 0.05$, margin $\ge 0.15$) to an existing subtrack $s_k$ (e.g., `s1: pillion_1`), the observation updates $s_k$ (`preserve_slot_role = True`) while the occluded rider (`s0: driver`) transitions to `TEMPORARILY_OCCLUDED`.
   - **Multi-Rider Apparent Ordering Exchange (`K >= 2`)**: When two riders exchange apparent vertical ordering in image space, spatial assignment overrides the swapped `slot_index` **if and only if** the spatial match is high-confidence and mutually disjoint ($\text{IoU} \ge \text{rider\_spatial\_override\_min\_iou} = 0.35$ and competing subtrack $\text{IoU} < 0.05$), preserving each rider's original `slot_role` (`driver` vs. `pillion_1`) and separate Phase 10 history.
   - **Unresolvable Conflict Abstention**: If `obs.slot_index != subtrack.slot_index` and the high-confidence disjoint spatial override criterion is not met (or if an observation claiming slot $s_k$ jumped outside $s_k$'s spatial gate), the tracker **abstains** (`rider_abstained_updates += 1`) and advances the affected subtrack via `update_unmatched`.
5. **Duplicate Track Prevention**: After initial track creation, a new `RiderSubTrack` slot is spawned for an unmatched observation **only if** all existing `RiderSubTrack`s on the motorcycle are already matched in the current frame (`len(matched_slots) == len(self.rider_subtracks)`) and no slot abstained, preventing displaced or conflicting detections from creating duplicate rider tracks.
