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
2. **Mounted-Person Bridge Score ($S_{\text{person}} \in [0, 1]$)**:
   - A `person` box $p$ is considered mounted on motorcycle $m$ if its horizontal center lies within $\mathcal{C}_m$ and $\text{IoU}(p, m) \ge 0.12$ or $\text{IoA}(p, m) \ge 0.25$.
   - A head $h$ matches person $p$ if at least $35\%$ of $B_h$ lies inside the upper $45\%$ cranial region of $B_p$.
   - When a mounted person bridges $h$ and $m$, $S_{\text{person}} = \sqrt{A_{\text{person,moto}} \cdot A_{\text{head,person}}}$.
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

## 3. Motorcycle Kalman Filter, Two-Stage ByteTrack & Phase 10 Integration

| Parameter | Default Value | Engineering Rationale |
| :--- | :---: | :--- |
| `high_conf_threshold` | `0.45` | Stage 1 ByteTrack pool for primary Hungarian IoU association. |
| `low_conf_threshold` | `0.18` | Stage 2 ByteTrack pool for recovering occluded or motion-blurred motorcycles. |
| `new_track_min_conf` | `0.45` | Prevents low-confidence background noise from spawning spurious `TENTATIVE` tracks. |
| `stage1_iou_threshold` | `0.25` | Minimum IoU between Kalman prediction and high-confidence detection. |
| `stage2_iou_threshold` | `0.20` | Slightly relaxed IoU gate for Stage 2 recovery of confirmed/lost tracks. |
| `min_confirm_hits` | `2` | Frames required to transition `TENTATIVE -> CONFIRMED`. |
| `max_lost_frames` | `10` | Maximum consecutive missed frames in `LOST` before `TERMINATED`. |
| `Phase 10 Aggregator` | `ema` ($\alpha=0.35$, $W=10$) | Preserves exact Phase 10 [`EMAAggregator`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/confidence_aggregator.py#L87-L133) per `RiderSubTrack` and `OrphanRiderTrack`. |
| `Phase 10 StateMachine` | `confirm_frames=5`, `confirm_thresh=0.70` | Preserves exact Phase 10 [`TemporalStateMachine`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/state_machine.py#L42-L225) hysteresis per individual rider. |
| `Phase 10 Cooldown` | `cooldown_frames=90` | Preserves [`ViolationDecisionEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/temporal/violation_decision.py#L46-L131) duplicate-alert suppression per rider. |
