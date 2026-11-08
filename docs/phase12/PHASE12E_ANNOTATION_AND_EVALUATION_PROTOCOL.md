# PHASE 12E — CROSS-CAMERA ANNOTATION AND EVALUATION PROTOCOL

**Project:** *An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification*  
**Phase:** Phase 12E — Cross-Camera Tracklet Association  
**Related Protocol:** [`docs/phase12/PHASE12B_DATA_ANNOTATION_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/PHASE12B_DATA_ANNOTATION_PROTOCOL.md)

---

## 1. Scientific Scope & Ground-Truth Requirement

Cross-camera identity association accuracy cannot be claimed from synthetic unit tests or unannotated operational diagnostics. To move the parameters in [`configs/cross_camera_association_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/cross_camera_association_config.yaml) from `engineering_hypothesis: true` to empirically validated parameters, a human-annotated cross-camera ground-truth dataset must be constructed across the 25 synchronized video pairs (`pair_001` through `pair_025`).

---

## 2. Three-Tier Ground-Truth Annotation Schema

### 2.1 Tier 1: Motorcycle-Level Cross-Camera Identity (`global_moto_gt_id`)
For each synchronized video pair (`pair_XXX`):
1. Every physical motorcycle entering either the Front or Back camera within the common synchronized overlap interval $[t_{\text{overlap,start}}, t_{\text{overlap,end}}]$ is assigned a unique ground-truth vehicle identifier `gt_moto_XXX_NN` (e.g., `gt_moto_001_01`).
2. Annotators record:
   - `front_entry_frame_idx`, `front_exit_frame_idx` (or `null` if visible only in Back).
   - `back_entry_frame_idx`, `back_exit_frame_idx` (or `null` if visible only in Front).
   - `visibility_regime`:
     - `BOTH_VIEWS_COMPLETE`: Clearly visible as a motorcycle in both cameras.
     - `FRONT_ONLY_OR_BACK_OCCLUDED`: Motorcycle chassis occluded or distant in Back (may appear only as an orphan head/person in Back).
     - `BACK_ONLY_OR_FRONT_OCCLUDED`: Visible only in Back.
     - `AMBIGUOUS_HUMAN_UNRESOLVABLE`: Dense traffic where human annotators cannot establish cross-camera correspondence with certainty (excluded from binary positive/negative link scoring, used to evaluate `AMBIGUOUS_ABSTAIN`).

### 2.2 Tier 2: Rider-Level Cross-Camera Identity & Seating Role (`global_rider_gt_id`)
Within each annotated `gt_moto_XXX_NN`:
1. Every rider seated on the motorcycle is assigned a unique rider ground-truth identifier `gt_rider_XXX_NN_sK` with seating role:
   - `driver` (`s0`): Primary operator controlling handlebars.
   - `pillion_1` (`s1`): First passenger behind the driver.
   - `pillion_2` (`s2`): Second passenger (triple-riding violation context).
2. Annotators record per-camera visibility (`front_visible: bool`, `back_visible: bool`) and per-camera ground-truth helmet state (`gt_helmet_state: "helmet" | "no_helmet" | "occluded"`).
3. **Strict Independence Rule:** Annotators must verify rider identity using seating position, clothing/torso appearance, and vehicle trajectory—never by assuming that predicted helmet labels match across views.

### 2.3 Tier 3: Orphan Rider Ground-Truth Correspondence
When a rider's head is visible in a camera where the parent motorcycle bounding box is occluded behind traffic or below the detector's resolution threshold:
1. Annotators link the orphan head trajectory to its true `gt_rider_XXX_NN_sK` identity and label `gt_entity_level = "ORPHAN_RIDER"`.
2. This enables separate evaluation of [`OrphanAssociationEngine`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/orphan_association.py) (`ORPHAN_LEVEL_LINK`) vs. full motorcycle-coupled association.

---

## 3. Quantitative Evaluation Metrics

Once ground-truth annotations are completed, Phase 12E must be evaluated using the following metrics:

1. **Motorcycle Cross-Camera Association Precision, Recall, and $F_1$:**
   $$\text{Precision}_{\text{moto}} = \frac{\text{TP}_{\text{linked}}}{\text{TP}_{\text{linked}} + \text{FP}_{\text{linked}}}, \quad \text{Recall}_{\text{moto}} = \frac{\text{TP}_{\text{linked}}}{\text{TP}_{\text{linked}} + \text{FN}_{\text{unmatched}}}$$
   where $\text{FP}_{\text{linked}}$ counts false cross-camera merges (identity switches across cameras).
2. **Selective Risk-Coverage & Abstention Calibration:**
   - **Abstention Rate:** Fraction of ground-truth pairs assigned `AMBIGUOUS_ABSTAIN` or `INSUFFICIENT_EVIDENCE`.
   - **Error Rate on Non-Abstained Links:** $\text{FP}_{\text{linked}} / (\text{TP}_{\text{linked}} + \text{FP}_{\text{linked}})$ as a function of `min_association_score` and `ambiguous_margin_threshold`.
3. **Rider-Level Cross-Camera Role Preservation Accuracy:**
   - Percentage of `LINKED` motorcycles in which `driver -> driver`, `pillion_1 -> pillion_1`, and `pillion_2 -> pillion_2` are matched without driver-pillion identity swaps.
4. **Cue Ablation Study:**
   - Evaluate $\text{Precision}_{\text{moto}}$, $\text{Recall}_{\text{moto}}$, and $\text{IDF1}$ across:
     - Cue A only (`temporal_transit`).
     - Cue A + Cue B (`temporal_transit` + `rider_cardinality`).
     - Cue A + Cue B + Cue C (`+ appearance_hsv`).
     - Cue A + Cue B + Cue C + Cue E (`+ convoy_order`).
