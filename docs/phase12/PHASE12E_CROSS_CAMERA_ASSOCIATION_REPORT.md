# PHASE 12E — CROSS-CAMERA TRACKLET ASSOCIATION REPORT

**Project:** *An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification*  
**Phase:** Phase 12E — Cross-Camera Tracklet Association  
**Starting Baseline Commit:** `39c1834` (Approved Phase 12D Review Corrections)  
**Configuration File:** [`configs/cross_camera_association_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/cross_camera_association_config.yaml)  
**Package Location:** [`src/association/`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/__init__.py)

---

## 1. Executive Summary

Phase 12E implements a conservative, explainable, multi-cue cross-camera tracklet association system that links independently tracked Front-camera and Back-camera motorcycle trajectories (`front_moto_*` $\leftrightarrow$ `back_moto_*`), their coupled rider subtracks (`front_moto_*_rider_s*` $\leftrightarrow$ `back_moto_*_rider_s*`), and uncoupled orphan rider trajectories (`front_orphan_*` $\leftrightarrow$ `back_orphan_*`) across synchronized dual-camera streams.

Key architectural guarantees enforced across [`src/association/`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/__init__.py):
1. **Strict Scope Discipline:** Phase 12E performs cross-camera identity association only. It does **not** implement license plate detection or OCR (Phase 12F), does **not** implement cross-camera violation confidence fusion (Phase 12G), and does **not** modify Phase 9 (`best.pt`), Phase 10 (`src/temporal/`), or Phase 11.6/12C synchronization metadata.
2. **Observed vs. Predicted Frame Separation:** [`TrackletBuilder`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_builder.py) separates physical matched observations ([`ObservedFrameRecord`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py)) from purely Kalman-predicted (`LOST`) frames (`predicted_only_frame_count`, `occlusion_intervals`). Purely predicted frames are never used as measured bounding-box geometry or appearance crops.
3. **Metrological Separation of Synchronization & Transit Uncertainties:** [`TransitTimeModel`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/transit_time_model.py) and [`CandidatePairScore`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py) maintain five separate terms in all outputs and audit logs:
   - `epsilon_sync_ms`: Phase 11.6 synchronization residual (`3.0 ms` to `38.0 ms`).
   - `delta_pts_front_ms`: Front container PTS discretization residual.
   - `delta_pts_back_ms`: Back container PTS discretization residual.
   - `delta_transit_sec`: Physical cross-camera transit delay ($t_{\text{back,centroid}} - t_{\text{front,centroid}}$) on the canonical reference clock.
   - `transit_uncertainty_sec`: Physical vehicle speed and corridor transit uncertainty (`0.35 s`).
4. **Missing-Cue Renormalization:** [`AssociationScorer`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/association_scorer.py) evaluates up to five cues (`temporal_transit`, `rider_cardinality`, `appearance_hsv`, `lane_spatial_context`, `convoy_order`), excludes unavailable cues (`available = False`), and renormalizes active weights to sum to `1.0` so missing cues never become silent zero penalties.
5. **Helmet Label Independence:** Helmet vs. no-helmet classification labels (`head_class_name`) are never used as an identity feature in motorcycle, rider, or orphan association (`test_12_helmet_class_difference_never_used_as_identity_feature`).
6. **Conservative 4-State Decision Outcomes:** Every motorcycle, rider, and orphan association produces one of four explicit outcomes (`AssociationOutcome`): `LINKED`, `UNMATCHED`, `AMBIGUOUS_ABSTAIN`, or `INSUFFICIENT_EVIDENCE`.
7. **Preservation of Independent Phase 10 Temporal Verification States:** Cross-camera linking preserves Front and Back `RiderSubTrack` IDs, `slot_role`s, and Phase 10 states as independent read-only snapshots (`front_phase10_readonly`, `back_phase10_readonly`) without merging or mutating per-camera Phase 10 state machines (`test_20_phase10_temporal_states_remain_strictly_independent_after_linking`).

---

## 2. Implemented Module Inventory (`src/association/`)

| Module | Primary Classes / Functions | Responsibility |
|---|---|---|
| [`src/association/tracklet_models.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py) | `AssociationOutcome`, `TrackletCompleteness`, `TimestampQualityRecord`, `ObservedFrameRecord`, `MotorcycleTracklet`, `RiderTracklet`, `OrphanRiderTracklet`, `CueScoreDetail`, `CandidatePairScore`, `MotorcycleAssociationResult`, `RiderAssociationResult`, `OrphanAssociationResult`, `CrossCameraAssociationBatchResult` | Versioned (`schema_version = "1.0.0"`), JSON-serializable dataclasses for tracklets, cues, and decisions |
| [`src/association/tracklet_builder.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_builder.py) | `TrackletBuilder`, `TrackletBuilderConfig` | Extracts deduplicated tracklets from `CameraTrackingPipeline`, converts timestamps to canonical $t_{\text{ref}}$, separates observed vs. predicted frames, and computes `occlusion_intervals` |
| [`src/association/transit_time_model.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/transit_time_model.py) | `TransitTimeModel`, `TransitTimeConfig`, `TransitEvaluationResult` | Evaluates Cue A (`temporal_transit`) across `bidirectional_corridor`, `front_to_back`, and `back_to_front` regimes while separating clock vs. transit uncertainties |
| [`src/association/appearance_features.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/appearance_features.py) | `AppearanceConfig`, `compute_crop_quality`, `compute_hsv_achromatic_histogram`, `extract_tracklet_appearance_descriptor`, `score_appearance_similarity` | Optional Cue C (`appearance_hsv`) on observed crops passing resolution (`>= 20x20 px`) and Laplacian sharpness (`>= 12.0`) checks |
| [`src/association/association_features.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/association_features.py) | `evaluate_rider_cardinality_cue`, `evaluate_lane_spatial_cue`, `evaluate_convoy_order_cue` | Evaluates Cue B (`rider_cardinality` with partial pillion occlusion tolerance), Cue D (`lane_spatial_context`, disabled without calibration), and Cue E (`convoy_order`) |
| [`src/association/candidate_generator.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/candidate_generator.py) | `CandidateGenerator`, `CandidatePairEvaluation` | Gates candidate `(FrontTracklet, BackTracklet)` pairs with explicit rejection reasons |
| [`src/association/association_scorer.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/association_scorer.py) | `AssociationScorer`, `AssociationScorerConfig` | Computes multi-cue scores and renormalizes weights over available cues |
| [`src/association/global_assignment.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/global_assignment.py) | `GlobalAssignmentSolver`, `GlobalAssignmentConfig` | Solves 1-to-1 Hungarian assignment with row/column and $2 \times 2$ swap margin abstention (`AMBIGUOUS_ABSTAIN`) |
| [`src/association/rider_association.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/rider_association.py) | `RiderAssociationEngine`, `RiderAssociationConfig` | Associates `RiderTracklet`s within `LINKED` motorcycles while keeping Phase 10 states independent |
| [`src/association/orphan_association.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/orphan_association.py) | `OrphanAssociationEngine`, `OrphanAssociationConfig` | First-class association of `OrphanRiderTracklet`s labeled `ORPHAN_LEVEL_LINK` (`implies_confirmed_motorcycle_link = False`) |
| [`src/association/association_audit.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/association_audit.py) | `AssociationAuditor` | Builds flat tabular audit rows, cue score distributions, and JSON/CSV exports |
| [`src/association/cross_camera_association.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/cross_camera_association.py) | `CrossCameraAssociationEngine`, `CrossCameraAssociationConfig` | End-to-end orchestrator loaded from [`configs/cross_camera_association_config.yaml`](file:///Users/sayemuddin/Desktop/research_helmet/configs/cross_camera_association_config.yaml) |
| [`src/association/__init__.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/__init__.py) | Package exports | Public API surface |

---

## 3. Verification & Test Suite Summary

- **Phase 12E Unit & Integration Tests ([`tests/test_phase12e_association.py`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12e_association.py)):** `26 / 26` passed (`3.22 s`), covering all 25 required synthetic scenarios plus YAML configuration override propagation.
- **Full Repository Test Suite (`tests/`):** `76 / 76` passed (`100%` pass rate across Phase 10, Phase 12A, Phase 12C, Phase 12D, and Phase 12E).
- **Controlled Real-Video Operational Diagnostics ([`scripts/phase12e_operational_diagnostics.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/phase12e_operational_diagnostics.py)):** Executed on `pair_001`, `pair_013` (VFR), `pair_015` (60/30 fps), and `pair_019` (`96` canonical 60 Hz ticks per pair in the active transit window), demonstrating `LINKED` (`pair_015`, `pair_019`), `AMBIGUOUS_ABSTAIN` (`pair_013` motorcycle level, `pair_001` orphan level), `UNMATCHED` (`pair_001` motorcycle level, `pair_019` single-view pillion), and `INSUFFICIENT_EVIDENCE` (`pair_015` 1-frame rider subtrack, `pair_001` 1-frame orphans) with sub-millisecond core solver overhead (`0.26 ms` to `1.07 ms`) and `1458.81 MB` peak RSS.
