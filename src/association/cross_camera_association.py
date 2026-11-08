"""Main Cross-Camera Tracklet Association Engine (`CrossCameraAssociationEngine`).

Orchestrates the complete Phase 12E pipeline:
1. Loads configuration from `configs/cross_camera_association_config.yaml` (supporting
   both `{value: X, engineering_hypothesis: true}` and flat YAML entries).
2. Extracts `MotorcycleTracklet`, `RiderTracklet`, and `OrphanRiderTracklet` objects
   from Front and Back `CameraTrackingPipeline` instances via `TrackletBuilder`.
3. Generates candidate `(FrontTracklet, BackTracklet)` pairs via `CandidateGenerator`.
4. Computes multi-cue scores (`Cue A..E`) with missing-cue renormalization via
   `AssociationScorer`.
5. Solves global 1-to-1 assignment with conservative abstention (`LINKED`, `UNMATCHED`,
   `AMBIGUOUS_ABSTAIN`, `INSUFFICIENT_EVIDENCE`) via `GlobalAssignmentSolver`.
6. Performs rider-level association within `LINKED` motorcycle pairs via
   `RiderAssociationEngine` without merging or mutating per-camera Phase 10 states.
7. Performs first-class orphan rider tracklet association via `OrphanAssociationEngine`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any
import numpy as np
import yaml

from src.association.appearance_features import AppearanceConfig
from src.association.association_features import (
    ConvoyOrderConfig,
    LaneSpatialConfig,
    RiderCardinalityConfig,
)
from src.association.association_scorer import (
    AssociationScorer,
    AssociationScorerConfig,
)
from src.association.candidate_generator import CandidateGenerator
from src.association.global_assignment import (
    GlobalAssignmentConfig,
    GlobalAssignmentSolver,
)
from src.association.orphan_association import (
    OrphanAssociationConfig,
    OrphanAssociationEngine,
)
from src.association.rider_association import (
    RiderAssociationConfig,
    RiderAssociationEngine,
)
from src.association.tracklet_builder import TrackletBuilder, TrackletBuilderConfig
from src.association.tracklet_models import (
    AssociationOutcome,
    CrossCameraAssociationBatchResult,
    MotorcycleTracklet,
    OrphanRiderTracklet,
    SCHEMA_VERSION,
    TimestampQualityRecord,
)
from src.association.transit_time_model import TransitTimeConfig, TransitTimeModel
from src.tracking.camera_tracker import CameraTrackingPipeline


def _unwrap_val(mapping: dict[str, Any], key: str, default: Any) -> Any:
    """Extract `value` from `{value: X, engineering_hypothesis: true}` or direct scalar."""
    if not isinstance(mapping, dict) or key not in mapping:
        return default
    raw = mapping[key]
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


@dataclass
class CrossCameraAssociationConfig:
    """Top-level configuration for `CrossCameraAssociationEngine`."""
    schema_version: str = SCHEMA_VERSION
    builder_config: TrackletBuilderConfig = field(default_factory=TrackletBuilderConfig)
    transit_config: TransitTimeConfig = field(default_factory=TransitTimeConfig)
    scorer_config: AssociationScorerConfig = field(default_factory=AssociationScorerConfig)
    global_assignment_config: GlobalAssignmentConfig = field(
        default_factory=GlobalAssignmentConfig
    )
    rider_association_config: RiderAssociationConfig = field(
        default_factory=RiderAssociationConfig
    )
    orphan_association_config: OrphanAssociationConfig = field(
        default_factory=OrphanAssociationConfig
    )


class CrossCameraAssociationEngine:
    """
    Complete Phase 12E Cross-Camera Tracklet Association Engine.
    """

    def __init__(self, config: CrossCameraAssociationConfig | None = None) -> None:
        self.config = config or CrossCameraAssociationConfig()
        self.builder = TrackletBuilder(config=self.config.builder_config)
        self.transit_model = TransitTimeModel(config=self.config.transit_config)
        self.candidate_generator = CandidateGenerator(
            transit_model=self.transit_model,
            min_observed_frames=self.config.builder_config.min_observed_frames_moto,
            require_in_overlap=True,
        )
        self.scorer = AssociationScorer(
            transit_model=self.transit_model,
            config=self.config.scorer_config,
        )
        self.rider_engine = RiderAssociationEngine(
            config=self.config.rider_association_config
        )
        self.global_solver = GlobalAssignmentSolver(
            config=self.config.global_assignment_config,
            rider_engine=self.rider_engine,
        )
        self.orphan_engine = OrphanAssociationEngine(
            transit_model=self.transit_model,
            config=self.config.orphan_association_config,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path = "configs/cross_camera_association_config.yaml",
    ) -> CrossCameraAssociationEngine:
        """Load `CrossCameraAssociationEngine` from `configs/cross_camera_association_config.yaml`."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        app_raw = raw.get("appearance_features", {})
        zw_raw = app_raw.get("zone_weights", {})
        app_cfg = AppearanceConfig(
            enabled=bool(_unwrap_val(app_raw, "enabled", True)),
            min_crop_width_px=int(_unwrap_val(app_raw, "min_crop_width_px", 20)),
            min_crop_height_px=int(_unwrap_val(app_raw, "min_crop_height_px", 20)),
            min_laplacian_sharpness=float(_unwrap_val(app_raw, "min_laplacian_sharpness", 12.0)),
            min_valid_crop_quality=float(_unwrap_val(app_raw, "min_valid_crop_quality", 0.35)),
            hsv_h_bins=int(_unwrap_val(app_raw, "hsv_h_bins", 12)),
            hsv_s_bins=int(_unwrap_val(app_raw, "hsv_s_bins", 4)),
            achromatic_v_bins=int(_unwrap_val(app_raw, "achromatic_v_bins", 8)),
            low_saturation_threshold=int(_unwrap_val(app_raw, "low_saturation_threshold", 45)),
            weight_cranial_cap=float(_unwrap_val(zw_raw, "cranial_cap", 0.30)),
            weight_upper_torso=float(_unwrap_val(zw_raw, "upper_torso", 0.45)),
            weight_moto_chassis=float(_unwrap_val(zw_raw, "moto_chassis", 0.25)),
        )

        tb_raw = raw.get("tracklet_builder", {})
        builder_cfg = TrackletBuilderConfig(
            min_observed_frames_moto=int(_unwrap_val(tb_raw, "min_observed_frames_moto", 2)),
            min_observed_frames_rider=int(_unwrap_val(tb_raw, "min_observed_frames_rider", 2)),
            min_observed_frames_orphan=int(_unwrap_val(tb_raw, "min_observed_frames_orphan", 2)),
            min_observation_density=float(_unwrap_val(tb_raw, "min_observation_density", 0.30)),
            border_margin_px=float(_unwrap_val(tb_raw, "border_margin_px", 45.0)),
            exclude_tentative_tracks=bool(_unwrap_val(tb_raw, "exclude_tentative_tracks", False)),
            appearance_config=app_cfg,
        )

        tt_raw = raw.get("transit_time_model", {})
        # Support case-insensitive key for max_temporal_search_window_sec
        max_search_sec = _unwrap_val(
            tt_raw,
            "max_temporal_search_window_sec",
            _unwrap_val(tt_raw, "max_temporal_Search_window_sec", 8.00),
        )
        transit_cfg = TransitTimeConfig(
            corridor_mode=str(_unwrap_val(tt_raw, "corridor_mode", "bidirectional_corridor")),
            min_transit_delay_sec=float(_unwrap_val(tt_raw, "min_transit_delay_sec", -3.50)),
            max_transit_delay_sec=float(_unwrap_val(tt_raw, "max_transit_delay_sec", 6.00)),
            expected_transit_delay_sec=float(_unwrap_val(tt_raw, "expected_transit_delay_sec", 0.75)),
            transit_sigma_sec=float(_unwrap_val(tt_raw, "transit_sigma_sec", 1.50)),
            transit_uncertainty_sec=float(_unwrap_val(tt_raw, "transit_uncertainty_sec", 0.35)),
            max_temporal_search_window_sec=float(max_search_sec),
            allow_simultaneous_fov_overlap=bool(_unwrap_val(tt_raw, "allow_simultaneous_fov_overlap", True)),
            max_allowed_sync_residual_ms=float(_unwrap_val(tt_raw, "max_allowed_sync_residual_ms", 100.0)),
        )

        rc_raw = raw.get("rider_cardinality", {})
        diff1_occ = _unwrap_val(
            rc_raw,
            "partial_pillion_occlusion_diff1_score",
            _unwrap_val(rc_raw, "partial_pillion_occlusion_Diff1_score", 0.65),
        )
        rc_cfg = RiderCardinalityConfig(
            exact_count_match_score=float(_unwrap_val(rc_raw, "exact_count_match_score", 1.00)),
            partial_pillion_occlusion_diff1_score=float(diff1_occ),
            unexplained_diff1_score=float(_unwrap_val(rc_raw, "unexplained_diff1_score", 0.35)),
            diff2_or_more_score=float(_unwrap_val(rc_raw, "diff2_or_more_score", 0.08)),
            partial_occlusion_visibility_threshold=float(
                _unwrap_val(rc_raw, "partial_occlusion_visibility_threshold", 0.75)
            ),
        )

        ls_raw = raw.get("lane_spatial_context", {})
        ls_cfg = LaneSpatialConfig(
            enabled=bool(_unwrap_val(ls_raw, "enabled", False)),
            require_explicit_calibration=bool(_unwrap_val(ls_raw, "require_explicit_calibration", True)),
        )

        co_raw = raw.get("convoy_order", {})
        co_cfg = ConvoyOrderConfig(
            enabled=bool(_unwrap_val(co_raw, "enabled", True)),
            convoy_window_sec=float(_unwrap_val(co_raw, "convoy_window_sec", 5.00)),
            consistent_order_score=float(_unwrap_val(co_raw, "consistent_order_score", 0.90)),
            overtake_or_inversion_score=float(_unwrap_val(co_raw, "overtake_or_inversion_score", 0.35)),
            min_anchor_temporal_gap_sec=float(_unwrap_val(co_raw, "min_anchor_temporal_gap_sec", 0.25)),
        )

        as_raw = raw.get("association_scoring", {})
        cw_raw = as_raw.get("cue_weights", {})
        ga_raw = raw.get("global_assignment", {})

        min_active_cues = int(_unwrap_val(as_raw, "min_active_cues_required", 2))
        min_temp_gate = float(_unwrap_val(ga_raw, "min_temporal_transit_score_gate", 0.20))

        scorer_cfg = AssociationScorerConfig(
            weight_temporal_transit=float(_unwrap_val(cw_raw, "temporal_transit", 0.45)),
            weight_rider_cardinality=float(_unwrap_val(cw_raw, "rider_cardinality", 0.25)),
            weight_appearance_hsv=float(_unwrap_val(cw_raw, "appearance_hsv", 0.15)),
            weight_lane_spatial_context=float(_unwrap_val(cw_raw, "lane_spatial_context", 0.05)),
            weight_convoy_order=float(_unwrap_val(cw_raw, "convoy_order", 0.10)),
            renormalize_missing_cues=bool(_unwrap_val(as_raw, "renormalize_missing_cues", True)),
            min_active_cues_required=min_active_cues,
            min_temporal_transit_score_gate=min_temp_gate,
            rider_cardinality_config=rc_cfg,
            appearance_config=app_cfg,
            lane_spatial_config=ls_cfg,
            convoy_order_config=co_cfg,
        )

        global_cfg = GlobalAssignmentConfig(
            min_association_score=float(_unwrap_val(ga_raw, "min_association_score", 0.62)),
            ambiguous_margin_threshold=float(_unwrap_val(ga_raw, "ambiguous_margin_threshold", 0.10)),
            min_temporal_transit_score_gate=min_temp_gate,
            insufficient_evidence_min_obs_frames=int(
                _unwrap_val(ga_raw, "insufficient_evidence_min_obs_frames", 2)
            ),
            min_active_cues_required=min_active_cues,
        )

        ra_raw = raw.get("rider_association", {})
        rider_cfg = RiderAssociationConfig(
            min_rider_association_score=float(_unwrap_val(ra_raw, "min_rider_association_score", 0.60)),
            rider_ambiguous_margin_threshold=float(
                _unwrap_val(ra_raw, "rider_ambiguous_margin_threshold", 0.12)
            ),
            min_observed_frames_rider=builder_cfg.min_observed_frames_rider,
            role_match_score=float(_unwrap_val(ra_raw, "role_match_score", 1.00)),
            role_ambiguous_ordering_score=float(
                _unwrap_val(ra_raw, "role_ambiguous_ordering_score", 0.55)
            ),
            role_mismatch_score=float(_unwrap_val(ra_raw, "role_mismatch_score", 0.10)),
            weight_slot_role=float(_unwrap_val(ra_raw, "weight_slot_role", 0.55)),
            weight_temporal_overlap=float(_unwrap_val(ra_raw, "weight_temporal_overlap", 0.25)),
            weight_appearance=float(_unwrap_val(ra_raw, "weight_appearance", 0.20)),
            appearance_config=app_cfg,
        )

        oa_raw = raw.get("orphan_association", {})
        orphan_cfg = OrphanAssociationConfig(
            enable_orphan_matching=bool(_unwrap_val(oa_raw, "enable_orphan_matching", True)),
            min_orphan_association_score=float(
                _unwrap_val(oa_raw, "min_orphan_association_score", 0.68)
            ),
            orphan_ambiguous_margin_threshold=float(
                _unwrap_val(oa_raw, "orphan_ambiguous_margin_threshold", 0.14)
            ),
            min_observed_frames_orphan=builder_cfg.min_observed_frames_orphan,
            require_appearance_or_strong_temporal=bool(
                _unwrap_val(oa_raw, "require_appearance_or_strong_temporal", True)
            ),
            min_strong_temporal_score_without_appearance=float(
                _unwrap_val(oa_raw, "min_strong_temporal_score_without_appearance", 0.82)
            ),
            appearance_config=app_cfg,
        )

        return cls(
            config=CrossCameraAssociationConfig(
                schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
                builder_config=builder_cfg,
                transit_config=transit_cfg,
                scorer_config=scorer_cfg,
                global_assignment_config=global_cfg,
                rider_association_config=rider_cfg,
                orphan_association_config=orphan_cfg,
            )
        )

    def associate_tracklets(
        self,
        pair_id: str,
        front_motorcycle_tracklets: list[MotorcycleTracklet],
        back_motorcycle_tracklets: list[MotorcycleTracklet],
        front_orphan_tracklets: list[OrphanRiderTracklet] | None = None,
        back_orphan_tracklets: list[OrphanRiderTracklet] | None = None,
        timestamp_quality: TimestampQualityRecord | None = None,
        calibrated_homography_available: bool = False,
        tracklet_build_ms: float = 0.0,
    ) -> CrossCameraAssociationBatchResult:
        """
        Execute candidate generation, multi-cue scoring, global 1-to-1 motorcycle
        assignment, rider-level association, and orphan association on pre-built tracklets.
        """
        front_orphans = list(front_orphan_tracklets or [])
        back_orphans = list(back_orphan_tracklets or [])
        tq = timestamp_quality or TimestampQualityRecord(pair_id=pair_id)

        t0 = time.perf_counter()
        candidates = self.candidate_generator.generate_candidates(
            front_motorcycle_tracklets, back_motorcycle_tracklets
        )
        t1 = time.perf_counter()

        candidate_scores = [
            self.scorer.score_candidate_pair(
                cand,
                all_front_tracklets=front_motorcycle_tracklets,
                all_back_tracklets=back_motorcycle_tracklets,
                calibrated_homography_available=calibrated_homography_available,
            )
            for cand in candidates
        ]
        t2 = time.perf_counter()

        moto_results = self.global_solver.solve(
            front_tracklets=front_motorcycle_tracklets,
            back_tracklets=back_motorcycle_tracklets,
            candidate_scores=candidate_scores,
        )
        t3 = time.perf_counter()

        orphan_results = self.orphan_engine.associate_orphans(
            front_orphans=front_orphans,
            back_orphans=back_orphans,
            front_motos=front_motorcycle_tracklets,
            back_motos=back_motorcycle_tracklets,
            moto_results=moto_results,
            epsilon_sync_ms=tq.epsilon_sync_ms,
            delta_pts_front_ms=tq.mean_delta_pts_front_ms,
            delta_pts_back_ms=tq.mean_delta_pts_back_ms,
        )
        t4 = time.perf_counter()

        moto_counts = Counter(o.value for o in AssociationOutcome)
        for o in AssociationOutcome:
            moto_counts[o.value] = 0
        for mr in moto_results:
            moto_counts[mr.outcome.value] += 1

        rider_counts = {o.value: 0 for o in AssociationOutcome}
        for mr in moto_results:
            for rr in mr.rider_associations:
                rider_counts[rr.outcome.value] += 1

        orphan_counts = {o.value: 0 for o in AssociationOutcome}
        for or_res in orphan_results:
            orphan_counts[or_res.outcome.value] += 1

        cand_ms = (t1 - t0) * 1000.0
        score_ms = (t2 - t1) * 1000.0
        solve_ms = (t3 - t2) * 1000.0
        orphan_ms = (t4 - t3) * 1000.0
        total_ms = tracklet_build_ms + cand_ms + score_ms + solve_ms + orphan_ms

        return CrossCameraAssociationBatchResult(
            schema_version=self.config.schema_version,
            pair_id=pair_id,
            timestamp_quality=tq,
            front_motorcycle_tracklets=front_motorcycle_tracklets,
            back_motorcycle_tracklets=back_motorcycle_tracklets,
            front_orphan_tracklets=front_orphans,
            back_orphan_tracklets=back_orphans,
            candidate_pair_scores=candidate_scores,
            motorcycle_results=moto_results,
            orphan_results=orphan_results,
            outcome_counts=dict(moto_counts),
            rider_outcome_counts=rider_counts,
            orphan_outcome_counts=orphan_counts,
            runtime_ms={
                "tracklet_build_ms": tracklet_build_ms,
                "candidate_generation_ms": cand_ms,
                "cue_scoring_ms": score_ms,
                "global_assignment_and_rider_ms": solve_ms,
                "orphan_association_ms": orphan_ms,
                "total_association_ms": total_ms,
            },
        )

    def associate_pipelines(
        self,
        pair_id: str,
        front_pipeline: CameraTrackingPipeline,
        back_pipeline: CameraTrackingPipeline,
        timestamp_quality: TimestampQualityRecord,
        front_frames_by_index: dict[int, np.ndarray] | None = None,
        back_frames_by_index: dict[int, np.ndarray] | None = None,
        front_canonical_pts_by_frame: dict[int, float] | None = None,
        back_canonical_pts_by_frame: dict[int, float] | None = None,
        front_delta_pts_ms_by_frame: dict[int, float] | None = None,
        back_delta_pts_ms_by_frame: dict[int, float] | None = None,
        calibrated_homography_available: bool = False,
    ) -> CrossCameraAssociationBatchResult:
        """
        Extract tracklets from `front_pipeline` and `back_pipeline` and execute
        full cross-camera association.
        """
        t0 = time.perf_counter()
        front_motos, front_orphans = self.builder.extract_from_pipeline(
            pipeline=front_pipeline,
            timestamp_quality=timestamp_quality,
            frame_images_by_index=front_frames_by_index,
            canonical_pts_by_frame=front_canonical_pts_by_frame,
            delta_pts_ms_by_frame=front_delta_pts_ms_by_frame,
        )
        back_motos, back_orphans = self.builder.extract_from_pipeline(
            pipeline=back_pipeline,
            timestamp_quality=timestamp_quality,
            frame_images_by_index=back_frames_by_index,
            canonical_pts_by_frame=back_canonical_pts_by_frame,
            delta_pts_ms_by_frame=back_delta_pts_ms_by_frame,
        )
        build_ms = (time.perf_counter() - t0) * 1000.0

        return self.associate_tracklets(
            pair_id=pair_id,
            front_motorcycle_tracklets=front_motos,
            back_motorcycle_tracklets=back_motos,
            front_orphan_tracklets=front_orphans,
            back_orphan_tracklets=back_orphans,
            timestamp_quality=timestamp_quality,
            calibrated_homography_available=calibrated_homography_available,
            tracklet_build_ms=build_ms,
        )
