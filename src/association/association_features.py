"""Multi-Cue Structural, Spatial, and Convoy Feature Evaluators (Cue B, Cue D, Cue E).

Implements:
1. Cue B (`rider_cardinality`): Compares observed rider occupancy distributions and
   slot role structures across Front and Back `MotorcycleTracklet`s, accounting for
   partial pillion occlusion in one view without hard-failing, while penalizing
   large occupancy mismatches (e.g., 1 rider vs. 3 riders).
2. Cue D (`lane_spatial_context`): Explicitly disabled by default
   (`unavailable_reason = "UNCALIBRATED_CAMERA_EXTRINSICS"`) to prevent invalid
   comparisons of raw image x-coordinates across uncalibrated opposing cameras.
3. Cue E (`convoy_order`): Evaluates relative temporal arrival ordering when >= 2
   motorcycles co-occur within the convoy window in both views, without circular
   reasoning. Gracefully marked unavailable when no multi-motorcycle convoy exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.association.tracklet_models import CueScoreDetail, MotorcycleTracklet


@dataclass
class RiderCardinalityConfig:
    """Configuration for Cue B (`rider_cardinality`)."""
    exact_count_match_score: float = 1.00
    partial_pillion_occlusion_diff1_score: float = 0.65
    unexplained_diff1_score: float = 0.35
    diff2_or_more_score: float = 0.08
    partial_occlusion_visibility_threshold: float = 0.75


@dataclass
class LaneSpatialConfig:
    """Configuration for Cue D (`lane_spatial_context`)."""
    enabled: bool = False
    require_explicit_calibration: bool = True


@dataclass
class ConvoyOrderConfig:
    """Configuration for Cue E (`convoy_order`)."""
    enabled: bool = True
    convoy_window_sec: float = 5.00
    consistent_order_score: float = 0.90
    overtake_or_inversion_score: float = 0.35
    min_anchor_temporal_gap_sec: float = 0.25


def evaluate_rider_cardinality_cue(
    front_tracklet: MotorcycleTracklet,
    back_tracklet: MotorcycleTracklet,
    config: RiderCardinalityConfig,
    nominal_weight: float = 0.25,
) -> CueScoreDetail:
    """
    Evaluate Cue B (`rider_cardinality`) between `front_tracklet` and `back_tracklet`.
    """
    f_count = max(front_tracklet.modal_rider_count, front_tracklet.rider_subtrack_count)
    b_count = max(back_tracklet.modal_rider_count, back_tracklet.rider_subtrack_count)

    if f_count == 0 and b_count == 0:
        return CueScoreDetail(
            cue_name="rider_cardinality",
            available=False,
            unavailable_reason="NO_COUPLED_RIDERS_IN_EITHER_VIEW",
            quality=0.0,
            raw_value={"front_riders": 0, "back_riders": 0},
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={"front_modal_riders": 0, "back_modal_riders": 0},
        )

    diff = abs(f_count - b_count)

    # Compute rider visibility ratio in each camera to detect plausible partial pillion occlusion
    def _visibility_ratio(mt: MotorcycleTracklet) -> float:
        if mt.observed_frame_count <= 0 or not mt.rider_tracklets:
            return 0.0
        ratios = [
            rt.observed_frame_count / float(max(1, mt.observed_frame_count))
            for rt in mt.rider_tracklets.values()
        ]
        return float(min(ratios)) if ratios else 0.0

    f_vis = _visibility_ratio(front_tracklet)
    b_vis = _visibility_ratio(back_tracklet)
    has_occlusion_evidence = bool(
        f_vis < config.partial_occlusion_visibility_threshold
        or b_vis < config.partial_occlusion_visibility_threshold
        or len(front_tracklet.occlusion_intervals) > 0
        or len(back_tracklet.occlusion_intervals) > 0
        or front_tracklet.observed_frame_count <= 4
        or back_tracklet.observed_frame_count <= 4
    )

    if diff == 0:
        score = config.exact_count_match_score
        reason = "EXACT_RIDER_COUNT_MATCH"
    elif diff == 1:
        if has_occlusion_evidence:
            score = config.partial_pillion_occlusion_diff1_score
            reason = "DIFF_1_WITH_PARTIAL_OCCLUSION_TOLERANCE"
        else:
            score = config.unexplained_diff1_score
            reason = "DIFF_1_WITHOUT_OCCLUSION_EVIDENCE"
    else:
        score = config.diff2_or_more_score
        reason = f"LARGE_RIDER_COUNT_MISMATCH_DIFF_{diff}"

    quality = float(
        np.clip(
            0.5 * (front_tracklet.observation_density + back_tracklet.observation_density),
            0.25,
            1.0,
        )
    )

    return CueScoreDetail(
        cue_name="rider_cardinality",
        available=True,
        unavailable_reason=None,
        quality=quality,
        raw_value={"front_riders": f_count, "back_riders": b_count, "diff": diff},
        normalized_score=float(np.clip(score, 0.0, 1.0)),
        nominal_weight=nominal_weight,
        effective_weight=nominal_weight,
        weighted_contribution=float(score * nominal_weight),
        diagnostics={
            "front_modal_riders": front_tracklet.modal_rider_count,
            "back_modal_riders": back_tracklet.modal_rider_count,
            "front_subtrack_count": front_tracklet.rider_subtrack_count,
            "back_subtrack_count": back_tracklet.rider_subtrack_count,
            "front_min_rider_visibility_ratio": round(f_vis, 4),
            "back_min_rider_visibility_ratio": round(b_vis, 4),
            "has_occlusion_evidence": has_occlusion_evidence,
            "match_regime": reason,
        },
    )


def evaluate_lane_spatial_cue(
    front_tracklet: MotorcycleTracklet,
    back_tracklet: MotorcycleTracklet,
    config: LaneSpatialConfig,
    nominal_weight: float = 0.05,
    calibrated_homography_available: bool = False,
) -> CueScoreDetail:
    """
    Evaluate Cue D (`lane_spatial_context`).
    Strictly disabled when camera extrinsics/homography are uncalibrated.
    """
    if not config.enabled or (config.require_explicit_calibration and not calibrated_homography_available):
        return CueScoreDetail(
            cue_name="lane_spatial_context",
            available=False,
            unavailable_reason="UNCALIBRATED_CAMERA_EXTRINSICS",
            quality=0.0,
            raw_value=None,
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={
                "front_id": front_tracklet.moto_track_id,
                "back_id": back_tracklet.moto_track_id,
                "calibrated_homography_available": bool(calibrated_homography_available),
            },
        )

    # Optional calibrated road-plane corridor comparison if explicit calibration is provided
    f_zone = front_tracklet.source_metadata.get("calibrated_corridor_zone")
    b_zone = back_tracklet.source_metadata.get("calibrated_corridor_zone")
    if f_zone is None or b_zone is None:
        return CueScoreDetail(
            cue_name="lane_spatial_context",
            available=False,
            unavailable_reason="MISSING_CALIBRATED_CORRIDOR_ZONE",
            quality=0.0,
            raw_value=None,
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={},
        )

    score = 1.0 if f_zone == b_zone else 0.25
    return CueScoreDetail(
        cue_name="lane_spatial_context",
        available=True,
        unavailable_reason=None,
        quality=1.0,
        raw_value={"front_zone": f_zone, "back_zone": b_zone},
        normalized_score=score,
        nominal_weight=nominal_weight,
        effective_weight=nominal_weight,
        weighted_contribution=score * nominal_weight,
        diagnostics={"front_zone": f_zone, "back_zone": b_zone},
    )


def evaluate_convoy_order_cue(
    front_tracklet: MotorcycleTracklet,
    back_tracklet: MotorcycleTracklet,
    all_front_tracklets: list[MotorcycleTracklet],
    all_back_tracklets: list[MotorcycleTracklet],
    config: ConvoyOrderConfig,
    nominal_weight: float = 0.10,
) -> CueScoreDetail:
    """
    Evaluate Cue E (`convoy_order`) over co-occurring motorcycles within `convoy_window_sec`.
    Never uses circular dependencies on uncertain associations: compares ordinal rank
    position of `front_tracklet` within its local Front convoy against the ordinal rank
    position of `back_tracklet` within its local Back convoy when both convoys have
    well-separated timestamps (`>= min_anchor_temporal_gap_sec`).
    """
    if not config.enabled:
        return CueScoreDetail(
            cue_name="convoy_order",
            available=False,
            unavailable_reason="CUE_DISABLED_IN_CONFIG",
            quality=0.0,
            raw_value=None,
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={},
        )

    f_neighbors = [
        t
        for t in all_front_tracklets
        if t.observed_frame_count >= 2
        and abs(t.t_centroid_sec - front_tracklet.t_centroid_sec) <= config.convoy_window_sec
    ]
    b_neighbors = [
        t
        for t in all_back_tracklets
        if t.observed_frame_count >= 2
        and abs(t.t_centroid_sec - back_tracklet.t_centroid_sec) <= config.convoy_window_sec
    ]

    if len(f_neighbors) < 2 or len(b_neighbors) < 2:
        return CueScoreDetail(
            cue_name="convoy_order",
            available=False,
            unavailable_reason="SINGLE_MOTORCYCLE_NO_CONVOY",
            quality=0.0,
            raw_value={"front_convoy_size": len(f_neighbors), "back_convoy_size": len(b_neighbors)},
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={
                "front_convoy_size": len(f_neighbors),
                "back_convoy_size": len(b_neighbors),
            },
        )

    f_sorted = sorted(f_neighbors, key=lambda t: (t.t_centroid_sec, t.moto_track_id))
    b_sorted = sorted(b_neighbors, key=lambda t: (t.t_centroid_sec, t.moto_track_id))

    # Verify minimum temporal separation between convoy members to avoid noisy rank ties
    f_gaps = [
        f_sorted[i + 1].t_centroid_sec - f_sorted[i].t_centroid_sec
        for i in range(len(f_sorted) - 1)
    ]
    b_gaps = [
        b_sorted[i + 1].t_centroid_sec - b_sorted[i].t_centroid_sec
        for i in range(len(b_sorted) - 1)
    ]
    if min(f_gaps) < config.min_anchor_temporal_gap_sec or min(b_gaps) < config.min_anchor_temporal_gap_sec:
        return CueScoreDetail(
            cue_name="convoy_order",
            available=False,
            unavailable_reason="SIMULTANEOUS_CONVOY_ARRIVAL_NO_ORDER_GAP",
            quality=0.0,
            raw_value=None,
            normalized_score=None,
            nominal_weight=nominal_weight,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics={
                "min_front_gap_sec": round(float(min(f_gaps)), 4),
                "min_back_gap_sec": round(float(min(b_gaps)), 4),
            },
        )

    f_rank = next(
        (idx for idx, t in enumerate(f_sorted) if t.moto_track_id == front_tracklet.moto_track_id),
        0,
    )
    b_rank = next(
        (idx for idx, t in enumerate(b_sorted) if t.moto_track_id == back_tracklet.moto_track_id),
        0,
    )

    f_rel_pos = float(f_rank) / float(max(1, len(f_sorted) - 1))
    b_rel_pos = float(b_rank) / float(max(1, len(b_sorted) - 1))
    rank_diff = abs(f_rel_pos - b_rel_pos)

    if rank_diff <= 1e-4:
        score = config.consistent_order_score
        order_status = "ORDER_PRESERVED"
    else:
        score = float(
            config.consistent_order_score * (1.0 - rank_diff)
            + config.overtake_or_inversion_score * rank_diff
        )
        order_status = "ORDER_INVERTED_POSSIBLE_OVERTAKE"

    return CueScoreDetail(
        cue_name="convoy_order",
        available=True,
        unavailable_reason=None,
        quality=0.85,
        raw_value={
            "front_rank": f_rank,
            "front_convoy_size": len(f_sorted),
            "back_rank": b_rank,
            "back_convoy_size": len(b_sorted),
        },
        normalized_score=float(np.clip(score, 0.0, 1.0)),
        nominal_weight=nominal_weight,
        effective_weight=nominal_weight,
        weighted_contribution=float(score * nominal_weight),
        diagnostics={
            "front_rank": f_rank,
            "back_rank": b_rank,
            "relative_rank_diff": round(rank_diff, 4),
            "order_status": order_status,
        },
    )
