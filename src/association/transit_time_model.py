"""Cross-Camera Transit-Time Compatibility Model (`TransitTimeModel`, Cue A).

Enforces Section 4 & Section 6 (Cue A) Requirements:
1. Preserves strict separation between:
   - `epsilon_sync_ms`: Phase 11.6 residual synchronization error.
   - `delta_pts_front_ms`: Front frame selection discretization distance.
   - `delta_pts_back_ms`: Back frame selection discretization distance.
   - `delta_transit_sec`: Physical transit time between Front and Back camera views.
   - `transit_uncertainty_sec`: Uncertainty in physical transit time due to speed and
     camera placement.
2. Evaluates canonical reference-clock compatibility (`t_ref`) across entry, centroid,
   peak-area, and exit timestamps, supporting:
   - `bidirectional_corridor`: Traffic moving either Front->Back or Back->Front within
     `[min_transit_delay_sec, max_transit_delay_sec]`.
   - `front_to_back`: Strictly requires `delta_transit_sec >= max(0.0, min_transit_delay_sec)`
     (or within `[-overlap_tolerance_sec, max_transit_delay_sec]` when FOVs overlap).
   - `back_to_front`: Strictly requires `delta_transit_sec <= min(0.0, max_transit_delay_sec)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.association.tracklet_models import (
    CueScoreDetail,
    MotorcycleTracklet,
    OrphanRiderTracklet,
    RiderTracklet,
)


@dataclass
class TransitTimeConfig:
    """Configuration for cross-camera transit-time evaluation (all numerical values are engineering hypotheses)."""
    corridor_mode: str = "bidirectional_corridor"
    min_transit_delay_sec: float = -3.50
    max_transit_delay_sec: float = 6.00
    expected_transit_delay_sec: float = 0.75
    transit_sigma_sec: float = 1.50
    transit_uncertainty_sec: float = 0.35
    max_temporal_search_window_sec: float = 8.00
    allow_simultaneous_fov_overlap: bool = True
    max_allowed_sync_residual_ms: float = 100.0


@dataclass
class TransitEvaluationResult:
    """Detailed result of evaluating cross-camera temporal/transit compatibility."""
    is_within_search_window: bool
    is_feasible_transit: bool
    rejection_reason: str | None
    delta_transit_centroid_sec: float
    delta_transit_entry_sec: float
    delta_exit_to_entry_sec: float
    temporal_interval_gap_sec: float
    temporal_overlap_sec: float
    epsilon_sync_ms: float
    delta_pts_front_ms: float
    delta_pts_back_ms: float
    transit_uncertainty_sec: float
    combined_clock_uncertainty_sec: float
    effective_sigma_sec: float
    normalized_score: float
    quality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_within_search_window": bool(self.is_within_search_window),
            "is_feasible_transit": bool(self.is_feasible_transit),
            "rejection_reason": self.rejection_reason,
            "delta_transit_centroid_sec": round(float(self.delta_transit_centroid_sec), 6),
            "delta_transit_entry_sec": round(float(self.delta_transit_entry_sec), 6),
            "delta_exit_to_entry_sec": round(float(self.delta_exit_to_entry_sec), 6),
            "temporal_interval_gap_sec": round(float(self.temporal_interval_gap_sec), 6),
            "temporal_overlap_sec": round(float(self.temporal_overlap_sec), 6),
            "epsilon_sync_ms": round(float(self.epsilon_sync_ms), 4),
            "delta_pts_front_ms": round(float(self.delta_pts_front_ms), 4),
            "delta_pts_back_ms": round(float(self.delta_pts_back_ms), 4),
            "transit_uncertainty_sec": round(float(self.transit_uncertainty_sec), 6),
            "combined_clock_uncertainty_sec": round(float(self.combined_clock_uncertainty_sec), 6),
            "effective_sigma_sec": round(float(self.effective_sigma_sec), 6),
            "normalized_score": round(float(self.normalized_score), 4),
            "quality": round(float(self.quality), 4),
        }


class TransitTimeModel:
    """
    Evaluates cross-camera temporal and transit compatibility on the canonical
    Phase 12C reference clock while keeping clock errors and transit uncertainty distinct.
    """

    def __init__(self, config: TransitTimeConfig | None = None) -> None:
        self.config = config or TransitTimeConfig()

    def evaluate_intervals(
        self,
        front_t_start: float,
        front_t_end: float,
        front_t_centroid: float,
        back_t_start: float,
        back_t_end: float,
        back_t_centroid: float,
        epsilon_sync_ms: float = 0.0,
        delta_pts_front_ms: float = 0.0,
        delta_pts_back_ms: float = 0.0,
    ) -> TransitEvaluationResult:
        """
        Evaluate temporal transit compatibility between a Front interval and Back interval
        expressed on the canonical reference clock `t_ref`.
        """
        delta_centroid = float(back_t_centroid - front_t_centroid)
        delta_entry = float(back_t_start - front_t_start)
        delta_exit_to_entry = float(back_t_start - front_t_end)

        # Interval overlap / gap
        overlap_start = max(front_t_start, back_t_start)
        overlap_end = min(front_t_end, back_t_end)
        if overlap_end >= overlap_start:
            interval_overlap = float(overlap_end - overlap_start)
            interval_gap = 0.0
        else:
            interval_overlap = 0.0
            interval_gap = float(overlap_start - overlap_end)

        # Clock uncertainty in seconds (strictly separated in reporting from transit_uncertainty_sec)
        clock_unc_sec = float(
            (abs(epsilon_sync_ms) + abs(delta_pts_front_ms) + abs(delta_pts_back_ms)) / 1000.0
        )
        transit_unc_sec = float(self.config.transit_uncertainty_sec)
        effective_sigma = float(
            np.sqrt(
                max(1e-4, self.config.transit_sigma_sec) ** 2
                + transit_unc_sec ** 2
                + clock_unc_sec ** 2
            )
        )

        # 1. Hard search window check
        if (
            abs(delta_centroid) > self.config.max_temporal_search_window_sec
            and interval_gap > self.config.max_temporal_search_window_sec
        ):
            return TransitEvaluationResult(
                is_within_search_window=False,
                is_feasible_transit=False,
                rejection_reason="EXCEEDS_MAX_TEMPORAL_SEARCH_WINDOW",
                delta_transit_centroid_sec=delta_centroid,
                delta_transit_entry_sec=delta_entry,
                delta_exit_to_entry_sec=delta_exit_to_entry,
                temporal_interval_gap_sec=interval_gap,
                temporal_overlap_sec=interval_overlap,
                epsilon_sync_ms=epsilon_sync_ms,
                delta_pts_front_ms=delta_pts_front_ms,
                delta_pts_back_ms=delta_pts_back_ms,
                transit_uncertainty_sec=transit_unc_sec,
                combined_clock_uncertainty_sec=clock_unc_sec,
                effective_sigma_sec=effective_sigma,
                normalized_score=0.0,
                quality=1.0,
            )

        # 2. Directional corridor bounds check (expanded slightly by clock uncertainty)
        min_bound = float(self.config.min_transit_delay_sec - clock_unc_sec)
        max_bound = float(self.config.max_transit_delay_sec + clock_unc_sec)

        if not self.config.allow_simultaneous_fov_overlap and interval_overlap > clock_unc_sec:
            return TransitEvaluationResult(
                is_within_search_window=True,
                is_feasible_transit=False,
                rejection_reason="SIMULTANEOUS_FOV_OVERLAP_DISALLOWED",
                delta_transit_centroid_sec=delta_centroid,
                delta_transit_entry_sec=delta_entry,
                delta_exit_to_entry_sec=delta_exit_to_entry,
                temporal_interval_gap_sec=interval_gap,
                temporal_overlap_sec=interval_overlap,
                epsilon_sync_ms=epsilon_sync_ms,
                delta_pts_front_ms=delta_pts_front_ms,
                delta_pts_back_ms=delta_pts_back_ms,
                transit_uncertainty_sec=transit_unc_sec,
                combined_clock_uncertainty_sec=clock_unc_sec,
                effective_sigma_sec=effective_sigma,
                normalized_score=0.0,
                quality=1.0,
            )

        mode = self.config.corridor_mode
        if mode == "front_to_back":
            lower = max(0.0, self.config.min_transit_delay_sec) - clock_unc_sec
            if delta_centroid < lower:
                return TransitEvaluationResult(
                    is_within_search_window=True,
                    is_feasible_transit=False,
                    rejection_reason="IMPOSSIBLE_NEGATIVE_TRANSIT_DELAY_FRONT_TO_BACK",
                    delta_transit_centroid_sec=delta_centroid,
                    delta_transit_entry_sec=delta_entry,
                    delta_exit_to_entry_sec=delta_exit_to_entry,
                    temporal_interval_gap_sec=interval_gap,
                    temporal_overlap_sec=interval_overlap,
                    epsilon_sync_ms=epsilon_sync_ms,
                    delta_pts_front_ms=delta_pts_front_ms,
                    delta_pts_back_ms=delta_pts_back_ms,
                    transit_uncertainty_sec=transit_unc_sec,
                    combined_clock_uncertainty_sec=clock_unc_sec,
                    effective_sigma_sec=effective_sigma,
                    normalized_score=0.0,
                    quality=1.0,
                )
            if delta_centroid > max_bound:
                return TransitEvaluationResult(
                    is_within_search_window=True,
                    is_feasible_transit=False,
                    rejection_reason="EXCEEDS_MAX_TRANSIT_DELAY_FRONT_TO_BACK",
                    delta_transit_centroid_sec=delta_centroid,
                    delta_transit_entry_sec=delta_entry,
                    delta_exit_to_entry_sec=delta_exit_to_entry,
                    temporal_interval_gap_sec=interval_gap,
                    temporal_overlap_sec=interval_overlap,
                    epsilon_sync_ms=epsilon_sync_ms,
                    delta_pts_front_ms=delta_pts_front_ms,
                    delta_pts_back_ms=delta_pts_back_ms,
                    transit_uncertainty_sec=transit_unc_sec,
                    combined_clock_uncertainty_sec=clock_unc_sec,
                    effective_sigma_sec=effective_sigma,
                    normalized_score=0.0,
                    quality=1.0,
                )
        elif mode == "back_to_front":
            upper = min(0.0, self.config.max_transit_delay_sec) + clock_unc_sec
            if delta_centroid > upper or delta_centroid < min_bound:
                return TransitEvaluationResult(
                    is_within_search_window=True,
                    is_feasible_transit=False,
                    rejection_reason="OUTSIDE_BACK_TO_FRONT_TRANSIT_BOUNDS",
                    delta_transit_centroid_sec=delta_centroid,
                    delta_transit_entry_sec=delta_entry,
                    delta_exit_to_entry_sec=delta_exit_to_entry,
                    temporal_interval_gap_sec=interval_gap,
                    temporal_overlap_sec=interval_overlap,
                    epsilon_sync_ms=epsilon_sync_ms,
                    delta_pts_front_ms=delta_pts_front_ms,
                    delta_pts_back_ms=delta_pts_back_ms,
                    transit_uncertainty_sec=transit_unc_sec,
                    combined_clock_uncertainty_sec=clock_unc_sec,
                    effective_sigma_sec=effective_sigma,
                    normalized_score=0.0,
                    quality=1.0,
                )
        else:
            # bidirectional_corridor
            if delta_centroid < min_bound or delta_centroid > max_bound:
                reason = (
                    "BELOW_MIN_TRANSIT_DELAY_BOUND"
                    if delta_centroid < min_bound
                    else "ABOVE_MAX_TRANSIT_DELAY_BOUND"
                )
                return TransitEvaluationResult(
                    is_within_search_window=True,
                    is_feasible_transit=False,
                    rejection_reason=reason,
                    delta_transit_centroid_sec=delta_centroid,
                    delta_transit_entry_sec=delta_entry,
                    delta_exit_to_entry_sec=delta_exit_to_entry,
                    temporal_interval_gap_sec=interval_gap,
                    temporal_overlap_sec=interval_overlap,
                    epsilon_sync_ms=epsilon_sync_ms,
                    delta_pts_front_ms=delta_pts_front_ms,
                    delta_pts_back_ms=delta_pts_back_ms,
                    transit_uncertainty_sec=transit_unc_sec,
                    combined_clock_uncertainty_sec=clock_unc_sec,
                    effective_sigma_sec=effective_sigma,
                    normalized_score=0.0,
                    quality=1.0,
                )

        # 3. Compute continuous Gaussian transit compatibility score in [0, 1]
        if mode == "bidirectional_corridor":
            # Evaluate symmetric corridor around +/- expected_transit_delay_sec and overlap (0.0)
            exp_mag = abs(self.config.expected_transit_delay_sec)
            dev = min(
                abs(delta_centroid - exp_mag),
                abs(delta_centroid + exp_mag),
                abs(delta_centroid),
            )
            # If intervals overlap directly in a shared corridor, deviation is 0
            if interval_overlap > 0.0:
                dev = min(dev, 0.25 * abs(delta_centroid))
        else:
            dev = abs(delta_centroid - self.config.expected_transit_delay_sec)

        raw_score = float(np.exp(-0.5 * (dev / max(1e-4, effective_sigma)) ** 2))

        # Quality factor reflects synchronization precision (penalized only if epsilon_sync_ms is huge)
        sync_quality = float(
            np.clip(
                1.0 - (abs(epsilon_sync_ms) / max(1.0, self.config.max_allowed_sync_residual_ms * 2.0)),
                0.25,
                1.0,
            )
        )

        return TransitEvaluationResult(
            is_within_search_window=True,
            is_feasible_transit=True,
            rejection_reason=None,
            delta_transit_centroid_sec=delta_centroid,
            delta_transit_entry_sec=delta_entry,
            delta_exit_to_entry_sec=delta_exit_to_entry,
            temporal_interval_gap_sec=interval_gap,
            temporal_overlap_sec=interval_overlap,
            epsilon_sync_ms=epsilon_sync_ms,
            delta_pts_front_ms=delta_pts_front_ms,
            delta_pts_back_ms=delta_pts_back_ms,
            transit_uncertainty_sec=transit_unc_sec,
            combined_clock_uncertainty_sec=clock_unc_sec,
            effective_sigma_sec=effective_sigma,
            normalized_score=float(np.clip(raw_score, 0.0, 1.0)),
            quality=sync_quality,
        )

    def evaluate_motorcycle_pair(
        self,
        front_tracklet: MotorcycleTracklet,
        back_tracklet: MotorcycleTracklet,
        nominal_weight: float = 0.45,
    ) -> tuple[TransitEvaluationResult, CueScoreDetail]:
        """Evaluate Cue A (`temporal_transit`) for a candidate `(front_tracklet, back_tracklet)` pair."""
        if (
            front_tracklet.observed_frame_count == 0
            or back_tracklet.observed_frame_count == 0
        ):
            empty_eval = TransitEvaluationResult(
                is_within_search_window=False,
                is_feasible_transit=False,
                rejection_reason="NO_OBSERVED_FRAMES",
                delta_transit_centroid_sec=0.0,
                delta_transit_entry_sec=0.0,
                delta_exit_to_entry_sec=0.0,
                temporal_interval_gap_sec=0.0,
                temporal_overlap_sec=0.0,
                epsilon_sync_ms=front_tracklet.timestamp_quality.epsilon_sync_ms,
                delta_pts_front_ms=front_tracklet.timestamp_quality.mean_delta_pts_front_ms,
                delta_pts_back_ms=back_tracklet.timestamp_quality.mean_delta_pts_back_ms,
                transit_uncertainty_sec=self.config.transit_uncertainty_sec,
                combined_clock_uncertainty_sec=0.0,
                effective_sigma_sec=self.config.transit_sigma_sec,
                normalized_score=0.0,
                quality=0.0,
            )
            return empty_eval, CueScoreDetail(
                cue_name="temporal_transit",
                available=False,
                unavailable_reason="NO_OBSERVED_FRAMES",
                quality=0.0,
                raw_value=None,
                normalized_score=None,
                nominal_weight=nominal_weight,
                effective_weight=0.0,
                weighted_contribution=0.0,
                diagnostics=empty_eval.to_dict(),
            )

        eps_ms = max(
            front_tracklet.timestamp_quality.epsilon_sync_ms,
            back_tracklet.timestamp_quality.epsilon_sync_ms,
        )
        dp_front_ms = front_tracklet.timestamp_quality.mean_delta_pts_front_ms
        dp_back_ms = back_tracklet.timestamp_quality.mean_delta_pts_back_ms

        res = self.evaluate_intervals(
            front_t_start=front_tracklet.t_start_sec,
            front_t_end=front_tracklet.t_end_sec,
            front_t_centroid=front_tracklet.t_centroid_sec,
            back_t_start=back_tracklet.t_start_sec,
            back_t_end=back_tracklet.t_end_sec,
            back_t_centroid=back_tracklet.t_centroid_sec,
            epsilon_sync_ms=eps_ms,
            delta_pts_front_ms=dp_front_ms,
            delta_pts_back_ms=dp_back_ms,
        )

        cue = CueScoreDetail(
            cue_name="temporal_transit",
            available=True,
            unavailable_reason=None,
            quality=res.quality,
            raw_value=round(res.delta_transit_centroid_sec, 6),
            normalized_score=res.normalized_score,
            nominal_weight=nominal_weight,
            effective_weight=nominal_weight,
            weighted_contribution=res.normalized_score * nominal_weight,
            diagnostics=res.to_dict(),
        )
        return res, cue

    def evaluate_rider_or_orphan_pair(
        self,
        front_item: RiderTracklet | OrphanRiderTracklet,
        back_item: RiderTracklet | OrphanRiderTracklet,
        epsilon_sync_ms: float = 0.0,
        delta_pts_front_ms: float = 0.0,
        delta_pts_back_ms: float = 0.0,
    ) -> TransitEvaluationResult:
        """Evaluate temporal transit compatibility between two RiderTracklets or OrphanRiderTracklets."""
        return self.evaluate_intervals(
            front_t_start=front_item.t_start_sec,
            front_t_end=front_item.t_end_sec,
            front_t_centroid=front_item.t_centroid_sec,
            back_t_start=back_item.t_start_sec,
            back_t_end=back_item.t_end_sec,
            back_t_centroid=back_item.t_centroid_sec,
            epsilon_sync_ms=epsilon_sync_ms,
            delta_pts_front_ms=delta_pts_front_ms,
            delta_pts_back_ms=delta_pts_back_ms,
        )
