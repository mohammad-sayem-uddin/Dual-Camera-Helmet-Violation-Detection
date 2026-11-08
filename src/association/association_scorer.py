"""Multi-Cue Association Scorer with Missing-Cue Renormalization (`AssociationScorer`).

Implements Section 6 & Section 7 Requirements:
1. Computes all five cross-camera cues:
   - Cue A: `temporal_transit` (`TransitTimeModel`)
   - Cue B: `rider_cardinality` (`evaluate_rider_cardinality_cue`)
   - Cue C: `appearance_hsv` (`score_appearance_similarity`)
   - Cue D: `lane_spatial_context` (`evaluate_lane_spatial_cue`)
   - Cue E: `convoy_order` (`evaluate_convoy_order_cue`)
2. Enforces Missing-Cue Renormalization:
   Unavailable cues (`available == False`) are excluded from the denominator and
   active cue weights are renormalized to sum to `1.0`. Missing cues never act as
   silent zero penalties.
3. Enforces hard temporal transit gating (`min_temporal_transit_score_gate`) so
   temporally implausible pairs cannot be rescued by appearance or cardinality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.association.appearance_features import (
    AppearanceConfig,
    score_appearance_similarity,
)
from src.association.association_features import (
    ConvoyOrderConfig,
    LaneSpatialConfig,
    RiderCardinalityConfig,
    evaluate_convoy_order_cue,
    evaluate_lane_spatial_cue,
    evaluate_rider_cardinality_cue,
)
from src.association.candidate_generator import CandidatePairEvaluation
from src.association.tracklet_models import (
    CandidatePairScore,
    CueScoreDetail,
    MotorcycleTracklet,
)
from src.association.transit_time_model import TransitTimeModel


@dataclass
class AssociationScorerConfig:
    """Configuration for multi-cue scoring and missing-cue renormalization."""
    weight_temporal_transit: float = 0.45
    weight_rider_cardinality: float = 0.25
    weight_appearance_hsv: float = 0.15
    weight_lane_spatial_context: float = 0.05
    weight_convoy_order: float = 0.10
    renormalize_missing_cues: bool = True
    min_active_cues_required: int = 2
    min_temporal_transit_score_gate: float = 0.20
    rider_cardinality_config: RiderCardinalityConfig = field(
        default_factory=RiderCardinalityConfig
    )
    appearance_config: AppearanceConfig = field(default_factory=AppearanceConfig)
    lane_spatial_config: LaneSpatialConfig = field(default_factory=LaneSpatialConfig)
    convoy_order_config: ConvoyOrderConfig = field(default_factory=ConvoyOrderConfig)


class AssociationScorer:
    """
    Computes multi-cue association scores `CandidatePairScore` with full diagnostic
    provenance and missing-cue weight renormalization.
    """

    def __init__(
        self,
        transit_model: TransitTimeModel,
        config: AssociationScorerConfig | None = None,
    ) -> None:
        self.transit_model = transit_model
        self.config = config or AssociationScorerConfig()

    def score_candidate_pair(
        self,
        candidate: CandidatePairEvaluation,
        all_front_tracklets: list[MotorcycleTracklet],
        all_back_tracklets: list[MotorcycleTracklet],
        calibrated_homography_available: bool = False,
    ) -> CandidatePairScore:
        """
        Compute individual cue scores, renormalize active weights over available cues,
        and return a `CandidatePairScore`.
        """
        f_trk = candidate.front_tracklet
        b_trk = candidate.back_tracklet

        # Cue A: Temporal / Transit Consistency
        t_eval, cue_a = self.transit_model.evaluate_motorcycle_pair(
            f_trk, b_trk, nominal_weight=self.config.weight_temporal_transit
        )

        # Cue B: Rider Occupancy / Cardinality Consistency
        cue_b = evaluate_rider_cardinality_cue(
            f_trk,
            b_trk,
            config=self.config.rider_cardinality_config,
            nominal_weight=self.config.weight_rider_cardinality,
        )

        # Cue C: Viewpoint-Robust Appearance Similarity (Optional)
        app_avail, app_score, app_qual, app_reason, app_diag = score_appearance_similarity(
            f_trk.appearance_descriptor,
            b_trk.appearance_descriptor,
            config=self.config.appearance_config,
        )
        cue_c = CueScoreDetail(
            cue_name="appearance_hsv",
            available=app_avail,
            unavailable_reason=app_reason,
            quality=app_qual,
            raw_value=round(app_score, 4) if app_score is not None else None,
            normalized_score=app_score,
            nominal_weight=self.config.weight_appearance_hsv,
            effective_weight=0.0,
            weighted_contribution=0.0,
            diagnostics=app_diag,
        )

        # Cue D: Lane / Spatial Context (Disabled unless calibrated)
        cue_d = evaluate_lane_spatial_cue(
            f_trk,
            b_trk,
            config=self.config.lane_spatial_config,
            nominal_weight=self.config.weight_lane_spatial_context,
            calibrated_homography_available=calibrated_homography_available,
        )

        # Cue E: Convoy / Relative Temporal Ordering (Optional)
        cue_e = evaluate_convoy_order_cue(
            f_trk,
            b_trk,
            all_front_tracklets=all_front_tracklets,
            all_back_tracklets=all_back_tracklets,
            config=self.config.convoy_order_config,
            nominal_weight=self.config.weight_convoy_order,
        )

        cues_by_name: dict[str, CueScoreDetail] = {
            "temporal_transit": cue_a,
            "rider_cardinality": cue_b,
            "appearance_hsv": cue_c,
            "lane_spatial_context": cue_d,
            "convoy_order": cue_e,
        }

        available_cues: list[str] = []
        unavailable_cues: list[str] = []
        for name, c_det in cues_by_name.items():
            if c_det.available and c_det.normalized_score is not None:
                available_cues.append(name)
            else:
                unavailable_cues.append(name)
                c_det.effective_weight = 0.0
                c_det.weighted_contribution = 0.0

        # Renormalize active cue weights so missing cues do not become zero penalties
        if available_cues:
            if self.config.renormalize_missing_cues:
                total_nom = sum(cues_by_name[name].nominal_weight for name in available_cues)
                for name in available_cues:
                    c_det = cues_by_name[name]
                    eff_w = (
                        c_det.nominal_weight / total_nom
                        if total_nom > 1e-9
                        else 1.0 / len(available_cues)
                    )
                    c_det.effective_weight = float(eff_w)
                    c_det.weighted_contribution = float(
                        (c_det.normalized_score or 0.0) * eff_w
                    )
            else:
                for name in available_cues:
                    c_det = cues_by_name[name]
                    c_det.effective_weight = float(c_det.nominal_weight)
                    c_det.weighted_contribution = float(
                        (c_det.normalized_score or 0.0) * c_det.nominal_weight
                    )

        composite_score = float(
            sum(cues_by_name[name].weighted_contribution for name in available_cues)
        )

        gate_reasons = list(candidate.rejection_reasons)
        passed_gate = bool(candidate.passed_gate)

        # Enforce hard temporal transit score gate
        if (
            cue_a.available
            and cue_a.normalized_score is not None
            and cue_a.normalized_score < self.config.min_temporal_transit_score_gate
        ):
            passed_gate = False
            if "TEMPORAL_SCORE_BELOW_MIN_GATE" not in gate_reasons:
                gate_reasons.append("TEMPORAL_SCORE_BELOW_MIN_GATE")

        if len(available_cues) < self.config.min_active_cues_required:
            # If only 1 cue is available (e.g., no riders coupled and no appearance),
            # record that active cues are below minimum required
            if "INSUFFICIENT_ACTIVE_CUES" not in gate_reasons:
                gate_reasons.append("INSUFFICIENT_ACTIVE_CUES")

        # If candidate failed hard gate, composite score is still computed for auditability,
        # or capped if transit was infeasible
        if not t_eval.is_feasible_transit or not t_eval.is_within_search_window:
            composite_score = 0.0

        return CandidatePairScore(
            front_moto_track_id=f_trk.moto_track_id,
            back_moto_track_id=b_trk.moto_track_id,
            passed_candidate_gate=passed_gate,
            gate_rejection_reasons=gate_reasons,
            composite_score=float(np.clip(composite_score, 0.0, 1.0)),
            active_cues_count=len(available_cues),
            available_cue_names=available_cues,
            unavailable_cue_names=unavailable_cues,
            cue_details=cues_by_name,
            epsilon_sync_ms=t_eval.epsilon_sync_ms,
            delta_pts_front_ms=t_eval.delta_pts_front_ms,
            delta_pts_back_ms=t_eval.delta_pts_back_ms,
            delta_transit_sec=t_eval.delta_transit_centroid_sec,
            transit_uncertainty_sec=t_eval.transit_uncertainty_sec,
        )
