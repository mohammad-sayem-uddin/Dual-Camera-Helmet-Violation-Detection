"""Cross-Camera Candidate Pair Generator (`CandidateGenerator`).

Implements Section 5 Requirements:
1. Evaluates all `(FrontTracklet, BackTracklet)` pairs against:
   - Common overlap validity (`is_in_overlap`).
   - Minimum observed frame count (`min_observed_frames_moto`).
   - Transit-time feasibility (`TransitTimeModel`).
2. Records explicit candidate gate pass/fail status and rejection reasons so no
   candidate pair is silently dropped without an audit record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.association.tracklet_models import (
    MotorcycleTracklet,
    TrackletCompleteness,
)
from src.association.transit_time_model import TransitEvaluationResult, TransitTimeModel


@dataclass
class CandidatePairEvaluation:
    """Result of candidate-generation gating for `(front_tracklet, back_tracklet)`."""
    front_tracklet: MotorcycleTracklet
    back_tracklet: MotorcycleTracklet
    transit_eval: TransitEvaluationResult
    passed_gate: bool
    rejection_reasons: list[str]
    insufficient_evidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_moto_track_id": self.front_tracklet.moto_track_id,
            "back_moto_track_id": self.back_tracklet.moto_track_id,
            "passed_gate": bool(self.passed_gate),
            "insufficient_evidence": bool(self.insufficient_evidence),
            "rejection_reasons": list(self.rejection_reasons),
            "transit_eval": self.transit_eval.to_dict(),
        }


class CandidateGenerator:
    """
    Generates and gates candidate `(front_tracklet, back_tracklet)` pairs with
    explicit rejection provenance.
    """

    def __init__(
        self,
        transit_model: TransitTimeModel,
        min_observed_frames: int = 2,
        require_in_overlap: bool = True,
    ) -> None:
        self.transit_model = transit_model
        self.min_observed_frames = int(min_observed_frames)
        self.require_in_overlap = bool(require_in_overlap)

    def generate_candidates(
        self,
        front_tracklets: list[MotorcycleTracklet],
        back_tracklets: list[MotorcycleTracklet],
    ) -> list[CandidatePairEvaluation]:
        """
        Evaluate every `(front_tracklet, back_tracklet)` pair and return structured
        `CandidatePairEvaluation` records.
        """
        candidates: list[CandidatePairEvaluation] = []
        for f_trk in front_tracklets:
            for b_trk in back_tracklets:
                reasons: list[str] = []
                insufficient = False

                if self.require_in_overlap and (
                    not f_trk.timestamp_quality.is_in_overlap
                    or not b_trk.timestamp_quality.is_in_overlap
                ):
                    reasons.append("OUTSIDE_SYNCHRONIZED_OVERLAP_WINDOW")

                if f_trk.observed_frame_count < self.min_observed_frames:
                    reasons.append(
                        f"FRONT_OBSERVED_FRAMES_{f_trk.observed_frame_count}_BELOW_MIN_{self.min_observed_frames}"
                    )
                    insufficient = True

                if b_trk.observed_frame_count < self.min_observed_frames:
                    reasons.append(
                        f"BACK_OBSERVED_FRAMES_{b_trk.observed_frame_count}_BELOW_MIN_{self.min_observed_frames}"
                    )
                    insufficient = True

                if f_trk.completeness == TrackletCompleteness.FRAGMENTED_TENTATIVE:
                    reasons.append("FRONT_TRACKLET_FRAGMENTED_TENTATIVE")
                    insufficient = True

                if b_trk.completeness == TrackletCompleteness.FRAGMENTED_TENTATIVE:
                    reasons.append("BACK_TRACKLET_FRAGMENTED_TENTATIVE")
                    insufficient = True

                t_eval, _ = self.transit_model.evaluate_motorcycle_pair(f_trk, b_trk)
                if not t_eval.is_within_search_window or not t_eval.is_feasible_transit:
                    if t_eval.rejection_reason:
                        reasons.append(t_eval.rejection_reason)
                    else:
                        reasons.append("TRANSIT_TIME_INFEASIBLE")

                passed = len(reasons) == 0
                candidates.append(
                    CandidatePairEvaluation(
                        front_tracklet=f_trk,
                        back_tracklet=b_trk,
                        transit_eval=t_eval,
                        passed_gate=passed,
                        rejection_reasons=reasons,
                        insufficient_evidence=insufficient,
                    )
                )
        return candidates
