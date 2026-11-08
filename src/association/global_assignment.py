"""Global 1-to-1 Bipartite Assignment and Conservative Abstention (`GlobalAssignmentSolver`).

Implements Section 8 & Section 9 Requirements:
1. Solves gated 1-to-1 bipartite matching across Front and Back `MotorcycleTracklet`s
   using `scipy.optimize.linear_sum_assignment`.
2. Enforces four explicit outcomes:
   - `LINKED`: High-confidence 1-to-1 match passing `min_association_score`,
     `min_active_cues_required`, and both row/column and 2x2 swap uniqueness margins.
   - `AMBIGUOUS_ABSTAIN`: Candidate passes or approaches threshold but has a competing
     row, column, or 2x2 assignment swap within `ambiguous_margin_threshold`.
   - `INSUFFICIENT_EVIDENCE`: Tracklet has fewer than `insufficient_evidence_min_obs_frames`
     observed frames, is `FRAGMENTED_TENTATIVE`, or lacks minimum required active cues.
   - `UNMATCHED`: Tracklet is complete in one camera but has no compatible match in
     the opposite camera.
3. Never allows one Front tracklet to link to multiple Back tracklets or vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.association.rider_association import RiderAssociationEngine
from src.association.tracklet_models import (
    AssociationOutcome,
    CandidatePairScore,
    MotorcycleAssociationResult,
    MotorcycleTracklet,
    SCHEMA_VERSION,
    TrackletCompleteness,
)


@dataclass
class GlobalAssignmentConfig:
    """Configuration for global 1-to-1 assignment and conservative abstention."""
    min_association_score: float = 0.62
    ambiguous_margin_threshold: float = 0.10
    min_temporal_transit_score_gate: float = 0.20
    insufficient_evidence_min_obs_frames: int = 2
    min_active_cues_required: int = 2


class GlobalAssignmentSolver:
    """
    Solves global 1-to-1 cross-camera motorcycle assignment with conservative
    abstention and invokes `RiderAssociationEngine` on `LINKED` pairs.
    """

    def __init__(
        self,
        config: GlobalAssignmentConfig | None = None,
        rider_engine: RiderAssociationEngine | None = None,
    ) -> None:
        self.config = config or GlobalAssignmentConfig()
        self.rider_engine = rider_engine or RiderAssociationEngine()

    def _is_insufficient_tracklet(self, tracklet: MotorcycleTracklet) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if tracklet.observed_frame_count < self.config.insufficient_evidence_min_obs_frames:
            reasons.append(
                f"OBSERVED_FRAMES_{tracklet.observed_frame_count}_BELOW_MIN_{self.config.insufficient_evidence_min_obs_frames}"
            )
        if tracklet.completeness == TrackletCompleteness.FRAGMENTED_TENTATIVE:
            reasons.append("TRACKLET_FRAGMENTED_TENTATIVE")
        return (len(reasons) > 0), reasons

    def solve(
        self,
        front_tracklets: list[MotorcycleTracklet],
        back_tracklets: list[MotorcycleTracklet],
        candidate_scores: list[CandidatePairScore],
    ) -> list[MotorcycleAssociationResult]:
        """
        Solve global 1-to-1 motorcycle assignment and return a complete list of
        `MotorcycleAssociationResult` records covering every Front and Back tracklet.
        """
        results: list[MotorcycleAssociationResult] = []
        score_lookup: dict[tuple[str, str], CandidatePairScore] = {
            (cs.front_moto_track_id, cs.back_moto_track_id): cs for cs in candidate_scores
        }

        # Partition tracklets into eligible vs. insufficient-evidence
        eligible_front: list[MotorcycleTracklet] = []
        eligible_back: list[MotorcycleTracklet] = []
        handled_front_ids: set[str] = set()
        handled_back_ids: set[str] = set()

        for ft in front_tracklets:
            insuff, r_list = self._is_insufficient_tracklet(ft)
            if insuff:
                handled_front_ids.add(ft.moto_track_id)
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=ft.moto_track_id,
                        back_moto_track_id=None,
                        outcome=AssociationOutcome.INSUFFICIENT_EVIDENCE,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        swap_margin_2x2=None,
                        decision_reasons=r_list,
                    )
                )
            else:
                eligible_front.append(ft)

        for bt in back_tracklets:
            insuff, r_list = self._is_insufficient_tracklet(bt)
            if insuff:
                handled_back_ids.add(bt.moto_track_id)
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=None,
                        back_moto_track_id=bt.moto_track_id,
                        outcome=AssociationOutcome.INSUFFICIENT_EVIDENCE,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        swap_margin_2x2=None,
                        decision_reasons=r_list,
                    )
                )
            else:
                eligible_back.append(bt)

        m = len(eligible_front)
        n = len(eligible_back)

        if m == 0 or n == 0:
            for ft in eligible_front:
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=ft.moto_track_id,
                        back_moto_track_id=None,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        swap_margin_2x2=None,
                        decision_reasons=["NO_ELIGIBLE_OPPOSITE_CAMERA_TRACKLETS"],
                    )
                )
            for bt in eligible_back:
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=None,
                        back_moto_track_id=bt.moto_track_id,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        swap_margin_2x2=None,
                        decision_reasons=["NO_ELIGIBLE_OPPOSITE_CAMERA_TRACKLETS"],
                    )
                )
            return results

        # Build score matrix S (m x n) over eligible tracklets
        score_mat = np.zeros((m, n), dtype=np.float64)
        gated_mat = np.zeros((m, n), dtype=bool)
        for i, ft in enumerate(eligible_front):
            for j, bt in enumerate(eligible_back):
                cs = score_lookup.get((ft.moto_track_id, bt.moto_track_id))
                if cs is not None:
                    score_mat[i, j] = float(cs.composite_score)
                    gated_mat[i, j] = bool(cs.passed_candidate_gate)

        # Solve global bipartite assignment maximizing total score
        cost_mat = 1.0 - score_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        assigned_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))

        next_global_idx = 1
        ambiguous_front_indices: set[int] = set()
        ambiguous_back_indices: set[int] = set()

        for i, j in assigned_pairs:
            ft = eligible_front[i]
            bt = eligible_back[j]
            cs = score_lookup.get((ft.moto_track_id, bt.moto_track_id))
            best_score = float(score_mat[i, j])

            # Compute row and column second-best competitor score
            row_competitors = [float(score_mat[i, j2]) for j2 in range(n) if j2 != j]
            col_competitors = [float(score_mat[i2, j]) for i2 in range(m) if i2 != i]
            second_best = max(row_competitors + col_competitors, default=0.0)
            rc_margin = float(best_score - second_best)

            # Compute minimum 2x2 swap margin against any other (i2, j2)
            swap_margin: float | None = None
            for i2, j2 in assigned_pairs:
                if i2 == i and j2 == j:
                    continue
                # Only consider 2x2 swap when at least one cross-term is a viable candidate
                if score_mat[i, j2] > 0.25 or score_mat[i2, j] > 0.25:
                    diff_2x2 = float(
                        (score_mat[i, j] + score_mat[i2, j2])
                        - (score_mat[i, j2] + score_mat[i2, j])
                    )
                    if swap_margin is None or diff_2x2 < swap_margin:
                        swap_margin = diff_2x2

            effective_margin = (
                min(rc_margin, swap_margin) if swap_margin is not None else rc_margin
            )

            # Determine outcome
            if cs is not None and cs.active_cues_count < self.config.min_active_cues_required:
                handled_front_ids.add(ft.moto_track_id)
                handled_back_ids.add(bt.moto_track_id)
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=ft.moto_track_id,
                        back_moto_track_id=bt.moto_track_id,
                        outcome=AssociationOutcome.INSUFFICIENT_EVIDENCE,
                        association_score=best_score,
                        second_best_score=second_best,
                        ambiguity_margin=effective_margin,
                        swap_margin_2x2=swap_margin,
                        decision_reasons=["INSUFFICIENT_ACTIVE_CUES"],
                        pair_score_detail=cs,
                    )
                )
                continue

            # If candidate failed hard gate or falls below min_association_score
            if cs is None or not cs.passed_candidate_gate or best_score < self.config.min_association_score:
                # Leave both ft and bt to be recorded as UNMATCHED (unless involved in ambiguity)
                continue

            # Candidate passes min_association_score: check uniqueness/ambiguity margins
            if effective_margin < self.config.ambiguous_margin_threshold:
                ambiguous_front_indices.add(i)
                ambiguous_back_indices.add(j)
                handled_front_ids.add(ft.moto_track_id)
                handled_back_ids.add(bt.moto_track_id)
                reasons = [
                    f"AMBIGUOUS_UNIQUENESS_MARGIN_{effective_margin:.4f}_BELOW_{self.config.ambiguous_margin_threshold}"
                ]
                if swap_margin is not None and swap_margin < self.config.ambiguous_margin_threshold:
                    reasons.append(f"AMBIGUOUS_2X2_SWAP_MARGIN_{swap_margin:.4f}")
                results.append(
                    MotorcycleAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_motorcycle_id=None,
                        front_moto_track_id=ft.moto_track_id,
                        back_moto_track_id=bt.moto_track_id,
                        outcome=AssociationOutcome.AMBIGUOUS_ABSTAIN,
                        association_score=best_score,
                        second_best_score=second_best,
                        ambiguity_margin=effective_margin,
                        swap_margin_2x2=swap_margin,
                        decision_reasons=reasons,
                        pair_score_detail=cs,
                    )
                )
                continue

            # Confirmed 1-to-1 LINKED pair!
            global_id = f"global_moto_{next_global_idx:04d}"
            next_global_idx += 1
            handled_front_ids.add(ft.moto_track_id)
            handled_back_ids.add(bt.moto_track_id)

            rider_results = self.rider_engine.associate_riders_for_linked_pair(
                global_motorcycle_id=global_id,
                front_moto=ft,
                back_moto=bt,
            )

            results.append(
                MotorcycleAssociationResult(
                    schema_version=SCHEMA_VERSION,
                    global_motorcycle_id=global_id,
                    front_moto_track_id=ft.moto_track_id,
                    back_moto_track_id=bt.moto_track_id,
                    outcome=AssociationOutcome.LINKED,
                    association_score=best_score,
                    second_best_score=second_best,
                    ambiguity_margin=effective_margin,
                    swap_margin_2x2=swap_margin,
                    decision_reasons=["CONFIRMED_1_TO_1_CROSS_CAMERA_MATCH"],
                    pair_score_detail=cs,
                    rider_associations=rider_results,
                )
            )

        # Handle remaining eligible Front tracklets
        for i, ft in enumerate(eligible_front):
            if ft.moto_track_id in handled_front_ids:
                continue
            best_j = int(np.argmax(score_mat[i, :]))
            best_sc = float(score_mat[i, best_j])
            cs = score_lookup.get((ft.moto_track_id, eligible_back[best_j].moto_track_id))

            # If this unassigned Front tracklet competed closely with an ambiguous Back tracklet
            if best_j in ambiguous_back_indices and best_sc >= (
                self.config.min_association_score - self.config.ambiguous_margin_threshold
            ):
                outcome = AssociationOutcome.AMBIGUOUS_ABSTAIN
                reasons = [
                    f"COMPETING_CANDIDATE_WITH_AMBIGUOUS_BACK_{eligible_back[best_j].moto_track_id}"
                ]
            else:
                outcome = AssociationOutcome.UNMATCHED
                reasons = (
                    list(cs.gate_rejection_reasons)
                    if cs is not None and cs.gate_rejection_reasons
                    else [f"BEST_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_association_score}"]
                )

            results.append(
                MotorcycleAssociationResult(
                    schema_version=SCHEMA_VERSION,
                    global_motorcycle_id=None,
                    front_moto_track_id=ft.moto_track_id,
                    back_moto_track_id=None,
                    outcome=outcome,
                    association_score=best_sc,
                    second_best_score=0.0,
                    ambiguity_margin=0.0,
                    swap_margin_2x2=None,
                    decision_reasons=reasons,
                    pair_score_detail=cs,
                )
            )

        # Handle remaining eligible Back tracklets
        for j, bt in enumerate(eligible_back):
            if bt.moto_track_id in handled_back_ids:
                continue
            best_i = int(np.argmax(score_mat[:, j]))
            best_sc = float(score_mat[best_i, j])
            cs = score_lookup.get((eligible_front[best_i].moto_track_id, bt.moto_track_id))

            if best_i in ambiguous_front_indices and best_sc >= (
                self.config.min_association_score - self.config.ambiguous_margin_threshold
            ):
                outcome = AssociationOutcome.AMBIGUOUS_ABSTAIN
                reasons = [
                    f"COMPETING_CANDIDATE_WITH_AMBIGUOUS_FRONT_{eligible_front[best_i].moto_track_id}"
                ]
            else:
                outcome = AssociationOutcome.UNMATCHED
                reasons = (
                    list(cs.gate_rejection_reasons)
                    if cs is not None and cs.gate_rejection_reasons
                    else [f"BEST_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_association_score}"]
                )

            results.append(
                MotorcycleAssociationResult(
                    schema_version=SCHEMA_VERSION,
                    global_motorcycle_id=None,
                    front_moto_track_id=None,
                    back_moto_track_id=bt.moto_track_id,
                    outcome=outcome,
                    association_score=best_sc,
                    second_best_score=0.0,
                    ambiguity_margin=0.0,
                    swap_margin_2x2=None,
                    decision_reasons=reasons,
                    pair_score_detail=cs,
                )
            )

        return results
