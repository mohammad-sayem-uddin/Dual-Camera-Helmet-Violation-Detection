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

        # Build valid candidate mask and score matrices over eligible tracklets.
        # Only candidates satisfying all candidate gates and minimum active cues participate.
        valid_mask = np.zeros((m, n), dtype=bool)
        valid_score_mat = np.zeros((m, n), dtype=np.float64)

        for i, ft in enumerate(eligible_front):
            for j, bt in enumerate(eligible_back):
                cs = score_lookup.get((ft.moto_track_id, bt.moto_track_id))
                if (
                    cs is not None
                    and cs.passed_candidate_gate
                    and cs.active_cues_count >= self.config.min_active_cues_required
                ):
                    valid_mask[i, j] = True
                    valid_score_mat[i, j] = float(cs.composite_score)

        # Solve global bipartite assignment maximizing valid candidate score.
        # We construct an augmented (m + n) x (m + n) cost matrix allowing Real-to-Dummy matching
        # so tracklets can remain unmatched at zero penalty rather than forcing invalid candidate pairs.
        matched_pairs: list[tuple[int, int]] = []
        if np.any(valid_mask):
            N = m + n
            FORBIDDEN_COST = 1e6
            augmented_cost = np.full((N, N), FORBIDDEN_COST, dtype=np.float64)

            # Block 1 (Top-Left m x n): Real Front to Real Back for valid candidates only
            for i in range(m):
                for j in range(n):
                    if valid_mask[i, j]:
                        augmented_cost[i, j] = 1.0 - valid_score_mat[i, j]

            # Block 2 (Top-Right m x m): Real Front to Dummy Back (diagonal = 1.0)
            for i in range(m):
                augmented_cost[i, n + i] = 1.0

            # Block 3 (Bottom-Left n x n): Dummy Front to Real Back (diagonal = 1.0)
            for j in range(n):
                augmented_cost[m + j, j] = 1.0

            # Block 4 (Bottom-Right n x m): Dummy Front to Dummy Back (all 0.0)
            augmented_cost[m:, n:] = 0.0

            row_ind, col_ind = linear_sum_assignment(augmented_cost)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if r < m and c < n and valid_mask[r, c]:
                    matched_pairs.append((r, c))

        next_global_idx = 1
        ambiguous_front_indices: set[int] = set()
        ambiguous_back_indices: set[int] = set()

        for i, j in matched_pairs:
            ft = eligible_front[i]
            bt = eligible_back[j]
            cs = score_lookup.get((ft.moto_track_id, bt.moto_track_id))
            best_score = float(valid_score_mat[i, j])

            # Compute row and column second-best competitor score among VALID candidates only
            row_competitors = [
                float(valid_score_mat[i, j2])
                for j2 in range(n)
                if j2 != j and valid_mask[i, j2]
            ]
            col_competitors = [
                float(valid_score_mat[i2, j])
                for i2 in range(m)
                if i2 != i and valid_mask[i2, j]
            ]
            second_best = max(row_competitors + col_competitors, default=0.0)
            rc_margin = round(float(best_score - second_best), 6)

            # Compute minimum 2x2 swap margin against any other (i2, j2)
            # A 2x2 swap is only evaluated if both cross-terms are valid candidates
            swap_margin: float | None = None
            for i2, j2 in matched_pairs:
                if i2 == i and j2 == j:
                    continue
                if valid_mask[i, j2] and valid_mask[i2, j]:
                    diff_2x2 = round(
                        float(
                            (valid_score_mat[i, j] + valid_score_mat[i2, j2])
                            - (valid_score_mat[i, j2] + valid_score_mat[i2, j])
                        ),
                        6,
                    )
                    if swap_margin is None or diff_2x2 < swap_margin:
                        swap_margin = diff_2x2

            effective_margin = (
                min(rc_margin, swap_margin) if swap_margin is not None else rc_margin
            )

            # If score falls below min_association_score, leave unassigned
            if best_score < self.config.min_association_score:
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

            valid_j_indices = [j for j in range(n) if valid_mask[i, j]]
            if valid_j_indices:
                best_j = max(valid_j_indices, key=lambda j: valid_score_mat[i, j])
                best_sc = float(valid_score_mat[i, best_j])
                cs = score_lookup.get((ft.moto_track_id, eligible_back[best_j].moto_track_id))

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
            else:
                # No valid candidate for this Front tracklet
                best_sc = 0.0
                cs = None
                reasons = []
                for j in range(n):
                    cand_cs = score_lookup.get((ft.moto_track_id, eligible_back[j].moto_track_id))
                    if cand_cs is not None:
                        cs = cand_cs
                        if cand_cs.gate_rejection_reasons:
                            reasons = list(cand_cs.gate_rejection_reasons)
                        break
                if not reasons:
                    reasons = ["NO_VALID_CROSS_CAMERA_CANDIDATES"]

                if (
                    cs is not None
                    and "INSUFFICIENT_ACTIVE_CUES" in cs.gate_rejection_reasons
                    and not any(r for r in cs.gate_rejection_reasons if "TRANSIT" in r or "OVERLAP" in r)
                ):
                    outcome = AssociationOutcome.INSUFFICIENT_EVIDENCE
                else:
                    outcome = AssociationOutcome.UNMATCHED

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

            valid_i_indices = [i for i in range(m) if valid_mask[i, j]]
            if valid_i_indices:
                best_i = max(valid_i_indices, key=lambda i: valid_score_mat[i, j])
                best_sc = float(valid_score_mat[best_i, j])
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
            else:
                best_sc = 0.0
                cs = None
                reasons = []
                for i in range(m):
                    cand_cs = score_lookup.get((eligible_front[i].moto_track_id, bt.moto_track_id))
                    if cand_cs is not None:
                        cs = cand_cs
                        if cand_cs.gate_rejection_reasons:
                            reasons = list(cand_cs.gate_rejection_reasons)
                        break
                if not reasons:
                    reasons = ["NO_VALID_CROSS_CAMERA_CANDIDATES"]

                if (
                    cs is not None
                    and "INSUFFICIENT_ACTIVE_CUES" in cs.gate_rejection_reasons
                    and not any(r for r in cs.gate_rejection_reasons if "TRANSIT" in r or "OVERLAP" in r)
                ):
                    outcome = AssociationOutcome.INSUFFICIENT_EVIDENCE
                else:
                    outcome = AssociationOutcome.UNMATCHED

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
