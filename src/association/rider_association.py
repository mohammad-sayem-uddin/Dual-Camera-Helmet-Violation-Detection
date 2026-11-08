"""Rider-Level Cross-Camera Association Within Linked Motorcycles (`RiderAssociationEngine`).

Implements Section 10 Requirements:
1. Operates strictly within a `LINKED` `(front_tracklet, back_tracklet)` motorcycle pair.
2. Preserves Front and Back `RiderSubTrack` IDs, `slot_role`s, and Phase 10 verification
   states independently as read-only snapshots (never merges or mutates Phase 10 states).
3. Never uses helmet vs. no-helmet classification labels as a rider identity feature.
4. Enforces 1-to-1 bipartite rider matching with row/column best-vs-second-best and
   2x2 swap ambiguity checks (`rider_ambiguous_margin_threshold`).
5. When rider identity is ambiguous (e.g., two riders with `ordering_confident=False`
   or swap margin below threshold), abstains at the rider level (`AMBIGUOUS_ABSTAIN`)
   while keeping the parent motorcycle association `LINKED`.
6. Supports partial rider visibility across views (e.g., driver + pillion in Front,
   only driver visible in Back), leaving the single-camera rider as `UNMATCHED`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.association.appearance_features import (
    AppearanceConfig,
    score_appearance_similarity,
)
from src.association.tracklet_models import (
    AssociationOutcome,
    MotorcycleTracklet,
    RiderAssociationResult,
    RiderTracklet,
    SCHEMA_VERSION,
)


@dataclass
class RiderAssociationConfig:
    """Configuration for rider-level association within linked motorcycles."""
    min_rider_association_score: float = 0.60
    rider_ambiguous_margin_threshold: float = 0.12
    min_observed_frames_rider: int = 2
    role_match_score: float = 1.00
    role_ambiguous_ordering_score: float = 0.55
    role_mismatch_score: float = 0.10
    weight_slot_role: float = 0.55
    weight_temporal_overlap: float = 0.25
    weight_appearance: float = 0.20
    appearance_config: AppearanceConfig = field(default_factory=AppearanceConfig)


def _readonly_phase10_dict(rt: RiderTracklet) -> dict[str, Any]:
    return {
        "subtrack_id": rt.subtrack_id,
        "camera_id": rt.camera_id,
        "temporal_state": rt.phase10_temporal_state,
        "aggregated_confidence": dict(rt.phase10_aggregated_confidence),
        "consecutive_no_helmet_frames": int(rt.phase10_consecutive_no_helmet_frames),
        "consecutive_helmet_frames": int(rt.phase10_consecutive_helmet_frames),
        "alerts_count": int(rt.phase10_alerts_count),
    }


class RiderAssociationEngine:
    """
    Associates `RiderTracklet`s between a `LINKED` Front and Back `MotorcycleTracklet`.
    """

    def __init__(self, config: RiderAssociationConfig | None = None) -> None:
        self.config = config or RiderAssociationConfig()

    def _score_rider_pair(
        self,
        f_rider: RiderTracklet,
        b_rider: RiderTracklet,
        front_moto: MotorcycleTracklet,
        back_moto: MotorcycleTracklet,
    ) -> tuple[float, dict[str, float], list[str]]:
        reasons: list[str] = []

        # 1. Slot role compatibility
        f_role = f_rider.slot_role
        b_role = b_rider.slot_role
        both_confident = bool(f_rider.ordering_confident and b_rider.ordering_confident)

        if not both_confident or f_role.startswith("ambiguous") or b_role.startswith("ambiguous"):
            s_role = self.config.role_ambiguous_ordering_score
            reasons.append("RIDER_ORDERING_NOT_CONFIDENT")
        elif f_role == b_role:
            s_role = self.config.role_match_score
            reasons.append(f"CONFIDENT_SLOT_ROLE_MATCH_{f_role.upper()}")
        elif (
            {f_role, b_role} == {"pillion_1", "pillion_2"}
        ):
            s_role = 0.45
            reasons.append("PILLION_ORDINAL_SHIFT")
        else:
            s_role = self.config.role_mismatch_score
            reasons.append(f"SLOT_ROLE_MISMATCH_{f_role}_VS_{b_role}")

        # 2. Relative visibility / temporal coverage within parent motorcycle tracklet
        f_rel_cov = float(
            np.clip(
                f_rider.observed_frame_count / float(max(1, front_moto.observed_frame_count)),
                0.0,
                1.0,
            )
        )
        b_rel_cov = float(
            np.clip(
                b_rider.observed_frame_count / float(max(1, back_moto.observed_frame_count)),
                0.0,
                1.0,
            )
        )
        s_temp = float(0.5 * (f_rel_cov + b_rel_cov) * (0.5 + 0.5 * min(f_rider.mean_coupling_score, b_rider.mean_coupling_score)))
        s_temp = float(np.clip(s_temp, 0.0, 1.0))

        # 3. Optional viewpoint-robust appearance similarity (cranial cap + upper torso)
        app_avail, app_score, _, _, _ = score_appearance_similarity(
            f_rider.appearance_descriptor,
            b_rider.appearance_descriptor,
            config=self.config.appearance_config,
        )

        if app_avail and app_score is not None:
            w_role = self.config.weight_slot_role
            w_temp = self.config.weight_temporal_overlap
            w_app = self.config.weight_appearance
            w_sum = max(1e-9, w_role + w_temp + w_app)
            total_score = (w_role * s_role + w_temp * s_temp + w_app * app_score) / w_sum
            breakdown = {
                "slot_role_score": s_role,
                "temporal_coverage_score": s_temp,
                "appearance_score": app_score,
            }
        else:
            # Renormalize over slot_role + temporal_coverage
            w_role = self.config.weight_slot_role
            w_temp = self.config.weight_temporal_overlap
            w_sum = max(1e-9, w_role + w_temp)
            total_score = (w_role * s_role + w_temp * s_temp) / w_sum
            breakdown = {
                "slot_role_score": s_role,
                "temporal_coverage_score": s_temp,
            }

        # Never allow a confident driver<->pillion swap to exceed min_rider_association_score
        if both_confident and f_role != b_role and ("driver" in (f_role, b_role)):
            total_score = min(total_score, self.config.role_mismatch_score + 0.15)

        return float(np.clip(total_score, 0.0, 1.0)), breakdown, reasons

    def associate_riders_for_linked_pair(
        self,
        global_motorcycle_id: str,
        front_moto: MotorcycleTracklet,
        back_moto: MotorcycleTracklet,
    ) -> list[RiderAssociationResult]:
        """
        Perform conservative 1-to-1 rider association between `front_moto.rider_tracklets`
        and `back_moto.rider_tracklets`.
        """
        f_riders = [front_moto.rider_tracklets[k] for k in sorted(front_moto.rider_tracklets.keys())]
        b_riders = [back_moto.rider_tracklets[k] for k in sorted(back_moto.rider_tracklets.keys())]

        results: list[RiderAssociationResult] = []
        if not f_riders and not b_riders:
            return results

        if not f_riders:
            for br in b_riders:
                results.append(
                    RiderAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        parent_global_motorcycle_id=global_motorcycle_id,
                        front_moto_track_id=front_moto.moto_track_id,
                        back_moto_track_id=back_moto.moto_track_id,
                        front_subtrack_id=None,
                        back_subtrack_id=br.subtrack_id,
                        front_slot_role=None,
                        back_slot_role=br.slot_role,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=["SINGLE_CAMERA_ONLY_BACK_RIDER"],
                        cue_breakdown={},
                        front_phase10_readonly=None,
                        back_phase10_readonly=_readonly_phase10_dict(br),
                    )
                )
            return results

        if not b_riders:
            for fr in f_riders:
                results.append(
                    RiderAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        parent_global_motorcycle_id=global_motorcycle_id,
                        front_moto_track_id=front_moto.moto_track_id,
                        back_moto_track_id=back_moto.moto_track_id,
                        front_subtrack_id=fr.subtrack_id,
                        back_subtrack_id=None,
                        front_slot_role=fr.slot_role,
                        back_slot_role=None,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=["SINGLE_CAMERA_ONLY_FRONT_RIDER"],
                        cue_breakdown={},
                        front_phase10_readonly=_readonly_phase10_dict(fr),
                        back_phase10_readonly=None,
                    )
                )
            return results

        m = len(f_riders)
        n = len(b_riders)
        score_mat = np.zeros((m, n), dtype=np.float64)
        breakdowns: dict[tuple[int, int], dict[str, float]] = {}
        pair_reasons: dict[tuple[int, int], list[str]] = {}

        for i, fr in enumerate(f_riders):
            for j, br in enumerate(b_riders):
                sc, brk, r_list = self._score_rider_pair(fr, br, front_moto, back_moto)
                score_mat[i, j] = sc
                breakdowns[(i, j)] = brk
                pair_reasons[(i, j)] = r_list

        cost_mat = 1.0 - score_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        assigned_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))

        matched_f: set[int] = set()
        matched_b: set[int] = set()

        for i, j in assigned_pairs:
            fr = f_riders[i]
            br = b_riders[j]
            best_sc = float(score_mat[i, j])

            other_row_scores = [float(score_mat[i, j2]) for j2 in range(n) if j2 != j]
            other_col_scores = [float(score_mat[i2, j]) for i2 in range(m) if i2 != i]
            second_best = max(other_row_scores + other_col_scores, default=0.0)
            margin = float(best_sc - second_best)

            # Check 2x2 swap margin against any other assigned pair (i2, j2)
            min_swap_margin = margin
            for i2, j2 in assigned_pairs:
                if i2 == i and j2 == j:
                    continue
                assigned_sum = float(score_mat[i, j] + score_mat[i2, j2])
                swapped_sum = float(score_mat[i, j2] + score_mat[i2, j])
                swap_diff = assigned_sum - swapped_sum
                if swap_diff < min_swap_margin:
                    min_swap_margin = swap_diff

            reasons = list(pair_reasons[(i, j)])

            # 1. Check minimum observed frames (INSUFFICIENT_EVIDENCE)
            if (
                fr.observed_frame_count < self.config.min_observed_frames_rider
                or br.observed_frame_count < self.config.min_observed_frames_rider
            ):
                reasons.append("RIDER_TRACKLET_INSUFFICIENT_OBSERVED_FRAMES")
                outcome = AssociationOutcome.INSUFFICIENT_EVIDENCE
            # 2. Check multi-rider ambiguity margin or unconfident ordering when m >= 2 or n >= 2
            elif (m >= 2 or n >= 2) and (
                min_swap_margin < self.config.rider_ambiguous_margin_threshold
                or (not fr.ordering_confident and m >= 2)
                or (not br.ordering_confident and n >= 2)
            ):
                reasons.append(
                    f"RIDER_AMBIGUOUS_MARGIN_{min_swap_margin:.3f}_BELOW_{self.config.rider_ambiguous_margin_threshold}"
                )
                outcome = AssociationOutcome.AMBIGUOUS_ABSTAIN
            # 3. Check minimum rider association score
            elif best_sc < self.config.min_rider_association_score:
                reasons.append(
                    f"RIDER_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_rider_association_score}"
                )
                outcome = AssociationOutcome.UNMATCHED
            else:
                reasons.append("RIDER_LINK_CONFIRMED")
                outcome = AssociationOutcome.LINKED

            matched_f.add(i)
            matched_b.add(j)
            results.append(
                RiderAssociationResult(
                    schema_version=SCHEMA_VERSION,
                    parent_global_motorcycle_id=global_motorcycle_id,
                    front_moto_track_id=front_moto.moto_track_id,
                    back_moto_track_id=back_moto.moto_track_id,
                    front_subtrack_id=fr.subtrack_id,
                    back_subtrack_id=br.subtrack_id,
                    front_slot_role=fr.slot_role,
                    back_slot_role=br.slot_role,
                    outcome=outcome,
                    association_score=best_sc,
                    second_best_score=second_best,
                    ambiguity_margin=min_swap_margin,
                    decision_reasons=reasons,
                    cue_breakdown=breakdowns[(i, j)],
                    front_phase10_readonly=_readonly_phase10_dict(fr),
                    back_phase10_readonly=_readonly_phase10_dict(br),
                )
            )

        # Record any unmatched single-view riders (e.g., pillion visible in only one camera)
        for i, fr in enumerate(f_riders):
            if i not in matched_f:
                best_comp = float(np.max(score_mat[i, :])) if n > 0 else 0.0
                results.append(
                    RiderAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        parent_global_motorcycle_id=global_motorcycle_id,
                        front_moto_track_id=front_moto.moto_track_id,
                        back_moto_track_id=back_moto.moto_track_id,
                        front_subtrack_id=fr.subtrack_id,
                        back_subtrack_id=None,
                        front_slot_role=fr.slot_role,
                        back_slot_role=None,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=best_comp,
                        ambiguity_margin=0.0,
                        decision_reasons=["SINGLE_CAMERA_ONLY_RIDER_UNMATCHED"],
                        cue_breakdown={},
                        front_phase10_readonly=_readonly_phase10_dict(fr),
                        back_phase10_readonly=None,
                    )
                )

        for j, br in enumerate(b_riders):
            if j not in matched_b:
                best_comp = float(np.max(score_mat[:, j])) if m > 0 else 0.0
                results.append(
                    RiderAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        parent_global_motorcycle_id=global_motorcycle_id,
                        front_moto_track_id=front_moto.moto_track_id,
                        back_moto_track_id=back_moto.moto_track_id,
                        front_subtrack_id=None,
                        back_subtrack_id=br.subtrack_id,
                        front_slot_role=None,
                        back_slot_role=br.slot_role,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=best_comp,
                        ambiguity_margin=0.0,
                        decision_reasons=["SINGLE_CAMERA_ONLY_RIDER_UNMATCHED"],
                        cue_breakdown={},
                        front_phase10_readonly=None,
                        back_phase10_readonly=_readonly_phase10_dict(br),
                    )
                )

        return results
