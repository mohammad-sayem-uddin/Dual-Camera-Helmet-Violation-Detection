"""First-Class Cross-Camera Orphan Rider Tracklet Association (`OrphanAssociationEngine`).

Implements Section 11 Requirements:
1. Treats `OrphanRiderTracklet` as a first-class entity so riders whose parent
   motorcycle was missed in one or both cameras are never silently dropped.
2. Evaluates cross-camera candidate links between:
   - Front `OrphanRiderTracklet` <-> Back `OrphanRiderTracklet`
   - Front `OrphanRiderTracklet` <-> Back `RiderTracklet` (from an `UNMATCHED` Back motorcycle)
   - Front `RiderTracklet` (from an `UNMATCHED` Front motorcycle) <-> Back `OrphanRiderTracklet`
3. Never links orphan riders on weak temporal proximity alone: requires either
   valid appearance similarity + feasible transit time, or strong temporal transit
   (`>= min_strong_temporal_score_without_appearance`) + observation density quality.
4. Labels any confirmed match `association_level = "ORPHAN_LEVEL_LINK"` with
   `implies_confirmed_motorcycle_link = False`.
5. Abstains (`AMBIGUOUS_ABSTAIN`) when competing orphan candidates fall within
   `orphan_ambiguous_margin_threshold`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.association.appearance_features import (
    AppearanceConfig,
    score_appearance_similarity,
)
from src.association.tracklet_models import (
    AssociationOutcome,
    MotorcycleAssociationResult,
    MotorcycleTracklet,
    OrphanAssociationResult,
    OrphanRiderTracklet,
    RiderTracklet,
    SCHEMA_VERSION,
)
from src.association.transit_time_model import TransitTimeModel


@dataclass
class OrphanAssociationConfig:
    """Configuration for conservative orphan rider tracklet association."""
    enable_orphan_matching: bool = True
    min_orphan_association_score: float = 0.68
    orphan_ambiguous_margin_threshold: float = 0.14
    min_observed_frames_orphan: int = 2
    require_appearance_or_strong_temporal: bool = True
    min_strong_temporal_score_without_appearance: float = 0.82
    appearance_config: AppearanceConfig = field(default_factory=AppearanceConfig)


@dataclass
class _OrphanCandidateItem:
    camera_id: str
    track_id: str
    track_kind: str  # "orphan_rider" or "unmatched_moto_rider"
    tracklet: OrphanRiderTracklet | RiderTracklet
    is_primary_orphan: bool


class OrphanAssociationEngine:
    """
    Associates `OrphanRiderTracklet`s across cameras conservatively without
    implying a confirmed motorcycle-level link.
    """

    def __init__(
        self,
        transit_model: TransitTimeModel,
        config: OrphanAssociationConfig | None = None,
    ) -> None:
        self.transit_model = transit_model
        self.config = config or OrphanAssociationConfig()

    def _score_pair(
        self,
        f_item: _OrphanCandidateItem,
        b_item: _OrphanCandidateItem,
        epsilon_sync_ms: float,
        delta_pts_front_ms: float,
        delta_pts_back_ms: float,
    ) -> tuple[bool, float, dict[str, float], list[str]]:
        f_trk = f_item.tracklet
        b_trk = b_item.tracklet

        t_eval = self.transit_model.evaluate_rider_or_orphan_pair(
            f_trk,
            b_trk,
            epsilon_sync_ms=epsilon_sync_ms,
            delta_pts_front_ms=delta_pts_front_ms,
            delta_pts_back_ms=delta_pts_back_ms,
        )
        if not t_eval.is_within_search_window or not t_eval.is_feasible_transit:
            return (
                False,
                0.0,
                {"temporal_transit_score": 0.0},
                [t_eval.rejection_reason or "TRANSIT_INFEASIBLE"],
            )

        s_temp = float(t_eval.normalized_score)
        s_dens = float(
            np.clip(
                0.5 * (f_trk.observation_density + b_trk.observation_density)
                * min(1.0, min(f_trk.observed_frame_count, b_trk.observed_frame_count) / 4.0),
                0.0,
                1.0,
            )
        )

        app_avail, app_score, _, _, _ = score_appearance_similarity(
            f_trk.appearance_descriptor,
            b_trk.appearance_descriptor,
            config=self.config.appearance_config,
        )

        reasons: list[str] = []
        if app_avail and app_score is not None:
            comp_score = float(0.50 * s_temp + 0.30 * app_score + 0.20 * s_dens)
            breakdown = {
                "temporal_transit_score": s_temp,
                "appearance_score": app_score,
                "density_quality_score": s_dens,
            }
            if s_temp < 0.35:
                reasons.append("WEAK_TEMPORAL_TRANSIT_FOR_ORPHAN")
                return False, comp_score, breakdown, reasons
        else:
            comp_score = float(0.75 * s_temp + 0.25 * s_dens)
            breakdown = {
                "temporal_transit_score": s_temp,
                "density_quality_score": s_dens,
            }
            if (
                self.config.require_appearance_or_strong_temporal
                and s_temp < self.config.min_strong_temporal_score_without_appearance
            ):
                reasons.append(
                    f"WEAK_TEMPORAL_SCORE_{s_temp:.3f}_WITHOUT_APPEARANCE_BELOW_{self.config.min_strong_temporal_score_without_appearance}"
                )
                return False, comp_score, breakdown, reasons

        return True, float(np.clip(comp_score, 0.0, 1.0)), breakdown, reasons

    def associate_orphans(
        self,
        front_orphans: list[OrphanRiderTracklet],
        back_orphans: list[OrphanRiderTracklet],
        front_motos: list[MotorcycleTracklet],
        back_motos: list[MotorcycleTracklet],
        moto_results: list[MotorcycleAssociationResult],
        epsilon_sync_ms: float = 0.0,
        delta_pts_front_ms: float = 0.0,
        delta_pts_back_ms: float = 0.0,
    ) -> list[OrphanAssociationResult]:
        """
        Evaluate cross-camera orphan rider associations and return a list of
        `OrphanAssociationResult` records covering every `OrphanRiderTracklet`.
        """
        results: list[OrphanAssociationResult] = []
        if not front_orphans and not back_orphans:
            return results

        if not self.config.enable_orphan_matching:
            for fo in front_orphans:
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=fo.orphan_track_id,
                        back_track_id=None,
                        front_track_kind="orphan_rider",
                        back_track_kind=None,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=["ORPHAN_MATCHING_DISABLED"],
                        cue_breakdown={},
                        implies_confirmed_motorcycle_link=False,
                    )
                )
            for bo in back_orphans:
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=None,
                        back_track_id=bo.orphan_track_id,
                        front_track_kind=None,
                        back_track_kind="orphan_rider",
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=["ORPHAN_MATCHING_DISABLED"],
                        cue_breakdown={},
                        implies_confirmed_motorcycle_link=False,
                    )
                )
            return results

        # Separate insufficient-evidence orphans first
        eligible_f: list[_OrphanCandidateItem] = []
        eligible_b: list[_OrphanCandidateItem] = []
        handled_f_ids: set[str] = set()
        handled_b_ids: set[str] = set()

        for fo in front_orphans:
            if fo.observed_frame_count < self.config.min_observed_frames_orphan:
                handled_f_ids.add(fo.orphan_track_id)
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=fo.orphan_track_id,
                        back_track_id=None,
                        front_track_kind="orphan_rider",
                        back_track_kind=None,
                        outcome=AssociationOutcome.INSUFFICIENT_EVIDENCE,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=[
                            f"ORPHAN_OBSERVED_FRAMES_{fo.observed_frame_count}_BELOW_MIN_{self.config.min_observed_frames_orphan}"
                        ],
                        cue_breakdown={},
                        implies_confirmed_motorcycle_link=False,
                    )
                )
            else:
                eligible_f.append(
                    _OrphanCandidateItem(
                        camera_id="front",
                        track_id=fo.orphan_track_id,
                        track_kind="orphan_rider",
                        tracklet=fo,
                        is_primary_orphan=True,
                    )
                )

        for bo in back_orphans:
            if bo.observed_frame_count < self.config.min_observed_frames_orphan:
                handled_b_ids.add(bo.orphan_track_id)
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=None,
                        back_track_id=bo.orphan_track_id,
                        front_track_kind=None,
                        back_track_kind="orphan_rider",
                        outcome=AssociationOutcome.INSUFFICIENT_EVIDENCE,
                        association_score=0.0,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=[
                            f"ORPHAN_OBSERVED_FRAMES_{bo.observed_frame_count}_BELOW_MIN_{self.config.min_observed_frames_orphan}"
                        ],
                        cue_breakdown={},
                        implies_confirmed_motorcycle_link=False,
                    )
                )
            else:
                eligible_b.append(
                    _OrphanCandidateItem(
                        camera_id="back",
                        track_id=bo.orphan_track_id,
                        track_kind="orphan_rider",
                        tracklet=bo,
                        is_primary_orphan=True,
                    )
                )

        # Also include RiderTracklets from UNMATCHED motorcycles as potential cross-camera targets
        unmatched_f_motos = {
            r.front_moto_track_id
            for r in moto_results
            if r.outcome == AssociationOutcome.UNMATCHED and r.front_moto_track_id is not None
        }
        unmatched_b_motos = {
            r.back_moto_track_id
            for r in moto_results
            if r.outcome == AssociationOutcome.UNMATCHED and r.back_moto_track_id is not None
        }

        if eligible_b:
            for fm in front_motos:
                if fm.moto_track_id in unmatched_f_motos:
                    for rt in fm.rider_tracklets.values():
                        if rt.observed_frame_count >= self.config.min_observed_frames_orphan:
                            eligible_f.append(
                                _OrphanCandidateItem(
                                    camera_id="front",
                                    track_id=rt.subtrack_id,
                                    track_kind="unmatched_moto_rider",
                                    tracklet=rt,
                                    is_primary_orphan=False,
                                )
                            )

        if eligible_f:
            for bm in back_motos:
                if bm.moto_track_id in unmatched_b_motos:
                    for rt in bm.rider_tracklets.values():
                        if rt.observed_frame_count >= self.config.min_observed_frames_orphan:
                            eligible_b.append(
                                _OrphanCandidateItem(
                                    camera_id="back",
                                    track_id=rt.subtrack_id,
                                    track_kind="unmatched_moto_rider",
                                    tracklet=rt,
                                    is_primary_orphan=False,
                                )
                            )

        m = len(eligible_f)
        n = len(eligible_b)

        if m == 0 or n == 0:
            for item in eligible_f:
                if item.is_primary_orphan and item.track_id not in handled_f_ids:
                    results.append(
                        OrphanAssociationResult(
                            schema_version=SCHEMA_VERSION,
                            global_orphan_link_id=None,
                            association_level="ORPHAN_LEVEL_LINK",
                            front_track_id=item.track_id,
                            back_track_id=None,
                            front_track_kind=item.track_kind,
                            back_track_kind=None,
                            outcome=AssociationOutcome.UNMATCHED,
                            association_score=0.0,
                            second_best_score=0.0,
                            ambiguity_margin=0.0,
                            decision_reasons=["NO_OPPOSITE_CAMERA_ORPHAN_CANDIDATES"],
                            cue_breakdown={},
                            implies_confirmed_motorcycle_link=False,
                        )
                    )
            for item in eligible_b:
                if item.is_primary_orphan and item.track_id not in handled_b_ids:
                    results.append(
                        OrphanAssociationResult(
                            schema_version=SCHEMA_VERSION,
                            global_orphan_link_id=None,
                            association_level="ORPHAN_LEVEL_LINK",
                            front_track_id=None,
                            back_track_id=item.track_id,
                            front_track_kind=None,
                            back_track_kind=item.track_kind,
                            outcome=AssociationOutcome.UNMATCHED,
                            association_score=0.0,
                            second_best_score=0.0,
                            ambiguity_margin=0.0,
                            decision_reasons=["NO_OPPOSITE_CAMERA_ORPHAN_CANDIDATES"],
                            cue_breakdown={},
                            implies_confirmed_motorcycle_link=False,
                        )
                    )
            return results

        score_mat = np.zeros((m, n), dtype=np.float64)
        gated_mat = np.zeros((m, n), dtype=bool)
        breakdowns: dict[tuple[int, int], dict[str, float]] = {}
        reasons_map: dict[tuple[int, int], list[str]] = {}

        for i, fi in enumerate(eligible_f):
            for j, bj in enumerate(eligible_b):
                # Require at least one side of the pair to be an actual OrphanRiderTracklet
                if not fi.is_primary_orphan and not bj.is_primary_orphan:
                    continue
                ok, sc, brk, r_list = self._score_pair(
                    fi, bj, epsilon_sync_ms, delta_pts_front_ms, delta_pts_back_ms
                )
                # Without strong positive appearance similarity (>= 0.85), prevent brief
                # orphan fragments from matching or creating false ambiguity against sustained
                # coupled riders of rejected/unmatched motorcycles via temporal-centroid shift.
                # Uncalibrated opposing-view HSV overlap (negative mean ~0.664 in TASK-10)
                # is insufficient to overturn extreme cross-kind duration asymmetry.
                app_val = brk.get("appearance_score", brk.get("appearance_hsv_score"))
                if (
                    ok
                    and fi.track_kind != bj.track_kind
                    and (app_val is None or float(app_val) < 0.85)
                ):
                    obs_f = int(fi.tracklet.observed_frame_count)
                    obs_b = int(bj.tracklet.observed_frame_count)
                    obs_min = min(obs_f, obs_b)
                    obs_max = max(1, max(obs_f, obs_b))
                    if obs_min < 15 and (obs_min / float(obs_max)) < 0.20:
                        ok = False
                        sc = 0.0
                        brk = {"temporal_transit_score": float(brk.get("temporal_transit_score", 0.0))}
                        suffix = "WITHOUT_APPEARANCE" if app_val is None else "WEAK_APPEARANCE"
                        r_list = [
                            f"UNMATCHED_MOTO_RIDER_DURATION_ASYMMETRY_{obs_f}vs{obs_b}_{suffix}"
                        ]
                score_mat[i, j] = sc
                gated_mat[i, j] = ok
                breakdowns[(i, j)] = brk
                reasons_map[(i, j)] = r_list

        cost_mat = 1.0 - score_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        assigned_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))

        next_orphan_global_idx = 1
        for i, j in assigned_pairs:
            fi = eligible_f[i]
            bj = eligible_b[j]
            if not fi.is_primary_orphan and not bj.is_primary_orphan:
                continue

            best_sc = float(score_mat[i, j])
            other_row = [float(score_mat[i, j2]) for j2 in range(n) if j2 != j]
            other_col = [float(score_mat[i2, j]) for i2 in range(m) if i2 != i]
            second_best = max(other_row + other_col, default=0.0)
            margin = float(best_sc - second_best)

            if not gated_mat[i, j] or best_sc < self.config.min_orphan_association_score:
                continue

            if margin < self.config.orphan_ambiguous_margin_threshold:
                handled_f_ids.add(fi.track_id)
                handled_b_ids.add(bj.track_id)
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=fi.track_id,
                        back_track_id=bj.track_id,
                        front_track_kind=fi.track_kind,
                        back_track_kind=bj.track_kind,
                        outcome=AssociationOutcome.AMBIGUOUS_ABSTAIN,
                        association_score=best_sc,
                        second_best_score=second_best,
                        ambiguity_margin=margin,
                        decision_reasons=[
                            f"ORPHAN_AMBIGUOUS_MARGIN_{margin:.3f}_BELOW_{self.config.orphan_ambiguous_margin_threshold}"
                        ],
                        cue_breakdown=breakdowns.get((i, j), {}),
                        implies_confirmed_motorcycle_link=False,
                    )
                )
                continue

            gid = f"global_orphan_link_{next_orphan_global_idx:04d}"
            next_orphan_global_idx += 1
            handled_f_ids.add(fi.track_id)
            handled_b_ids.add(bj.track_id)
            results.append(
                OrphanAssociationResult(
                    schema_version=SCHEMA_VERSION,
                    global_orphan_link_id=gid,
                    association_level="ORPHAN_LEVEL_LINK",
                    front_track_id=fi.track_id,
                    back_track_id=bj.track_id,
                    front_track_kind=fi.track_kind,
                    back_track_kind=bj.track_kind,
                    outcome=AssociationOutcome.LINKED,
                    association_score=best_sc,
                    second_best_score=second_best,
                    ambiguity_margin=margin,
                    decision_reasons=[
                        "CONFIRMED_ORPHAN_LEVEL_LINK_WITHOUT_MOTORCYCLE_CONFIRMATION"
                    ],
                    cue_breakdown=breakdowns.get((i, j), {}),
                    implies_confirmed_motorcycle_link=False,
                )
            )

        # Record remaining primary orphan tracklets as UNMATCHED
        for i, fi in enumerate(eligible_f):
            if fi.is_primary_orphan and fi.track_id not in handled_f_ids:
                best_j = int(np.argmax(score_mat[i, :]))
                best_sc = float(score_mat[i, best_j])
                r_list = reasons_map.get((i, best_j), []) or [
                    f"ORPHAN_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_orphan_association_score}"
                ]
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=fi.track_id,
                        back_track_id=None,
                        front_track_kind=fi.track_kind,
                        back_track_kind=None,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=best_sc,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=r_list,
                        cue_breakdown=breakdowns.get((i, best_j), {}),
                        implies_confirmed_motorcycle_link=False,
                    )
                )

        for j, bj in enumerate(eligible_b):
            if bj.is_primary_orphan and bj.track_id not in handled_b_ids:
                best_i = int(np.argmax(score_mat[:, j]))
                best_sc = float(score_mat[best_i, j])
                r_list = reasons_map.get((best_i, j), []) or [
                    f"ORPHAN_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_orphan_association_score}"
                ]
                results.append(
                    OrphanAssociationResult(
                        schema_version=SCHEMA_VERSION,
                        global_orphan_link_id=None,
                        association_level="ORPHAN_LEVEL_LINK",
                        front_track_id=None,
                        back_track_id=bj.track_id,
                        front_track_kind=None,
                        back_track_kind=bj.track_kind,
                        outcome=AssociationOutcome.UNMATCHED,
                        association_score=best_sc,
                        second_best_score=0.0,
                        ambiguity_margin=0.0,
                        decision_reasons=r_list,
                        cue_breakdown=breakdowns.get((best_i, j), {}),
                        implies_confirmed_motorcycle_link=False,
                    )
                )

        return results
