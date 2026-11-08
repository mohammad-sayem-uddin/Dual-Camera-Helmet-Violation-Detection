"""Cross-Camera Association Audit Formatter and Exporter (`AssociationAuditor`).

Implements Section 12 Requirements:
1. Generates complete, explainable audit records for every motorcycle, rider, and
   orphan decision (`LINKED`, `UNMATCHED`, `AMBIGUOUS_ABSTAIN`, `INSUFFICIENT_EVIDENCE`).
2. Preserves strict uncertainty separation (`epsilon_sync_ms`, `delta_pts_front_ms`,
   `delta_pts_back_ms`, `delta_transit_sec`, `transit_uncertainty_sec`).
3. Exports structured CSV and JSON diagnostics (`tracklet_association_audit.csv`,
   `cue_score_distributions.csv`, `phase12e_diagnostics_summary.json`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.association.tracklet_models import (
    CrossCameraAssociationBatchResult,
)


class AssociationAuditor:
    """Formats and exports Phase 12E cross-camera association audit artifacts."""

    @staticmethod
    def build_flat_audit_rows(
        batch_result: CrossCameraAssociationBatchResult,
    ) -> list[dict[str, Any]]:
        """
        Convert a `CrossCameraAssociationBatchResult` into flat tabular audit rows
        covering motorcycle-level, rider-level, and orphan-level decisions.
        """
        rows: list[dict[str, Any]] = []
        tq = batch_result.timestamp_quality

        for m_res in batch_result.motorcycle_results:
            psd = m_res.pair_score_detail
            eps_ms = psd.epsilon_sync_ms if psd else tq.epsilon_sync_ms
            dp_f_ms = psd.delta_pts_front_ms if psd else tq.mean_delta_pts_front_ms
            dp_b_ms = psd.delta_pts_back_ms if psd else tq.mean_delta_pts_back_ms
            d_tr_sec = psd.delta_transit_sec if psd else 0.0
            tr_unc_sec = psd.transit_uncertainty_sec if psd else 0.0

            def _cue_score(name: str) -> float | None:
                if psd is None or name not in psd.cue_details:
                    return None
                return psd.cue_details[name].normalized_score

            def _cue_eff_w(name: str) -> float:
                if psd is None or name not in psd.cue_details:
                    return 0.0
                return psd.cue_details[name].effective_weight

            rows.append(
                {
                    "pair_id": batch_result.pair_id,
                    "entity_level": "MOTORCYCLE",
                    "global_id": m_res.global_motorcycle_id or "",
                    "front_track_id": m_res.front_moto_track_id or "",
                    "back_track_id": m_res.back_moto_track_id or "",
                    "outcome": m_res.outcome.value,
                    "association_score": round(float(m_res.association_score), 4),
                    "second_best_score": round(float(m_res.second_best_score), 4),
                    "ambiguity_margin": round(float(m_res.ambiguity_margin), 4),
                    "swap_margin_2x2": (
                        round(float(m_res.swap_margin_2x2), 4)
                        if m_res.swap_margin_2x2 is not None
                        else ""
                    ),
                    "active_cues_count": psd.active_cues_count if psd else 0,
                    "available_cues": ";".join(psd.available_cue_names) if psd else "",
                    "unavailable_cues": ";".join(psd.unavailable_cue_names) if psd else "",
                    "cue_temporal_transit_score": (
                        round(float(_cue_score("temporal_transit")), 4)
                        if _cue_score("temporal_transit") is not None
                        else ""
                    ),
                    "cue_temporal_transit_eff_weight": round(_cue_eff_w("temporal_transit"), 4),
                    "cue_rider_cardinality_score": (
                        round(float(_cue_score("rider_cardinality")), 4)
                        if _cue_score("rider_cardinality") is not None
                        else ""
                    ),
                    "cue_rider_cardinality_eff_weight": round(_cue_eff_w("rider_cardinality"), 4),
                    "cue_appearance_hsv_score": (
                        round(float(_cue_score("appearance_hsv")), 4)
                        if _cue_score("appearance_hsv") is not None
                        else ""
                    ),
                    "cue_appearance_hsv_eff_weight": round(_cue_eff_w("appearance_hsv"), 4),
                    "cue_lane_spatial_score": (
                        round(float(_cue_score("lane_spatial_context")), 4)
                        if _cue_score("lane_spatial_context") is not None
                        else ""
                    ),
                    "cue_convoy_order_score": (
                        round(float(_cue_score("convoy_order")), 4)
                        if _cue_score("convoy_order") is not None
                        else ""
                    ),
                    "epsilon_sync_ms": round(float(eps_ms), 4),
                    "delta_pts_front_ms": round(float(dp_f_ms), 4),
                    "delta_pts_back_ms": round(float(dp_b_ms), 4),
                    "delta_transit_sec": round(float(d_tr_sec), 6),
                    "transit_uncertainty_sec": round(float(tr_unc_sec), 6),
                    "decision_reasons": ";".join(m_res.decision_reasons),
                }
            )

            for r_res in m_res.rider_associations:
                rows.append(
                    {
                        "pair_id": batch_result.pair_id,
                        "entity_level": "RIDER_SUBTRACK",
                        "global_id": r_res.parent_global_motorcycle_id,
                        "front_track_id": r_res.front_subtrack_id or "",
                        "back_track_id": r_res.back_subtrack_id or "",
                        "outcome": r_res.outcome.value,
                        "association_score": round(float(r_res.association_score), 4),
                        "second_best_score": round(float(r_res.second_best_score), 4),
                        "ambiguity_margin": round(float(r_res.ambiguity_margin), 4),
                        "swap_margin_2x2": "",
                        "active_cues_count": len(r_res.cue_breakdown),
                        "available_cues": ";".join(sorted(r_res.cue_breakdown.keys())),
                        "unavailable_cues": "",
                        "cue_temporal_transit_score": r_res.cue_breakdown.get(
                            "temporal_coverage_score", ""
                        ),
                        "cue_temporal_transit_eff_weight": "",
                        "cue_rider_cardinality_score": r_res.cue_breakdown.get(
                            "slot_role_score", ""
                        ),
                        "cue_rider_cardinality_eff_weight": "",
                        "cue_appearance_hsv_score": r_res.cue_breakdown.get(
                            "appearance_score", ""
                        ),
                        "cue_appearance_hsv_eff_weight": "",
                        "cue_lane_spatial_score": "",
                        "cue_convoy_order_score": "",
                        "epsilon_sync_ms": round(float(eps_ms), 4),
                        "delta_pts_front_ms": round(float(dp_f_ms), 4),
                        "delta_pts_back_ms": round(float(dp_b_ms), 4),
                        "delta_transit_sec": round(float(d_tr_sec), 6),
                        "transit_uncertainty_sec": round(float(tr_unc_sec), 6),
                        "decision_reasons": ";".join(r_res.decision_reasons),
                    }
                )

        for o_res in batch_result.orphan_results:
            rows.append(
                {
                    "pair_id": batch_result.pair_id,
                    "entity_level": o_res.association_level,
                    "global_id": o_res.global_orphan_link_id or "",
                    "front_track_id": o_res.front_track_id or "",
                    "back_track_id": o_res.back_track_id or "",
                    "outcome": o_res.outcome.value,
                    "association_score": round(float(o_res.association_score), 4),
                    "second_best_score": round(float(o_res.second_best_score), 4),
                    "ambiguity_margin": round(float(o_res.ambiguity_margin), 4),
                    "swap_margin_2x2": "",
                    "active_cues_count": len(o_res.cue_breakdown),
                    "available_cues": ";".join(sorted(o_res.cue_breakdown.keys())),
                    "unavailable_cues": "",
                    "cue_temporal_transit_score": o_res.cue_breakdown.get(
                        "temporal_transit_score", ""
                    ),
                    "cue_temporal_transit_eff_weight": "",
                    "cue_rider_cardinality_score": "",
                    "cue_rider_cardinality_eff_weight": "",
                    "cue_appearance_hsv_score": o_res.cue_breakdown.get("appearance_score", ""),
                    "cue_appearance_hsv_eff_weight": "",
                    "cue_lane_spatial_score": "",
                    "cue_convoy_order_score": "",
                    "epsilon_sync_ms": round(float(tq.epsilon_sync_ms), 4),
                    "delta_pts_front_ms": round(float(tq.mean_delta_pts_front_ms), 4),
                    "delta_pts_back_ms": round(float(tq.mean_delta_pts_back_ms), 4),
                    "delta_transit_sec": 0.0,
                    "transit_uncertainty_sec": 0.0,
                    "decision_reasons": ";".join(o_res.decision_reasons),
                }
            )
        return rows

    @staticmethod
    def build_cue_distribution_rows(
        batch_result: CrossCameraAssociationBatchResult,
    ) -> list[dict[str, Any]]:
        """
        Convert all evaluated `CandidatePairScore` cue details into flat rows for
        `cue_score_distributions.csv`.
        """
        rows: list[dict[str, Any]] = []
        for cps in batch_result.candidate_pair_scores:
            for cue_name, c_det in cps.cue_details.items():
                rows.append(
                    {
                        "pair_id": batch_result.pair_id,
                        "front_moto_track_id": cps.front_moto_track_id,
                        "back_moto_track_id": cps.back_moto_track_id,
                        "passed_candidate_gate": cps.passed_candidate_gate,
                        "composite_score": round(float(cps.composite_score), 4),
                        "cue_name": cue_name,
                        "available": c_det.available,
                        "unavailable_reason": c_det.unavailable_reason or "",
                        "quality": round(float(c_det.quality), 4),
                        "normalized_score": (
                            round(float(c_det.normalized_score), 4)
                            if c_det.normalized_score is not None
                            else ""
                        ),
                        "nominal_weight": round(float(c_det.nominal_weight), 4),
                        "effective_weight": round(float(c_det.effective_weight), 4),
                        "weighted_contribution": round(float(c_det.weighted_contribution), 4),
                        "epsilon_sync_ms": round(float(cps.epsilon_sync_ms), 4),
                        "delta_pts_front_ms": round(float(cps.delta_pts_front_ms), 4),
                        "delta_pts_back_ms": round(float(cps.delta_pts_back_ms), 4),
                        "delta_transit_sec": round(float(cps.delta_transit_sec), 6),
                        "transit_uncertainty_sec": round(float(cps.transit_uncertainty_sec), 6),
                    }
                )
        return rows

    @staticmethod
    def write_csv(rows: list[dict[str, Any]], output_path: str | Path, fieldnames: list[str] | None = None) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows and not fieldnames:
            return
        cols = fieldnames or list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    @staticmethod
    def write_json(data: dict[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
