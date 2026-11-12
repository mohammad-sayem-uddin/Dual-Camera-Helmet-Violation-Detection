"""Dual-camera track-to-event fusion orchestrator (`DualCameraFusionEngine`).

Connects GAP-C `RiderSubTrack`/`OrphanRiderTrack` genuine observation evidence,
`RiderAssociationResult`, `OrphanAssociationResult`, and `TimestampQualityRecord`
synchronization uncertainty into quality-weighted log-odds decisions with formal
contradiction (`EVIDENCE_CONFLICT_ABSTAIN`) and ambiguous-link quarantine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.association.tracklet_models import (
    AssociationOutcome,
    CrossCameraAssociationBatchResult,
    OrphanAssociationResult,
    RiderAssociationResult,
    TimestampQualityRecord,
)
from src.fusion.evidence_fusion import (
    FusionConfig,
    FusionDecisionState,
    RiderFusionResult,
    ViewRiderEvidence,
    compute_mean_head_area_from_history,
    compute_sync_reliability,
    evaluate_single_view_belief,
    fuse_rider_beliefs,
)


_PLATE_UNAVAILABLE_FRONT = {
    "plate_visibility_tier": "NOT_ASSESSED",
    "gt_plate_text_normalized": None,
    "cross_camera_plate_link_permitted": False,
    "note": "Front-view only event: no Back view for rear plate linking.",
}

_PLATE_QUARANTINED = {
    "plate_visibility_tier": "NOT_ASSESSED",
    "gt_plate_text_normalized": None,
    "cross_camera_plate_link_permitted": False,
    "note": "Cross-camera plate attribution blocked by AMBIGUOUS_ASSOC_ABSTAIN quarantine.",
}

_PLATE_DEFAULT = {
    "plate_visibility_tier": "NOT_ASSESSED",
    "gt_plate_text_normalized": None,
    "note": (
        "Plate assessment is out of scope for the integrated pipeline "
        "(no Phase 12F/OCR); no plate text is asserted."
    ),
}


@dataclass(frozen=True)
class OrphanFusionResult:
    """First-class fused or single-view helmet decision for an orphan rider link."""

    global_orphan_link_id: str | None
    front_track_id: str | None
    back_track_id: str | None
    association_outcome: str
    rider_fusion: RiderFusionResult
    implies_confirmed_motorcycle_link: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_orphan_link_id": self.global_orphan_link_id,
            "front_track_id": self.front_track_id,
            "back_track_id": self.back_track_id,
            "association_outcome": self.association_outcome,
            "helmet_state": self.rider_fusion.helmet_state,
            "fusion_decision_state": self.rider_fusion.fusion_decision_state.value,
            "fused_p_no_helmet": round(float(self.rider_fusion.fused_p_no_helmet), 4),
            "contradiction_k": round(float(self.rider_fusion.contradiction_k), 4),
            "cross_view_fused": bool(self.rider_fusion.cross_view_fused),
            "quarantined": bool(self.rider_fusion.quarantined),
            "abstained": bool(self.rider_fusion.abstained),
            "implies_confirmed_motorcycle_link": bool(self.implies_confirmed_motorcycle_link),
            "decision_reasons": list(self.rider_fusion.decision_reasons),
        }


@dataclass
class PairFusionOutput:
    """Complete output of dual-camera (or projected single-camera) fusion for one pair."""

    pair_id: str
    mode: str
    sync_reliability: float
    events: list[dict[str, Any]]
    orphan_results: list[OrphanFusionResult] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def _compute_near_border_fraction(
    pts_history: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    frame_wh: tuple[int, int] | None,
    margin_ratio: float = 0.02,
) -> float:
    if not frame_wh or not pts_history:
        return 0.0
    w_img, h_img = float(frame_wh[0]), float(frame_wh[1])
    if w_img <= 0.0 or h_img <= 0.0:
        return 0.0
    mx = margin_ratio * w_img
    my = margin_ratio * h_img
    near = 0
    total = 0
    for entry in pts_history:
        if len(entry) < 3 or len(entry[2]) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in entry[2]]
        total += 1
        if x1 <= mx or y1 <= my or x2 >= (w_img - mx) or y2 >= (h_img - my):
            near += 1
    return float(near / total) if total > 0 else 0.0


def _canonical_slot_and_role(raw_slot: int, role: str) -> tuple[int, str]:
    clean = str(role or "").strip()
    if clean == "driver":
        return 0, "driver"
    if clean == "pillion_1":
        return 1, "pillion_1"
    if clean == "pillion_2":
        return 2, "pillion_2"
    if clean.startswith("pillion_"):
        try:
            idx = int(clean.split("_", 1)[1])
            if idx >= 1:
                return idx, clean
        except (ValueError, IndexError):
            pass
    if not clean or clean.startswith("ambiguous_slot_"):
        fallback = {0: "driver", 1: "pillion_1", 2: "pillion_2"}.get(raw_slot, f"slot_{raw_slot}")
        return int(raw_slot), fallback
    return 100 + int(raw_slot), clean


def extract_camera_moto_and_orphan_records(
    cam_out: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, ViewRiderEvidence]]:
    """Extract normalized moto track records and orphan evidence from a `CameraRunOutput`."""
    pipe = cam_out.pipeline
    shift = float(cam_out.canonical_shift_sec)
    cam_id = str(cam_out.camera_id)
    frame_wh = getattr(pipe, "frame_dimensions", None)

    moto_records: dict[str, dict[str, Any]] = {}
    all_motos = list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values())
    for m in all_motos:
        t0 = round(float(m.first_pts_sec) + shift, 6)
        t1 = round(float(m.last_pts_sec) + shift, 6)
        subs: dict[str, ViewRiderEvidence] = {}
        subs_by_slot: dict[int, ViewRiderEvidence] = {}
        for slot, sub in sorted(m.rider_subtracks.items()):
            ev_dict = sub.helmet_evidence.export(sub.aggregated_confidence)
            pts_hist = getattr(sub, "pts_history", ())
            area = compute_mean_head_area_from_history(pts_hist)
            border_frac = _compute_near_border_fraction(pts_hist, frame_wh)
            ro = {
                "temporal_state": sub.temporal_state.value if hasattr(sub.temporal_state, "value") else str(sub.temporal_state),
                "alerts_count": len(sub.alerts),
                "consensus_slot_role": getattr(sub, "consensus_slot_role", sub.slot_role),
                "genuine_observation_helmet_evidence": ev_dict,
            }
            ev = ViewRiderEvidence.from_phase10_readonly(
                camera_id=cam_id,
                subtrack_id=sub.subtrack_id,
                slot_role=sub.slot_role,
                readonly_dict=ro,
                slot_index=int(slot),
                consensus_slot_role=getattr(sub, "consensus_slot_role", sub.slot_role),
                ordering_confident=bool(getattr(sub, "ordering_confident", True)),
                alert_ids=tuple(a.alert_id for a in sub.alerts),
                mean_head_area_px=area,
                near_border_fraction=border_frac,
                t_start_ref_sec=t0,
                t_end_ref_sec=t1,
            )
            subs[sub.subtrack_id] = ev
            subs_by_slot[int(slot)] = ev
        moto_records[m.moto_track_id] = {
            "camera_id": cam_id,
            "track_id": m.moto_track_id,
            "t_start_ref_sec": t0,
            "t_end_ref_sec": t1,
            "subs_by_id": subs,
            "subs_by_slot": subs_by_slot,
        }

    orphan_records: dict[str, ViewRiderEvidence] = {}
    exported = (
        list(pipe.exported_orphan_tracks())
        if hasattr(pipe, "exported_orphan_tracks")
        else list(pipe.terminated_orphan_tracks) + list(pipe.active_orphan_tracks.values())
    )
    for o in exported:
        ev_dict = o.helmet_evidence.export(o.phase10_track.aggregator.get_aggregated_confidence())
        pts_hist = getattr(o, "pts_history", ())
        ro = {
            "temporal_state": o.phase10_track.state.value if hasattr(o.phase10_track.state, "value") else str(o.phase10_track.state),
            "alerts_count": len(o.alerts),
            "genuine_observation_helmet_evidence": ev_dict,
        }
        orphan_records[o.orphan_track_id] = ViewRiderEvidence.from_phase10_readonly(
            camera_id=cam_id,
            subtrack_id=o.orphan_track_id,
            slot_role="orphan_rider",
            readonly_dict=ro,
            slot_index=None,
            consensus_slot_role="orphan_rider",
            ordering_confident=False,
            alert_ids=tuple(a.alert_id for a in o.alerts),
            mean_head_area_px=compute_mean_head_area_from_history(pts_hist),
            near_border_fraction=_compute_near_border_fraction(pts_hist, frame_wh),
            t_start_ref_sec=round(float(o.first_pts_sec) + shift, 6),
            t_end_ref_sec=round(float(o.last_pts_sec) + shift, 6),
        )

    return moto_records, orphan_records


def derive_event_type_from_fused_riders(
    rider_results: list[RiderFusionResult],
) -> tuple[str, bool | None]:
    """Derive motorcycle-level violation event type from fused rider decisions."""
    if not rider_results:
        return "INSUFFICIENT_VISUAL_EVIDENCE", None

    # Contradiction or quarantine in an ambiguous rider association preserves explicit abstention
    non_unresolved = [
        r
        for r in rider_results
        if not (str(r.role).startswith("ambiguous_slot_") or r.role == "uncertain")
    ]
    drivers = [r for r in non_unresolved if r.role == "driver"]
    pillions = [r for r in non_unresolved if r.role.startswith("pillion_") or r.canonical_slot > 0]

    if not drivers and any(r.role.startswith("ambiguous_slot_") or r.role == "uncertain" for r in rider_results):
        return "INSUFFICIENT_VISUAL_EVIDENCE", None

    d_nh = any(r.helmet_state == "NO_HELMET" for r in drivers)
    p_nh = any(r.helmet_state == "NO_HELMET" for r in pillions)
    if d_nh and p_nh:
        return "BOTH_NO_HELMET", True
    if d_nh:
        return "DRIVER_NO_HELMET", True
    if p_nh:
        return "PILLION_NO_HELMET", True

    if not drivers:
        return "INSUFFICIENT_VISUAL_EVIDENCE", None

    # Compliant only when every driver & pillion slot is positively HELMET with zero abstention/conflict
    if all(r.helmet_state == "HELMET" for r in drivers + pillions) and not any(
        r.fusion_decision_state == FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN
        for r in rider_results
    ):
        return "NO_VIOLATION", False
    return "INSUFFICIENT_VISUAL_EVIDENCE", None


class DualCameraFusionEngine:
    """Orchestrates confidence-aware dual-camera and single-camera event fusion."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def _allocate_canonical_slot(
        self,
        raw_slot: int,
        role: str,
        used_slots: set[int],
    ) -> tuple[int, str]:
        canon_slot, canon_role = _canonical_slot_and_role(raw_slot, role)
        if canon_slot in used_slots:
            if raw_slot not in used_slots:
                canon_slot = raw_slot
            else:
                candidate = max(canon_slot + 1, 100 + raw_slot)
                while candidate in used_slots:
                    candidate += 1
                canon_slot = candidate
        used_slots.add(canon_slot)
        return canon_slot, canon_role

    def _fuse_single_moto_riders(
        self,
        rec: dict[str, Any],
        *,
        sync_reliability: float,
        quarantined_reason: str | None = None,
    ) -> list[RiderFusionResult]:
        used: set[int] = set()
        out: list[RiderFusionResult] = []
        for raw_slot, ev in sorted(rec["subs_by_slot"].items()):
            c_slot, c_role = self._allocate_canonical_slot(raw_slot, ev.effective_role, used)
            out.append(
                evaluate_single_view_belief(
                    ev,
                    canonical_slot=c_slot,
                    role=c_role,
                    sync_reliability=sync_reliability,
                    quarantined_reason=quarantined_reason,
                    config=self.config,
                )
            )
        return out

    def _fuse_linked_moto_riders(
        self,
        front_rec: dict[str, Any] | None,
        back_rec: dict[str, Any] | None,
        rider_assocs: list[RiderAssociationResult],
        *,
        sync_reliability: float,
    ) -> list[RiderFusionResult]:
        if front_rec is None and back_rec is None:
            return []
        if front_rec is not None and back_rec is None:
            return self._fuse_single_moto_riders(front_rec, sync_reliability=sync_reliability)
        if back_rec is not None and front_rec is None:
            return self._fuse_single_moto_riders(back_rec, sync_reliability=sync_reliability)

        assert front_rec is not None and back_rec is not None
        f_subs = front_rec["subs_by_id"]
        b_subs = back_rec["subs_by_id"]
        used_f: set[str] = set()
        used_b: set[str] = set()
        used_slots: set[int] = set()
        fused: list[RiderFusionResult] = []

        # 1. Pair riders confirmed by RiderAssociationResult first (respecting IDs across shifted raw slots)
        for ra in rider_assocs:
            fid = ra.front_subtrack_id
            bid = ra.back_subtrack_id
            f_ev = f_subs.get(fid) if fid else None
            b_ev = b_subs.get(bid) if bid else None
            if ra.outcome == AssociationOutcome.LINKED and f_ev is not None and b_ev is not None:
                used_f.add(fid)  # type: ignore[arg-type]
                used_b.add(bid)  # type: ignore[arg-type]
                raw_slot = f_ev.slot_index if f_ev.slot_index is not None else (b_ev.slot_index or 0)
                pref_role = (
                    f_ev.effective_role
                    if f_ev.effective_role != "uncertain"
                    else b_ev.effective_role
                )
                c_slot, c_role = self._allocate_canonical_slot(raw_slot, pref_role, used_slots)
                fused.append(
                    fuse_rider_beliefs(
                        f_ev,
                        b_ev,
                        canonical_slot=c_slot,
                        role=c_role,
                        association_score=ra.association_score,
                        sync_reliability=sync_reliability,
                        config=self.config,
                    )
                )
            elif ra.outcome == AssociationOutcome.AMBIGUOUS_ABSTAIN:
                # Never fuse ambiguous rider associations across views; keep distinct single-view evaluations
                # flagged as quarantined from cross-view rider pooling.
                q_tag = "RIDER_AMBIGUOUS_ASSOC_QUARANTINED"
                if f_ev is not None and fid and fid not in used_f:
                    used_f.add(fid)
                    c_slot, c_role = self._allocate_canonical_slot(
                        f_ev.slot_index or 0, f_ev.effective_role, used_slots
                    )
                    fused.append(
                        evaluate_single_view_belief(
                            f_ev,
                            canonical_slot=c_slot,
                            role=c_role,
                            sync_reliability=sync_reliability,
                            quarantined_reason=q_tag,
                            config=self.config,
                        )
                    )
                if b_ev is not None and bid and bid not in used_b:
                    used_b.add(bid)
                    c_slot, c_role = self._allocate_canonical_slot(
                        b_ev.slot_index or 0, b_ev.effective_role, used_slots
                    )
                    fused.append(
                        evaluate_single_view_belief(
                            b_ev,
                            canonical_slot=c_slot,
                            role=c_role,
                            sync_reliability=sync_reliability,
                            quarantined_reason=q_tag,
                            config=self.config,
                        )
                    )

        # Track explicit UNMATCHED / INSUFFICIENT_EVIDENCE pairs returned by RiderAssociation
        unmatched_rejected_pairs: set[tuple[str, str]] = {
            (ra.front_subtrack_id, ra.back_subtrack_id)
            for ra in rider_assocs
            if ra.outcome in (AssociationOutcome.UNMATCHED, AssociationOutcome.INSUFFICIENT_EVIDENCE)
            and ra.front_subtrack_id
            and ra.back_subtrack_id
        }

        # 2. Align remaining unique singleton driver / role pairs so a LINKED moto cannot emit
        #    two contradictory "driver" entries without undergoing belief fusion or conflict abstain.
        f_by_role: dict[str, list[ViewRiderEvidence]] = {}
        b_by_role: dict[str, list[ViewRiderEvidence]] = {}
        for ev in f_subs.values():
            if ev.track_id not in used_f:
                f_by_role.setdefault(ev.effective_role, []).append(ev)
        for ev in b_subs.values():
            if ev.track_id not in used_b:
                b_by_role.setdefault(ev.effective_role, []).append(ev)

        candidate_roles = (
            sorted(f_by_role.keys()) if not rider_assocs else (["driver"] if "driver" in f_by_role else [])
        )
        for role_name in candidate_roles:
            flist = f_by_role.get(role_name, [])
            blist = b_by_role.get(role_name, [])
            if role_name == "uncertain" or len(flist) != 1 or len(blist) != 1:
                continue
            f_ev, b_ev = flist[0], blist[0]
            used_f.add(f_ev.track_id)
            used_b.add(b_ev.track_id)
            c_slot, c_role = self._allocate_canonical_slot(
                f_ev.slot_index or 0, role_name, used_slots
            )
            explicitly_rejected = (f_ev.track_id, b_ev.track_id) in unmatched_rejected_pairs
            states_oppose = (
                f_ev.track_helmet_state in ("HELMET", "NO_HELMET")
                and b_ev.track_helmet_state in ("HELMET", "NO_HELMET")
                and f_ev.track_helmet_state != b_ev.track_helmet_state
            )
            if explicitly_rejected and states_oppose:
                # Contradictory driver labels when RiderAssociation rejected pairing -> conflict abstain
                from src.fusion.evidence_fusion import (
                    compute_contradiction_metric,
                    compute_view_posterior_and_weight,
                )

                f_sum = compute_view_posterior_and_weight(f_ev, self.config)
                b_sum = compute_view_posterior_and_weight(b_ev, self.config)
                k_conf = max(
                    compute_contradiction_metric(
                        f_sum.posterior_no_helmet, b_sum.posterior_no_helmet
                    ),
                    self.config.conflict_threshold,
                )
                fused.append(
                    RiderFusionResult(
                        canonical_slot=c_slot,
                        role=c_role,
                        helmet_state="UNCERTAIN",
                        provenance="UNCERTAIN",
                        fusion_decision_state=FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN,
                        fused_p_no_helmet=0.5,
                        fused_log_odds=0.0,
                        contradiction_k=k_conf,
                        sync_reliability=sync_reliability,
                        association_score=0.0,
                        cross_view_fused=False,
                        quarantined=True,
                        abstained=True,
                        decision_reasons=(
                            "UNMATCHED_DRIVER_CONTRADICTION_ABSTAIN",
                            f"EVIDENCE_CONFLICT_K_{k_conf:.3f}",
                        ),
                        violation_window_ref_sec=None,
                        front_evidence=f_ev,
                        back_evidence=b_ev,
                        front_summary=f_sum,
                        back_summary=b_sum,
                        raw_subtrack_slots=tuple(
                            sorted(
                                {s for s in (f_ev.slot_index, b_ev.slot_index) if s is not None}
                            )
                        ),
                        recorded_slot_roles=(role_name,),
                        alert_ids=tuple(sorted(set(f_ev.alert_ids) | set(b_ev.alert_ids))),
                    )
                )
            else:
                fused.append(
                    fuse_rider_beliefs(
                        f_ev,
                        b_ev,
                        canonical_slot=c_slot,
                        role=c_role,
                        association_score=1.0,
                        sync_reliability=sync_reliability,
                        config=self.config,
                    )
                )

        # 3. Remaining single-view rider subtracks on front or back
        for ev in sorted(f_subs.values(), key=lambda x: (x.slot_index or 0, x.track_id)):
            if ev.track_id in used_f:
                continue
            c_slot, c_role = self._allocate_canonical_slot(
                ev.slot_index or 0, ev.effective_role, used_slots
            )
            fused.append(
                evaluate_single_view_belief(
                    ev,
                    canonical_slot=c_slot,
                    role=c_role,
                    sync_reliability=sync_reliability,
                    config=self.config,
                )
            )
        for ev in sorted(b_subs.values(), key=lambda x: (x.slot_index or 0, x.track_id)):
            if ev.track_id in used_b:
                continue
            c_slot, c_role = self._allocate_canonical_slot(
                ev.slot_index or 0, ev.effective_role, used_slots
            )
            fused.append(
                evaluate_single_view_belief(
                    ev,
                    canonical_slot=c_slot,
                    role=c_role,
                    sync_reliability=sync_reliability,
                    config=self.config,
                )
            )

        fused.sort(key=lambda r: r.canonical_slot)
        return fused

    def _build_event_dict(
        self,
        *,
        front_rec: dict[str, Any] | None,
        back_rec: dict[str, Any] | None,
        rider_results: list[RiderFusionResult],
        outcome: str,
        score: float | None,
        reasons: list[str],
        link_status: str,
        sync_reliability: float,
        cross_view_fused: bool,
        quarantined: bool,
        quarantine_detail: dict[str, Any] | None = None,
        override_event_type: tuple[str, bool | None] | None = None,
    ) -> dict[str, Any]:
        if override_event_type is not None:
            etype, is_viol = override_event_type
        else:
            etype, is_viol = derive_event_type_from_fused_riders(rider_results)

        wins = [r.violation_window_ref_sec for r in rider_results if r.violation_window_ref_sec is not None]
        win = [min(w[0] for w in wins), max(w[1] for w in wins)] if wins else None
        alert_ids = sorted({aid for r in rider_results for aid in r.alert_ids})

        if quarantined:
            plate_block = dict(_PLATE_QUARANTINED)
        elif front_rec is not None and back_rec is None:
            plate_block = dict(_PLATE_UNAVAILABLE_FRONT)
        else:
            plate_block = dict(_PLATE_DEFAULT)

        times = [r["t_start_ref_sec"] for r in (front_rec, back_rec) if r is not None]
        any_conflict = any(
            r.fusion_decision_state == FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN
            for r in rider_results
        )
        max_k = max((r.contradiction_k for r in rider_results), default=0.0)

        payload: dict[str, Any] = {
            "global_moto_id": None,
            "cross_camera_link_status": link_status,
            "association_outcome": outcome,
            "association_score": round(float(score), 4) if score is not None else None,
            "decision_reasons": list(reasons),
            "rider_count": len(rider_results),
            "event_is_violation": is_viol,
            "violation_event_type": etype,
            "violation_window_ref_sec": win,
            "riders": [r.to_event_rider_dict() for r in rider_results],
            "front_track_id": front_rec["track_id"] if front_rec is not None else None,
            "back_track_id": back_rec["track_id"] if back_rec is not None else None,
            "phase10_alert_ids": alert_ids,
            "plate_ground_truth": plate_block,
            "fusion_metadata": {
                "cross_view_fused": bool(cross_view_fused),
                "quarantined": bool(quarantined),
                "has_evidence_conflict_abstain": bool(any_conflict),
                "max_contradiction_k": round(float(max_k), 4),
                "sync_reliability": round(float(sync_reliability), 4),
            },
            "_sort": min(times, default=0.0),
        }
        if quarantine_detail is not None:
            payload["fusion_metadata"]["quarantined_single_view_decisions"] = quarantine_detail
        return payload

    def fuse_orphans(
        self,
        orphan_assocs: list[OrphanAssociationResult],
        front_orphans: dict[str, ViewRiderEvidence],
        back_orphans: dict[str, ViewRiderEvidence],
        *,
        sync_reliability: float = 1.0,
    ) -> list[OrphanFusionResult]:
        """Apply evidence fusion / conflict / quarantine rules to cross-camera orphan links."""
        results: list[OrphanFusionResult] = []
        for orp in orphan_assocs:
            f_ev = front_orphans.get(orp.front_track_id) if orp.front_track_id else None
            b_ev = back_orphans.get(orp.back_track_id) if orp.back_track_id else None
            oc = orp.outcome.value
            q_reason = (
                "ORPHAN_AMBIGUOUS_ASSOC_QUARANTINED"
                if (oc == "AMBIGUOUS_ABSTAIN" and self.config.quarantine_ambiguous_links)
                else None
            )
            rf = fuse_rider_beliefs(
                f_ev,
                b_ev,
                canonical_slot=0,
                role="orphan_rider",
                association_score=orp.association_score if oc == "LINKED" else 0.0,
                sync_reliability=sync_reliability,
                quarantined_reason=q_reason,
                config=self.config,
            )
            results.append(
                OrphanFusionResult(
                    global_orphan_link_id=orp.global_orphan_link_id,
                    front_track_id=orp.front_track_id,
                    back_track_id=orp.back_track_id,
                    association_outcome=oc,
                    rider_fusion=rf,
                    implies_confirmed_motorcycle_link=bool(orp.implies_confirmed_motorcycle_link),
                )
            )
        return results

    def fuse_pair(
        self,
        pair_id: str,
        front_out: Any | None,
        back_out: Any | None,
        batch: CrossCameraAssociationBatchResult | None = None,
        *,
        timestamp_quality: TimestampQualityRecord | None = None,
        mode: str = "dual_fused",
    ) -> PairFusionOutput:
        """Fuse a dual-camera pair (or project a single-camera baseline cleanly)."""
        norm_mode = mode.lower()
        if norm_mode not in ("dual_fused", "front_only", "back_only"):
            raise ValueError(f"Unsupported fusion mode: {mode!r}")

        f_motos, f_orphans = (
            extract_camera_moto_and_orphan_records(front_out)
            if (front_out is not None and norm_mode in ("dual_fused", "front_only"))
            else ({}, {})
        )
        b_motos, b_orphans = (
            extract_camera_moto_and_orphan_records(back_out)
            if (back_out is not None and norm_mode in ("dual_fused", "back_only"))
            else ({}, {})
        )

        tq = timestamp_quality or (batch.timestamp_quality if batch is not None else None)
        eps_ms = float(tq.epsilon_sync_ms) if tq is not None else 0.0
        dpts_ms = (
            0.5 * (abs(float(tq.mean_delta_pts_front_ms)) + abs(float(tq.mean_delta_pts_back_ms)))
            if tq is not None
            else 0.0
        )
        phi_sync = compute_sync_reliability(eps_ms, dpts_ms, self.config) if norm_mode == "dual_fused" else 1.0

        raw_events: list[dict[str, Any]] = []
        covered: set[tuple[str, str]] = set()
        stats = {
            "linked_events": 0,
            "single_view_events": 0,
            "ambiguous_quarantined": 0,
            "evidence_conflict_abstains": 0,
            "low_confidence_abstains": 0,
        }

        if norm_mode == "dual_fused" and batch is not None:
            for mr in batch.motorcycle_results:
                f_rec = f_motos.get(mr.front_moto_track_id) if mr.front_moto_track_id else None
                b_rec = b_motos.get(mr.back_moto_track_id) if mr.back_moto_track_id else None
                if f_rec is not None:
                    covered.add(("front", f_rec["track_id"]))
                if b_rec is not None:
                    covered.add(("back", b_rec["track_id"]))

                oc = mr.outcome.value
                reasons = list(mr.decision_reasons)

                if oc == "LINKED":
                    rider_res = self._fuse_linked_moto_riders(
                        f_rec,
                        b_rec,
                        list(mr.rider_associations),
                        sync_reliability=phi_sync,
                    )
                    stats["linked_events"] += 1
                    raw_events.append(
                        self._build_event_dict(
                            front_rec=f_rec,
                            back_rec=b_rec,
                            rider_results=rider_res,
                            outcome=oc,
                            score=mr.association_score,
                            reasons=reasons,
                            link_status="LINKED_BOTH_VIEWS",
                            sync_reliability=phi_sync,
                            cross_view_fused=True,
                            quarantined=False,
                        )
                    )
                elif oc == "AMBIGUOUS_ABSTAIN" and self.config.quarantine_ambiguous_links and f_rec is not None and b_rec is not None:
                    stats["ambiguous_quarantined"] += 1
                    if self.config.split_ambiguous_moto_to_single_view_events:
                        for cam_name, c_rec in (("front", f_rec), ("back", b_rec)):
                            if c_rec is None:
                                continue
                            single_res = self._fuse_single_moto_riders(
                                c_rec,
                                sync_reliability=phi_sync,
                                quarantined_reason="AMBIGUOUS_MOTO_ASSOC_SPLIT_SINGLE_VIEW",
                            )
                            raw_events.append(
                                self._build_event_dict(
                                    front_rec=c_rec if cam_name == "front" else None,
                                    back_rec=c_rec if cam_name == "back" else None,
                                    rider_results=single_res,
                                    outcome=oc,
                                    score=mr.association_score,
                                    reasons=reasons + ["QUARANTINED_SINGLE_VIEW_SPLIT"],
                                    link_status="AMBIGUOUS_HUMAN_UNCERTAIN",
                                    sync_reliability=phi_sync,
                                    cross_view_fused=False,
                                    quarantined=True,
                                )
                            )
                    else:
                        f_res = (
                            self._fuse_single_moto_riders(
                                f_rec,
                                sync_reliability=phi_sync,
                                quarantined_reason="AMBIGUOUS_MOTO_ASSOC_QUARANTINED",
                            )
                            if f_rec is not None
                            else []
                        )
                        b_res = (
                            self._fuse_single_moto_riders(
                                b_rec,
                                sync_reliability=phi_sync,
                                quarantined_reason="AMBIGUOUS_MOTO_ASSOC_QUARANTINED",
                            )
                            if b_rec is not None
                            else []
                        )
                        q_detail = {
                            "front": (
                                {
                                    "event_type": derive_event_type_from_fused_riders(f_res)[0],
                                    "riders": [r.to_event_rider_dict() for r in f_res],
                                }
                                if f_rec is not None
                                else None
                            ),
                            "back": (
                                {
                                    "event_type": derive_event_type_from_fused_riders(b_res)[0],
                                    "riders": [r.to_event_rider_dict() for r in b_res],
                                }
                                if b_rec is not None
                                else None
                            ),
                        }
                        # Keep separated single-view rider slots if only one camera has track, else abstain
                        combined_riders = f_res if (b_rec is None) else (b_res if f_rec is None else [])
                        override_et = (
                            ("INSUFFICIENT_VISUAL_EVIDENCE", None)
                            if self.config.ambiguous_link_abstain_citation
                            else None
                        )
                        raw_events.append(
                            self._build_event_dict(
                                front_rec=f_rec,
                                back_rec=b_rec,
                                rider_results=combined_riders,
                                outcome=oc,
                                score=mr.association_score,
                                reasons=reasons + ["AMBIGUOUS_ASSOC_ABSTAIN_QUARANTINED"],
                                link_status="AMBIGUOUS_HUMAN_UNCERTAIN",
                                sync_reliability=phi_sync,
                                cross_view_fused=False,
                                quarantined=True,
                                quarantine_detail=q_detail,
                                override_event_type=override_et,
                            )
                        )
                else:
                    single_rec = f_rec if f_rec is not None else b_rec
                    link_st = (
                        "FRONT_VIEW_ONLY"
                        if (f_rec is not None and b_rec is None)
                        else ("BACK_VIEW_ONLY" if (b_rec is not None and f_rec is None) else "AMBIGUOUS_HUMAN_UNCERTAIN")
                    )
                    rider_res = (
                        self._fuse_single_moto_riders(single_rec, sync_reliability=phi_sync)
                        if (single_rec is not None and (f_rec is None or b_rec is None))
                        else []
                    )
                    stats["single_view_events"] += 1
                    raw_events.append(
                        self._build_event_dict(
                            front_rec=f_rec,
                            back_rec=b_rec,
                            rider_results=rider_res,
                            outcome=oc,
                            score=mr.association_score,
                            reasons=reasons,
                            link_status=link_st,
                            sync_reliability=phi_sync,
                            cross_view_fused=False,
                            quarantined=False,
                        )
                    )

        # Uncovered / single-view projection motorcycles
        for cam_name, cam_dict in (("front", f_motos), ("back", b_motos)):
            for tid, rec in cam_dict.items():
                if (cam_name, tid) in covered:
                    continue
                rider_res = self._fuse_single_moto_riders(rec, sync_reliability=phi_sync)
                stats["single_view_events"] += 1
                raw_events.append(
                    self._build_event_dict(
                        front_rec=rec if cam_name == "front" else None,
                        back_rec=rec if cam_name == "back" else None,
                        rider_results=rider_res,
                        outcome="UNMATCHED",
                        score=None,
                        reasons=(
                            [f"SINGLE_CAMERA_MODE_{norm_mode.upper()}"]
                            if norm_mode != "dual_fused"
                            else ["NO_ASSOCIATION_RESULT_FOR_TRACK"]
                        ),
                        link_status="FRONT_VIEW_ONLY" if cam_name == "front" else "BACK_VIEW_ONLY",
                        sync_reliability=phi_sync,
                        cross_view_fused=False,
                        quarantined=False,
                    )
                )

        raw_events.sort(
            key=lambda e: (
                e["_sort"],
                e["front_track_id"] or "",
                e["back_track_id"] or "",
            )
        )
        events: list[dict[str, Any]] = []
        for idx, ev in enumerate(raw_events, 1):
            ev["global_moto_id"] = f"{pair_id}_moto_{idx:02d}"
            ev.pop("_sort")
            if ev["fusion_metadata"]["has_evidence_conflict_abstain"]:
                stats["evidence_conflict_abstains"] += 1
            for r in ev["riders"]:
                if r["fusion_decision_state"] == FusionDecisionState.LOW_CONFIDENCE_ABSTAIN.value:
                    stats["low_confidence_abstains"] += 1
            events.append(ev)

        if norm_mode == "dual_fused" and batch is not None:
            orphan_res = self.fuse_orphans(
                list(batch.orphan_results),
                f_orphans,
                b_orphans,
                sync_reliability=phi_sync,
            )
        else:
            src_orphans = f_orphans if norm_mode == "front_only" else (b_orphans if norm_mode == "back_only" else {})
            orphan_res = [
                OrphanFusionResult(
                    global_orphan_link_id=None,
                    front_track_id=ev.track_id if ev.camera_id == "front" else None,
                    back_track_id=ev.track_id if ev.camera_id == "back" else None,
                    association_outcome="UNMATCHED",
                    rider_fusion=evaluate_single_view_belief(
                        ev,
                        canonical_slot=0,
                        role="orphan_rider",
                        sync_reliability=1.0,
                        config=self.config,
                    ),
                    implies_confirmed_motorcycle_link=False,
                )
                for ev in sorted(src_orphans.values(), key=lambda x: x.track_id)
            ]

        return PairFusionOutput(
            pair_id=pair_id,
            mode=norm_mode,
            sync_reliability=phi_sync,
            events=events,
            orphan_results=orphan_res,
            stats=stats,
        )


__all__ = [
    "DualCameraFusionEngine",
    "OrphanFusionResult",
    "PairFusionOutput",
    "derive_event_type_from_fused_riders",
    "extract_camera_moto_and_orphan_records",
]
