"""Tests for TASK-11 dual-camera evidence fusion (`src/fusion/`)."""
from __future__ import annotations

from types import SimpleNamespace

from src.association.tracklet_models import (
    AssociationOutcome,
    CrossCameraAssociationBatchResult,
    MotorcycleAssociationResult,
    OrphanAssociationResult,
    RiderAssociationResult,
    TimestampQualityRecord,
)
from src.fusion import (
    DualCameraFusionEngine,
    FusionConfig,
    FusionDecisionState,
    ViewRiderEvidence,
    compute_contradiction_metric,
    compute_sync_reliability,
    evaluate_single_view_belief,
    fuse_rider_beliefs,
)
from src.pipeline.dual_camera_pipeline import derive_pair_events
from src.temporal.state_machine import TemporalState
from src.tracking.camera_tracker import HelmetEvidenceAccumulator


class _StubAlert:
    def __init__(self, alert_id: str) -> None:
        self.alert_id = alert_id


class _StubSubTrack:
    def __init__(
        self,
        *,
        subtrack_id: str,
        slot_index: int,
        slot_role: str,
        consensus_slot_role: str | None = None,
        ordering_confident: bool = True,
        helmet_counts: tuple[int, int] = (0, 0),
        conf_sums: tuple[float, float] = (0.0, 0.0),
        state_switches: int = 0,
        temporal_state: TemporalState = TemporalState.UNKNOWN,
        alerts: tuple[str, ...] = (),
        bbox_wh: tuple[float, float] = (48.0, 48.0),
    ) -> None:
        self.subtrack_id = subtrack_id
        self.slot_index = slot_index
        self.slot_role = slot_role
        self.consensus_slot_role = consensus_slot_role or slot_role
        self.ordering_confident = ordering_confident
        self.temporal_state = temporal_state
        self.alerts = [_StubAlert(aid) for aid in alerts]
        self.aggregated_confidence = {"helmet": 0.8 if helmet_counts[0] > helmet_counts[1] else 0.1, "no_helmet": 0.8 if helmet_counts[1] > helmet_counts[0] else 0.1}
        self.helmet_evidence = HelmetEvidenceAccumulator()
        h_n, nh_n = helmet_counts
        h_avg = (conf_sums[0] / h_n) if h_n > 0 and conf_sums[0] > 0 else 0.85
        nh_avg = (conf_sums[1] / nh_n) if nh_n > 0 and conf_sums[1] > 0 else 0.85
        fidx = 1
        pts_hist = []
        w, h = bbox_wh
        bbox = (500.0, 400.0, 500.0 + w, 400.0 + h)
        for _ in range(h_n):
            self.helmet_evidence.record("helmet", h_avg, fidx, fidx / 60.0, {"helmet": h_avg, "no_helmet": 1.0 - h_avg})
            pts_hist.append((fidx, fidx / 60.0, bbox, "helmet", h_avg))
            fidx += 1
        for _ in range(nh_n):
            self.helmet_evidence.record("no_helmet", nh_avg, fidx, fidx / 60.0, {"helmet": 1.0 - nh_avg, "no_helmet": nh_avg})
            pts_hist.append((fidx, fidx / 60.0, bbox, "no_helmet", nh_avg))
            fidx += 1
        self.helmet_evidence.state_switches = state_switches
        self.pts_history = pts_hist

    @property
    def track_helmet_state(self) -> str:
        return self.helmet_evidence.export(self.aggregated_confidence)["track_helmet_state"]


def _make_stub_moto(
    moto_track_id: str,
    subs: list[_StubSubTrack],
    *,
    t0: float = 1.0,
    t1: float = 4.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        moto_track_id=moto_track_id,
        first_pts_sec=t0,
        last_pts_sec=t1,
        rider_subtracks={s.slot_index: s for s in subs},
    )


def _make_stub_cam_out(
    cam: str,
    motos: list[SimpleNamespace],
    *,
    shift: float = 0.0,
    orphans: list[SimpleNamespace] | None = None,
    frame_wh: tuple[int, int] = (3840, 2160),
) -> SimpleNamespace:
    pipe = SimpleNamespace(
        terminated_moto_tracks=motos,
        active_moto_tracks={},
        terminated_orphan_tracks=orphans or [],
        active_orphan_tracks={},
        frame_dimensions=frame_wh,
        exported_orphan_tracks=lambda: list(orphans or []),
    )
    return SimpleNamespace(camera_id=cam, canonical_shift_sec=shift, pipeline=pipe)


def _empty_batch(pair_id: str, moto_results: list[MotorcycleAssociationResult], orphan_results: list[OrphanAssociationResult] | None = None) -> CrossCameraAssociationBatchResult:
    tq = TimestampQualityRecord(
        pair_id=pair_id,
        global_offset_sec=0.0,
        epsilon_sync_ms=5.0,
        mean_delta_pts_front_ms=2.0,
        mean_delta_pts_back_ms=2.0,
    )
    return CrossCameraAssociationBatchResult(
        schema_version="test",
        pair_id=pair_id,
        timestamp_quality=tq,
        front_motorcycle_tracklets=[],
        back_motorcycle_tracklets=[],
        front_orphan_tracklets=[],
        back_orphan_tracklets=[],
        candidate_pair_scores=[],
        motorcycle_results=moto_results,
        orphan_results=orphan_results or [],
        outcome_counts={},
        rider_outcome_counts={},
        orphan_outcome_counts={},
        runtime_ms={},
    )


def test_k1_opposing_override_eliminated_by_beta_shrinkage_and_log_odds() -> None:
    """Dense Front HELMET (174H/50NH) must override K=1 noisy Back NO_HELMET (Stage J pair_012 fix)."""
    f_ev = ViewRiderEvidence(
        camera_id="front",
        track_id="front_moto_0004_rider_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=174,
        no_helmet_count=50,
        helmet_conf_sum=140.0,
        no_helmet_conf_sum=30.0,
        track_helmet_state="HELMET",
        mean_head_area_px=2500.0,
    )
    b_ev = ViewRiderEvidence(
        camera_id="back",
        track_id="back_moto_0002_rider_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=0,
        no_helmet_count=1,
        helmet_conf_sum=0.0,
        no_helmet_conf_sum=0.90,
        track_helmet_state="NO_HELMET",
        mean_head_area_px=1200.0,
    )
    res = fuse_rider_beliefs(f_ev, b_ev, canonical_slot=0, role="driver", association_score=0.95)
    assert res.helmet_state == "HELMET"
    assert res.fusion_decision_state == FusionDecisionState.VERIFIED_DUAL_COMPLIANT
    assert res.contradiction_k < 0.35
    assert not res.abstained


def test_strong_opposing_views_trigger_evidence_conflict_abstain() -> None:
    """Genuine multi-frame contradiction (e.g., pair_015 false link) triggers EVIDENCE_CONFLICT_ABSTAIN."""
    f_ev = ViewRiderEvidence(
        camera_id="front",
        track_id="f_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=0,
        no_helmet_count=2,
        helmet_conf_sum=0.0,
        no_helmet_conf_sum=1.80,
        track_helmet_state="NO_HELMET",
    )
    b_ev = ViewRiderEvidence(
        camera_id="back",
        track_id="b_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=51,
        no_helmet_count=7,
        helmet_conf_sum=42.0,
        no_helmet_conf_sum=3.5,
        track_helmet_state="HELMET",
    )
    res = fuse_rider_beliefs(f_ev, b_ev, canonical_slot=0, role="driver", association_score=0.897)
    assert res.contradiction_k >= 0.35
    assert res.fusion_decision_state == FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN
    assert res.helmet_state == "UNCERTAIN"
    assert res.abstained


def test_weak_single_frame_abstains_while_strong_decides() -> None:
    weak = ViewRiderEvidence(
        camera_id="front",
        track_id="w_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=0,
        no_helmet_count=1,
        helmet_conf_sum=0.0,
        no_helmet_conf_sum=0.25,
        track_helmet_state="NO_HELMET",
    )
    strong = ViewRiderEvidence(
        camera_id="front",
        track_id="s_s0",
        slot_index=0,
        slot_role="driver",
        helmet_count=0,
        no_helmet_count=1,
        helmet_conf_sum=0.0,
        no_helmet_conf_sum=0.85,
        track_helmet_state="NO_HELMET",
    )
    w_res = evaluate_single_view_belief(weak, canonical_slot=0)
    s_res = evaluate_single_view_belief(strong, canonical_slot=0)
    assert w_res.fusion_decision_state == FusionDecisionState.LOW_CONFIDENCE_ABSTAIN
    assert w_res.helmet_state == "UNCERTAIN"
    assert s_res.fusion_decision_state == FusionDecisionState.SINGLE_VIEW_VIOLATION_FRONT
    assert s_res.helmet_state == "NO_HELMET"


def test_sync_reliability_and_conflict_metric_bounds() -> None:
    assert compute_sync_reliability(10.0, 2.0) == 1.0
    near_rel = compute_sync_reliability(38.0, 5.0)
    assert 0.80 <= near_rel < 1.0
    assert compute_contradiction_metric(0.8, 0.9) == 0.0
    assert abs(compute_contradiction_metric(0.85, 0.15) - 0.70) < 1e-6


def test_rider_association_cross_slot_pairing_and_ambiguous_quarantine() -> None:
    """Linked riders align via RiderAssociationResult IDs (f s2 <-> b s0), while AMBIGUOUS riders quarantine."""
    f_m = _make_stub_moto(
        "front_moto_0002",
        [
            _StubSubTrack(subtrack_id="front_moto_0002_rider_s2", slot_index=2, slot_role="driver", helmet_counts=(0, 40)),
            _StubSubTrack(subtrack_id="front_moto_0002_rider_s0", slot_index=0, slot_role="pillion_1", helmet_counts=(1, 1)),
        ],
    )
    b_m = _make_stub_moto(
        "back_moto_0002",
        [
            _StubSubTrack(subtrack_id="back_moto_0002_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(0, 35)),
            _StubSubTrack(subtrack_id="back_moto_0002_rider_s1", slot_index=1, slot_role="pillion_1", helmet_counts=(0, 30)),
        ],
    )
    ra_drv = RiderAssociationResult(
        schema_version="1",
        parent_global_motorcycle_id="pair_019_g01",
        front_moto_track_id="front_moto_0002",
        back_moto_track_id="back_moto_0002",
        front_subtrack_id="front_moto_0002_rider_s2",
        back_subtrack_id="back_moto_0002_rider_s0",
        front_slot_role="driver",
        back_slot_role="driver",
        outcome=AssociationOutcome.LINKED,
        association_score=0.92,
        second_best_score=0.30,
        ambiguity_margin=0.62,
        decision_reasons=["ROLE_MATCH"],
        cue_breakdown={},
    )
    ra_pil = RiderAssociationResult(
        schema_version="1",
        parent_global_motorcycle_id="pair_019_g01",
        front_moto_track_id="front_moto_0002",
        back_moto_track_id="back_moto_0002",
        front_subtrack_id="front_moto_0002_rider_s0",
        back_subtrack_id="back_moto_0002_rider_s1",
        front_slot_role="pillion_1",
        back_slot_role="pillion_1",
        outcome=AssociationOutcome.AMBIGUOUS_ABSTAIN,
        association_score=0.71,
        second_best_score=0.69,
        ambiguity_margin=0.02,
        decision_reasons=["AMBIGUOUS_SWAP"],
        cue_breakdown={},
    )
    mr = MotorcycleAssociationResult(
        schema_version="1",
        global_motorcycle_id="pair_019_g01",
        front_moto_track_id="front_moto_0002",
        back_moto_track_id="back_moto_0002",
        outcome=AssociationOutcome.LINKED,
        association_score=0.86,
        second_best_score=0.72,
        ambiguity_margin=0.14,
        swap_margin_2x2=None,
        decision_reasons=["LINKED"],
        rider_associations=[ra_drv, ra_pil],
    )
    f_out = _make_stub_cam_out("front", [f_m])
    b_out = _make_stub_cam_out("back", [b_m])
    batch = _empty_batch("pair_019", [mr])

    events = derive_pair_events("pair_019", f_out, b_out, batch, fusion_mode="dual_fused")
    assert len(events) == 1
    ev = events[0]
    assert ev["violation_event_type"] == "BOTH_NO_HELMET"
    drv = [r for r in ev["riders"] if r["role"] == "driver"][0]
    assert drv["fusion_decision_state"] == FusionDecisionState.VERIFIED_DUAL_VIOLATION.value
    assert drv["cross_view_fused"] is True
    # Ambiguous rider association must NOT cross-fuse front s0 and back s1 together
    pil_entries = [r for r in ev["riders"] if r["role"] == "pillion_1"]
    assert len(pil_entries) == 2
    assert all(not r["cross_view_fused"] for r in pil_entries)
    assert all(r["quarantined"] for r in pil_entries)


def test_ambiguous_motorcycle_link_quarantined_split_and_schema_preserve() -> None:
    """AMBIGUOUS_ABSTAIN links must never cross-fuse Front rider with Back bicycle."""
    f_m = _make_stub_moto(
        "front_moto_0001",
        [_StubSubTrack(subtrack_id="front_moto_0001_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(0, 80))],
    )
    b_m = _make_stub_moto("back_moto_0003", [])
    mr = MotorcycleAssociationResult(
        schema_version="1",
        global_motorcycle_id=None,
        front_moto_track_id="front_moto_0001",
        back_moto_track_id="back_moto_0003",
        outcome=AssociationOutcome.AMBIGUOUS_ABSTAIN,
        association_score=0.74,
        second_best_score=0.69,
        ambiguity_margin=0.05,
        swap_margin_2x2=None,
        decision_reasons=["MARGIN_BELOW_0.10"],
    )
    f_out = _make_stub_cam_out("front", [f_m])
    b_out = _make_stub_cam_out("back", [b_m])
    batch = _empty_batch("pair_020", [mr])

    # 1. Quarantine split mode: decouples Front motorcycle and Back bicycle into distinct single-view events
    split_events = derive_pair_events(
        "pair_020",
        f_out,
        b_out,
        batch,
        fusion_mode="dual_fused",
        fusion_config=FusionConfig(split_ambiguous_moto_to_single_view_events=True),
    )
    assert len(split_events) == 2
    assert {e["front_track_id"] or e["back_track_id"] for e in split_events} == {"front_moto_0001", "back_moto_0003"}
    fev = [e for e in split_events if e["front_track_id"] == "front_moto_0001"][0]
    assert fev["violation_event_type"] == "DRIVER_NO_HELMET"
    assert fev["fusion_metadata"]["cross_view_fused"] is False
    assert fev["plate_ground_truth"]["cross_camera_plate_link_permitted"] is False

    # 2. Schema-preserving mode: keeps tuple key while blocking cross-view fusion and citation
    pres_events = derive_pair_events(
        "pair_020",
        f_out,
        b_out,
        batch,
        fusion_mode="dual_fused",
        fusion_config=FusionConfig(split_ambiguous_moto_to_single_view_events=False, ambiguous_link_abstain_citation=True),
    )
    assert len(pres_events) == 1
    assert pres_events[0]["violation_event_type"] == "INSUFFICIENT_VISUAL_EVIDENCE"
    assert pres_events[0]["fusion_metadata"]["quarantined"] is True
    assert pres_events[0]["fusion_metadata"]["quarantined_single_view_decisions"]["front"]["event_type"] == "DRIVER_NO_HELMET"


def test_orphan_fusion_and_single_camera_modes_share_unified_engine() -> None:
    """Orphan links fuse safely without claiming confirmed motorcycle membership; single-camera mode isolates views."""
    f_m = _make_stub_moto(
        "front_moto_0001",
        [_StubSubTrack(subtrack_id="front_moto_0001_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(0, 30))],
    )
    b_m = _make_stub_moto(
        "back_moto_0001",
        [_StubSubTrack(subtrack_id="back_moto_0001_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(25, 0))],
    )
    mr = MotorcycleAssociationResult(
        schema_version="1",
        global_motorcycle_id=None,
        front_moto_track_id="front_moto_0001",
        back_moto_track_id=None,
        outcome=AssociationOutcome.UNMATCHED,
        association_score=0.0,
        second_best_score=0.0,
        ambiguity_margin=0.0,
        swap_margin_2x2=None,
        decision_reasons=["UNMATCHED"],
    )
    orp_res = OrphanAssociationResult(
        schema_version="1",
        global_orphan_link_id="pair_009_orp_01",
        association_level="ORPHAN_LEVEL_LINK",
        front_track_id="front_orphan_0001",
        back_track_id="back_orphan_0001",
        front_track_kind="orphan_rider",
        back_track_kind="orphan_rider",
        outcome=AssociationOutcome.LINKED,
        association_score=0.88,
        second_best_score=0.30,
        ambiguity_margin=0.58,
        decision_reasons=["LINKED"],
        cue_breakdown={},
        implies_confirmed_motorcycle_link=False,
    )
    f_orp = SimpleNamespace(
        orphan_track_id="front_orphan_0001",
        first_pts_sec=1.0,
        last_pts_sec=2.0,
        alerts=[],
        pts_history=[],
        phase10_track=SimpleNamespace(state=TemporalState.NO_HELMET_CONFIRMED, aggregator=SimpleNamespace(get_aggregated_confidence=lambda: {"helmet": 0.05, "no_helmet": 0.95})),
        helmet_evidence=_StubSubTrack(subtrack_id="fo", slot_index=0, slot_role="driver", helmet_counts=(0, 12)).helmet_evidence,
    )
    b_orp = SimpleNamespace(
        orphan_track_id="back_orphan_0001",
        first_pts_sec=3.0,
        last_pts_sec=4.0,
        alerts=[],
        pts_history=[],
        phase10_track=SimpleNamespace(state=TemporalState.NO_HELMET_CONFIRMED, aggregator=SimpleNamespace(get_aggregated_confidence=lambda: {"helmet": 0.05, "no_helmet": 0.95})),
        helmet_evidence=_StubSubTrack(subtrack_id="bo", slot_index=0, slot_role="driver", helmet_counts=(0, 10)).helmet_evidence,
    )
    f_out = _make_stub_cam_out("front", [f_m], orphans=[f_orp])
    b_out = _make_stub_cam_out("back", [b_m], orphans=[b_orp])
    batch = _empty_batch("pair_009", [mr], [orp_res])

    engine = DualCameraFusionEngine()
    dual = engine.fuse_pair("pair_009", f_out, b_out, batch, mode="dual_fused")
    assert len(dual.orphan_results) == 1
    assert dual.orphan_results[0].rider_fusion.helmet_state == "NO_HELMET"
    assert dual.orphan_results[0].implies_confirmed_motorcycle_link is False

    front_only = engine.fuse_pair("pair_009", f_out, b_out, batch, mode="front_only")
    assert len(front_only.events) == 1
    assert front_only.events[0]["front_track_id"] == "front_moto_0001"
    assert front_only.events[0]["back_track_id"] is None
    assert front_only.events[0]["violation_event_type"] == "DRIVER_NO_HELMET"

    back_only = engine.fuse_pair("pair_009", f_out, b_out, batch, mode="back_only")
    assert len(back_only.events) == 1
    assert back_only.events[0]["back_track_id"] == "back_moto_0001"
    assert back_only.events[0]["front_track_id"] is None
    assert back_only.events[0]["violation_event_type"] == "NO_VIOLATION"

    # Legacy mode remains intact
    leg = derive_pair_events("pair_009", f_out, b_out, batch, fusion_mode="legacy")
    assert len(leg) == 2
    assert "fusion_metadata" not in leg[0]


def test_unmatched_opposing_drivers_on_linked_moto_conflict_abstain() -> None:
    """When RiderAssociation rejects pairing two opposing drivers on a LINKED moto (pair_015 pattern),
    fusion must abstain via EVIDENCE_CONFLICT_ABSTAIN rather than emitting two contradicting drivers."""
    f_m = _make_stub_moto(
        "front_moto_0003",
        [_StubSubTrack(subtrack_id="front_moto_0003_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(0, 2), conf_sums=(0.0, 0.56))],
    )
    b_m = _make_stub_moto(
        "back_moto_0001",
        [_StubSubTrack(subtrack_id="back_moto_0001_rider_s0", slot_index=0, slot_role="driver", helmet_counts=(51, 7), conf_sums=(42.0, 3.5))],
    )
    ra_rej = RiderAssociationResult(
        schema_version="1",
        parent_global_motorcycle_id="pair_015_g01",
        front_moto_track_id="front_moto_0003",
        back_moto_track_id="back_moto_0001",
        front_subtrack_id="front_moto_0003_rider_s0",
        back_subtrack_id="back_moto_0001_rider_s0",
        front_slot_role="driver",
        back_slot_role="driver",
        outcome=AssociationOutcome.UNMATCHED,
        association_score=0.45,
        second_best_score=0.0,
        ambiguity_margin=0.45,
        decision_reasons=["TEMPORAL_MISMATCH"],
        cue_breakdown={},
    )
    mr = MotorcycleAssociationResult(
        schema_version="1",
        global_motorcycle_id="pair_015_g01",
        front_moto_track_id="front_moto_0003",
        back_moto_track_id="back_moto_0001",
        outcome=AssociationOutcome.LINKED,
        association_score=0.8967,
        second_best_score=0.0,
        ambiguity_margin=0.8967,
        swap_margin_2x2=None,
        decision_reasons=["LINKED"],
        rider_associations=[ra_rej],
    )
    batch = _empty_batch("pair_015", [mr])
    events = derive_pair_events(
        "pair_015",
        _make_stub_cam_out("front", [f_m]),
        _make_stub_cam_out("back", [b_m]),
        batch,
        fusion_mode="dual_fused",
    )
    assert len(events) == 1
    ev = events[0]
    assert ev["rider_count"] == 1
    assert ev["violation_event_type"] == "INSUFFICIENT_VISUAL_EVIDENCE"
    assert ev["fusion_metadata"]["has_evidence_conflict_abstain"] is True
    assert ev["riders"][0]["fusion_decision_state"] == FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN.value
