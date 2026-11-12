"""TASK-04: genuine-observation helmet track-state export (design: ARGON_TASK03).

Synthetic, deterministic; no models, no video.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from src.detection.camera_detector import StructuredDetection
from src.temporal.rider_tracker import Track
from src.tracking.camera_tracker import (
    HelmetEvidenceAccumulator,
    MotoTrack,  # noqa: F401  (import check)
    OrphanRiderTrack,
    Phase10AdapterConfig,
    RiderSubTrack,
    derive_conf_weighted_helmet_state,
    derive_ema_helmet_state,
    derive_majority_helmet_state,
)
from src.tracking.kalman_bytetrack import ByteTrackConfig
from src.tracking.moto_rider_coupler import (
    CoupledRiderObservation,
    OrphanRiderObservation,
)

NEW_KEYS = {
    "track_helmet_state", "track_helmet_state_majority", "track_helmet_state_conf_weighted",
    "track_helmet_state_last_coupled", "track_helmet_state_ema_final",
    "last_coupled_aggregated_confidence", "coupled_confidence_weights",
    "coupled_observations_count", "coupled_helmet_count", "coupled_no_helmet_count",
    "coupled_state_switches", "last_coupled_frame_index", "last_coupled_pts_sec",
}


def _head(cls: str, conf: float, f: int = 0) -> StructuredDetection:
    return StructuredDetection(
        x1=100, y1=100, x2=140, y2=140, confidence=conf,
        class_id=0 if cls == "no_helmet" else 1, class_name=cls, category="head",
        source_model="t", camera_id="front", frame_index=f, pts_timestamp_sec=f / 60.0,
    )


def _coupled(cls, conf, f=0):
    return CoupledRiderObservation(
        head_detection=_head(cls, conf, f), person_detection=None, slot_index=0,
        slot_role="driver", ordering_confident=True, coupling_score=0.9,
        second_best_score=0.1, margin_over_second_best=0.8, geometric_features=None,
    )


def _orphan_obs(cls, conf, f=0):
    return OrphanRiderObservation(
        head_detection=_head(cls, conf, f), person_detection=None,
        rejection_reason="NO_CANDIDATE_MOTORCYCLES", best_candidate_moto_idx=None,
        best_candidate_score=0.0, second_best_score=0.0, candidate_features=None,
    )


def _sub(first=("no_helmet", 0.8), p10=None):
    return RiderSubTrack(
        camera_id="front", parent_moto_id="front_moto_0001", numeric_track_id=1, slot_index=0,
        slot_role="driver", ordering_confident=True, initial_obs=_coupled(*first, f=0),
        frame_index=0, pts_timestamp_sec=0.0, p10_config=p10 or Phase10AdapterConfig(),
    )


def _orphan(first=("no_helmet", 0.8)):
    return OrphanRiderTrack(
        camera_id="front", numeric_id=1, initial_orphan=_orphan_obs(*first, f=0),
        frame_index=0, pts_timestamp_sec=0.0, bt_config=ByteTrackConfig(),
        p10_config=Phase10AdapterConfig(),
    )


def _feed(t, seq, start=1):
    """Feed genuine matched observations on consecutive frames."""
    for i, (cls, conf) in enumerate(seq):
        f = start + i
        if isinstance(t, RiderSubTrack):
            t.update_matched(_coupled(cls, conf, f), f, f / 60.0)
        else:
            t.update_matched(_orphan_obs(cls, conf, f), f, f / 60.0)
    return start + len(seq)


def _gap(t, start, n):
    for f in range(start, start + n):
        t.update_unmatched(f)
    return start + n


# --- Test A: long tail gap -------------------------------------------------
@pytest.mark.parametrize("make", [_sub, _orphan])
def test_A_long_tail_gap_preserves_primary_state(make):
    t = make(("no_helmet", 0.8))
    f = _feed(t, [("no_helmet", 0.8)] * 19)
    _gap(t, f, 50)
    d = t.to_dict()
    assert d["coupled_observations_count"] == 20
    assert d["track_helmet_state"] == "NO_HELMET"
    assert d["track_helmet_state_last_coupled"] == "NO_HELMET"
    # legacy online window flushed (orphans would terminate, but the aggregator itself flushes too)
    assert d["aggregated_confidence"] == {"helmet": 0.0, "no_helmet": 0.0}
    assert d["track_helmet_state_ema_final"] == "UNCERTAIN"


# --- Test B / H: short gap, legacy equivalence ------------------------------
def test_B_H_short_gap_keeps_legacy_ema_and_matches_it():
    t = _sub(("helmet", 0.7))
    f = _feed(t, [("helmet", 0.7)] * 4)
    _gap(t, f, 5)
    d = t.to_dict()
    ac = t.phase10_track.aggregator.get_aggregated_confidence()
    assert d["aggregated_confidence"] == ac and max(ac.values()) >= 0.5
    assert d["track_helmet_state_ema_final"] == d["track_helmet_state"] == "HELMET"
    assert d["last_coupled_aggregated_confidence"] == ac  # no flush yet -> identical


# --- Test C: conflicting observations, strict majority, tie ------------------
def test_C_majority_tie_and_weighted_separate():
    t = _sub(("helmet", 0.63))
    _feed(t, [("no_helmet", 0.27)])
    d = t.to_dict()
    assert d["coupled_helmet_count"] == d["coupled_no_helmet_count"] == 1
    assert d["track_helmet_state"] == d["track_helmet_state_majority"] == "UNCERTAIN"
    assert d["track_helmet_state_conf_weighted"] == "HELMET"  # separate diagnostic only
    assert d["coupled_state_switches"] == 1
    t2 = _sub(("no_helmet", 0.9))
    _feed(t2, [("helmet", 0.3), ("helmet", 0.3)])
    d2 = t2.to_dict()
    assert d2["track_helmet_state"] == "HELMET"  # 2 vs 1 by count
    assert d2["track_helmet_state_conf_weighted"] == "NO_HELMET"  # 0.9 > 0.6


# --- Test D: state transition -------------------------------------------------
def test_D_transition_latest_vs_historical_majority():
    t = _sub(("no_helmet", 0.8))
    f = _feed(t, [("no_helmet", 0.8)] * 5)
    _feed(t, [("helmet", 0.9)] * 3, start=f)
    d = t.to_dict()
    assert d["track_helmet_state_majority"] == "NO_HELMET"  # historical dominant
    assert d["track_helmet_state_last_coupled"] == "HELMET"  # latest regime (EMA)
    assert d["coupled_state_switches"] == 1
    assert d["last_coupled_frame_index"] == 8


# --- Test E: no valid observation ---------------------------------------------
def test_E_zero_observations_all_uncertain():
    acc = HelmetEvidenceAccumulator()
    out = acc.export({"helmet": 0.0, "no_helmet": 0.0})
    for k in ("track_helmet_state", "track_helmet_state_majority", "track_helmet_state_conf_weighted",
              "track_helmet_state_last_coupled", "track_helmet_state_ema_final"):
        assert out[k] == "UNCERTAIN"
    assert out["coupled_observations_count"] == 0
    assert out["last_coupled_frame_index"] is None and out["last_coupled_pts_sec"] is None
    # non helmet classes never create evidence
    assert acc.record("person", 0.9, 1, 0.1, {"helmet": 1.0, "no_helmet": 0.0}) is False


# --- Test F: stale Kalman / unmatched frames add nothing -----------------------
@pytest.mark.parametrize("make", [_sub, _orphan])
def test_F_unmatched_frames_do_not_add_evidence(make):
    t = make(("helmet", 0.6))
    before = t.helmet_evidence.export(t.phase10_track.aggregator.get_aggregated_confidence())
    _gap(t, 1, 7)  # < orphan termination so both types stay legal
    t.predict(8) if make is _orphan else t.predict()
    after = t.helmet_evidence.export(t.phase10_track.aggregator.get_aggregated_confidence())
    for k in ("coupled_observations_count", "coupled_helmet_count", "coupled_no_helmet_count",
              "coupled_confidence_weights", "last_coupled_frame_index", "track_helmet_state"):
        assert before[k] == after[k]


# --- Test G: parent outlives rider (subtrack never terminates; evidence persists)
def test_G_parent_outlives_rider_keeps_evidence():
    t = _sub(("no_helmet", 0.7))
    f = _feed(t, [("no_helmet", 0.7)] * 2)
    _gap(t, f, 300)  # parent keeps stepping the subtrack
    assert t.to_dict()["coupled_observations_count"] == 3
    assert t.track_helmet_state == "NO_HELMET"


# --- Additional ----------------------------------------------------------------
def test_repeated_update_protection():
    acc = HelmetEvidenceAccumulator()
    ac = {"helmet": 0.0, "no_helmet": 1.0}
    assert acc.record("no_helmet", 0.9, 5, 0.1, ac)
    assert not acc.record("no_helmet", 0.9, 5, 0.1, ac)  # same frame
    assert not acc.record("no_helmet", 0.9, 4, 0.1, ac)  # earlier frame
    assert acc.observations == 1


def test_missing_timestamp_and_single_observation():
    acc = HelmetEvidenceAccumulator()
    acc.record("helmet", 0.4, 3, None, {"helmet": 1.0, "no_helmet": 0.0})
    out = acc.export({"helmet": 0.0, "no_helmet": 0.0})
    assert out["last_coupled_pts_sec"] is None
    assert out["track_helmet_state"] == "HELMET" and out["coupled_observations_count"] == 1


def test_weighted_math_and_ties():
    assert derive_conf_weighted_helmet_state(0.5, 0.5) == "UNCERTAIN"
    assert derive_conf_weighted_helmet_state(0.0, 0.0) == "UNCERTAIN"
    assert derive_conf_weighted_helmet_state(1.2, 0.3) == "HELMET"
    assert derive_majority_helmet_state(2, 2) == "UNCERTAIN"
    assert derive_ema_helmet_state({"helmet": 0.5, "no_helmet": 0.5}) == "UNCERTAIN"
    assert derive_ema_helmet_state({"helmet": 0.5, "no_helmet": 0.5}, tie_break_helmet=True) == "HELMET"
    assert derive_ema_helmet_state({"helmet": 0.0, "no_helmet": 0.0}) == "UNCERTAIN"


def test_orphan_and_subtrack_parity_and_schema():
    seq = [("no_helmet", 0.5), ("helmet", 0.9), ("no_helmet", 0.4), ("no_helmet", 0.6)]
    s, o = _sub(("helmet", 0.5)), _orphan(("helmet", 0.5))
    _feed(s, seq), _feed(o, seq)
    ds, do = s.to_dict(), o.to_dict()
    assert NEW_KEYS <= set(ds) and NEW_KEYS <= set(do)
    for k in NEW_KEYS:
        assert ds[k] == do[k], k
    json.dumps(ds), json.dumps(do)  # serializable
    assert ds["coupled_observations_count"] == len(s.pts_history) == len(s.observation_log) == 5
    assert ds["coupled_helmet_count"] + ds["coupled_no_helmet_count"] == 5


def test_legacy_keys_preserved():
    d = _sub().to_dict()
    for k in ("subtrack_id", "aggregated_confidence", "alerts_emitted", "temporal_state", "hits"):
        assert k in d
    for k in ("orphan_track_id", "aggregated_confidence", "alerts_emitted", "temporal_state"):
        assert k in _orphan().to_dict()


def test_ema_alpha_is_forwarded():
    from src.temporal.rider_tracker import BoundingBox
    b = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=1, class_name="helmet")
    t = Track(1, b, 0, aggregator_method="ema", ema_alpha=0.8)
    assert t.aggregator.alpha == pytest.approx(0.8)
    assert Track(1, b, 0, aggregator_method="ema").aggregator.alpha == pytest.approx(0.35)
    s = _sub(p10=Phase10AdapterConfig(ema_alpha=0.6))
    assert s.phase10_track.aggregator.alpha == pytest.approx(0.6)
    assert _orphan().phase10_track.aggregator.alpha == pytest.approx(0.35)


# --- Test I: Stage J artifact fallback (no inference) -------------------------
def test_I_stage_j_fallback_reproduces_legacy_and_design_numbers():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, "results/phase12_stagej/pair_001")) or not os.path.isdir(
        os.path.join(root, "data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited")
    ):
        pytest.skip("Stage J artifacts / frozen GT not present")
    sys.path.insert(0, os.path.join(root, "scripts/reproducibility"))
    import track_state_redesign as tsr

    res = tsr.evaluate()
    res.pop("_rows")
    allg = res["groups"]["all"]
    assert res["source_mode"] == "pts_history_fallback"
    assert allg["ema_final"]["correct"] == 134 and allg["ema_final"]["abstentions"] == 73  # Stage K legacy
    assert allg["primary"]["abstentions"] == 0
    assert allg["primary"]["total_eligible_gt_tracks"] == 209
