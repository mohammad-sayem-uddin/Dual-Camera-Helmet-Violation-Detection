"""Phase 12 Pipeline (Stage G) — Stage H Real-Video Test Suite.

Real-video half of the Stage H matrix (design §7.13): pairs 001, 013, 014,
015, 019 exercised through the REAL Phase 12C reader / FramePTSIndex and the
pipeline's SyncContext — PTS-table integrity against the Phase 12C forensic
frame counts, three-way offset agreement (metadata == pair_manifest ==
reader), overlap windows, canonical-time spot checks, pair_015's 29.97 fps
back-camera unique-frame counting on the 60 Hz grid, and the forensic VFR
classifications.

Bounded by design: index/PTS-level checks plus a handful of decoded frames
per pair (get_synchronized_pair spot checks, one 24-tick scan on pair_015).
NO model inference is run in Stage H (that is Stage J); run_pair is never
invoked on real videos here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import src.pipeline.dual_camera_pipeline as dcp
from src.synchronization.synchronized_reader import SynchronizedVideoReader

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "dual_camera_test_v1"
METADATA_PATH = DATA_ROOT / "metadata" / "synchronization_metadata.json"
MANIFEST_CSV = DATA_ROOT / "metadata" / "pair_manifest.csv"
COVERAGE_CSV = DATA_ROOT / "annotations" / "frame_coverage_manifest.csv"

PAIRS = ["pair_001", "pair_013", "pair_014", "pair_015", "pair_019"]

# Phase 11/12C reference values (pair_manifest.csv + synchronization_metadata.json).
EXPECTED = {
    "pair_001": {"offset": -0.205, "epsilon": 3.0, "drift": 0.0, "quality": "HIGH",
                 "overlap": 3.102},
    "pair_013": {"offset": 6.695, "epsilon": 3.0, "drift": 0.0, "quality": "HIGH",
                 "overlap": 6.784},
    "pair_014": {"offset": -1.82, "epsilon": 3.0, "drift": 0.0, "quality": "HIGH",
                 "overlap": 2.66},
    "pair_015": {"offset": -1.47, "epsilon": 37.0, "drift": 74.0, "quality": "HIGH",
                 "overlap": 4.247},
    "pair_019": {"offset": -2.380, "epsilon": 8.0, "drift": -16.0, "quality": "MEDIUM",
                 "overlap": 3.86},
}


def _config() -> dcp.PipelineConfig:
    return dcp.PipelineConfig(
        data_root=DATA_ROOT,
        metadata_path=METADATA_PATH,
        tracking_config_path=REPO / "configs" / "tracking_config.yaml",
        association_config_path=REPO / "configs" / "cross_camera_association_config.yaml",
        output_root=REPO / "work" / "stageH" / "real_video_no_output",
        device="cpu",
    )


@pytest.fixture(scope="module")
def forensic_counts() -> dict[tuple[str, str], int]:
    """Per-(pair, camera) decoded frame counts from the Phase 12C forensic
    coverage manifest (15,241 frames total across the corpus)."""
    counts: dict[tuple[str, str], int] = {}
    with open(COVERAGE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["pair_id"], row["camera"])
            counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    with open(MANIFEST_CSV, newline="") as f:
        return {row["pair_id"]: row for row in csv.DictReader(f)}


@pytest.fixture(scope="module")
def metadata_items() -> dict[str, dict]:
    items = json.loads(METADATA_PATH.read_text())
    return {it["pair_id"]: it for it in items}


@pytest.fixture(scope="module")
def contexts():
    cfg = _config()
    ctxs = {pid: dcp.SyncContext.from_reader(pid, cfg) for pid in PAIRS}
    yield ctxs
    for ctx in ctxs.values():
        ctx.close()


# --------------------------------------------------------------------------
# PTS-table integrity vs Phase 12C forensic counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair_id", PAIRS)
def test_real_pts_table_integrity(pair_id, contexts, forensic_counts):
    ctx = contexts[pair_id]
    for cam in ("front", "back"):
        idx = ctx.index_for(cam)
        assert idx.frame_count == forensic_counts[(pair_id, cam)], \
            f"{pair_id}/{cam}: PTS table count != Phase 12C forensic count"
        assert idx.pts_norm_sec[0] == 0.0
        # Presentation order is strictly increasing for these ten streams
        # (no duplicate packet timestamps in the forensic corpus).
        assert np.all(np.diff(idx.pts_norm_sec) > 0), \
            f"{pair_id}/{cam}: PTS table not strictly increasing"
        assert idx.frame_count > 100


# --------------------------------------------------------------------------
# Three-way agreement: pinned metadata == pair_manifest == reader/pipeline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair_id", PAIRS)
def test_real_offset_three_way_agreement(pair_id, contexts, manifest):
    ctx = contexts[pair_id]
    exp = EXPECTED[pair_id]
    row = manifest[pair_id]
    assert ctx.offset_sec == pytest.approx(float(row["temporal_offset_sec"]), abs=1e-9)
    assert ctx.offset_sec == pytest.approx(exp["offset"], abs=1e-9)
    assert ctx.epsilon_sync_ms == pytest.approx(float(row["residual_error_ms"]), abs=1e-9)
    assert ctx.epsilon_sync_ms == pytest.approx(exp["epsilon"], abs=1e-9)
    assert ctx.drift_ms == pytest.approx(float(row["drift_ms"]), abs=1e-9)
    assert ctx.quality == row["quality"] == exp["quality"]
    # Canonical shift is the offset for front, zero for back (D1).
    assert ctx.canonical_shift_sec("front") == pytest.approx(exp["offset"], abs=1e-9)
    assert ctx.canonical_shift_sec("back") == 0.0


# --------------------------------------------------------------------------
# Overlap windows vs manifest + the sanctioned formulas
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair_id", PAIRS)
def test_real_overlap_windows(pair_id, contexts, manifest, metadata_items):
    ctx = contexts[pair_id]
    info = ctx.sync_info
    offset = EXPECTED[pair_id]["offset"]
    # Sanctioned formulas (§2.1): the metadata window [max(0, offset),
    # min(back_dur, offset + front_dur)] is then TIGHTENED by the reader
    # (synchronized_reader.py __init__) against the PTS-index durations
    # (VideoReader.duration_sec == FramePTSIndex.duration_norm_sec), which
    # can only shrink the end. Expected values below reproduce both steps.
    item = metadata_items[pair_id]
    exp_start = max(0.0, offset)
    meta_end = min(float(item["back_duration"]),
                   offset + float(item["front_duration"]))
    true_end = min(float(ctx.back_index.duration_norm_sec),
                   offset + float(ctx.front_index.duration_norm_sec))
    exp_end = max(exp_start, min(meta_end, true_end))
    assert info.overlap_start_back_sec == pytest.approx(exp_start, abs=0.01)
    assert info.overlap_end_back_sec == pytest.approx(exp_end, abs=0.01)
    # The reader recomputes duration from the tightened window (exact), and
    # it tracks the manifest value to within one frame interval (<= 0.1 s).
    assert info.overlap_duration_sec == pytest.approx(
        info.overlap_end_back_sec - info.overlap_start_back_sec, abs=1e-6)
    assert info.overlap_duration_sec == pytest.approx(
        float(manifest[pair_id]["overlap_duration_sec"]), abs=0.1)


# --------------------------------------------------------------------------
# Canonical-time spot checks at the overlap midpoint (bounded decode)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair_id", PAIRS)
def test_real_canonical_spot_checks(pair_id, contexts):
    ctx = contexts[pair_id]
    info = ctx.sync_info
    t_mid = (info.overlap_start_back_sec + info.overlap_end_back_sec) / 2.0
    fb, ff, status = ctx.reader.get_synchronized_pair(t_mid)
    assert fb is not None and ff is not None
    assert status["is_in_overlap"] is True
    assert status["valid_pair"] is True

    offset = ctx.offset_sec
    exp_front_idx = ctx.front_index.nearest_frame(t_mid - offset).frame_index
    exp_back_idx = ctx.back_index.nearest_frame(t_mid).frame_index
    assert status["frame_idx_front"] == exp_front_idx
    assert status["frame_idx_back"] == exp_back_idx
    # Canonicalization: the chosen front frame's canonical time is its
    # pts_norm + offset; the index-equality assertions above already prove
    # the reader picked it as the nearest frame to t_mid on that axis.
    front_actual = float(ctx.front_index.pts_norm_sec[exp_front_idx])
    assert front_actual == pytest.approx(status["t_front_actual_pts_sec"], abs=1e-6)
    # Reader skew definition stays a small presentation-level quantity.
    assert status["inter_camera_pts_skew_ms"] < 50.0


# --------------------------------------------------------------------------
# pair_013 — the design §2.2 worked example on the real pair
# --------------------------------------------------------------------------


def test_real_pair_013_offset_worked_example(contexts):
    ctx = contexts["pair_013"]
    info = ctx.sync_info
    assert ctx.offset_sec == pytest.approx(6.695)
    assert info.overlap_start_back_sec == pytest.approx(6.695, abs=0.01)
    # Design §2.2 quotes the metadata window end 13.479 (= 6.695 + 6.784
    # container duration); the reader tightens it to offset + front
    # PTS-index duration = 6.695 + 6.715 = 13.410. The code is authoritative
    # (design D1); the canonical-time relations of §2.2 are unaffected.
    assert info.overlap_end_back_sec == pytest.approx(
        6.695 + float(ctx.front_index.duration_norm_sec), abs=0.01)

    # The front frame nearest stream time 0.500 s has canonical time
    # 0.500 + 6.695 = 7.195 s on the back timeline, and the reader's lookup
    # at t_ref = canonical returns exactly that frame.
    k = ctx.front_index.nearest_frame(0.5).frame_index
    pts_k = float(ctx.front_index.pts_norm_sec[k])
    fb, ff, status = ctx.reader.get_synchronized_pair(pts_k + 6.695)
    assert status["frame_idx_front"] == k
    assert status["t_front_actual_pts_sec"] == pytest.approx(pts_k, abs=1e-6)
    assert status["t_front_target_sec"] == pytest.approx(pts_k, abs=1e-6)


# --------------------------------------------------------------------------
# VFR classification (Phase 12C forensic): 001 front, 013 back, 014 back
# are VFR; pair_015's 29.97 fps back stream is constant-rate.
# --------------------------------------------------------------------------


def test_real_vfr_classification(contexts):
    assert contexts["pair_001"].front_index.is_vfr is True
    assert contexts["pair_013"].back_index.is_vfr is True
    assert contexts["pair_014"].back_index.is_vfr is True
    back15 = contexts["pair_015"].back_index
    assert back15.is_vfr is False
    assert back15.effective_fps == pytest.approx(29.97, abs=0.1)
    assert back15.frame_count == 190


# --------------------------------------------------------------------------
# pair_015 — unique-frame counting on the 60 Hz tick grid (bounded scan):
# is_new flags must fire exactly once per distinct physical frame.
# --------------------------------------------------------------------------


def test_real_pair_015_unique_frame_counting():
    reader = SynchronizedVideoReader(
        "pair_015", data_root=DATA_ROOT, metadata_dir=DATA_ROOT / "metadata")
    try:
        t0 = reader.sync_info.overlap_start_back_sec
        seen_back: list[int] = []
        seen_front: list[int] = []
        new_back = 0
        new_front = 0
        n_ticks = 24
        for k in range(n_ticks):
            fb, ff, status = reader.get_synchronized_pair(t0 + k / 60.0)
            assert status["is_in_overlap"] is True
            seen_back.append(int(status["frame_idx_back"]))
            seen_front.append(int(status["frame_idx_front"]))
            new_back += 1 if status["is_new_back_frame"] else 0
            new_front += 1 if status["is_new_front_frame"] else 0
        # The invariant: updates == distinct physical frames presented.
        assert new_back == len(set(seen_back))
        assert new_front == len(set(seen_front))
        # And repetition genuinely occurred for the 29.97 fps back camera:
        # 24 ticks present fewer than 24 distinct back frames.
        assert len(set(seen_back)) < n_ticks
        assert len(set(seen_back)) >= 10  # ~half the ticks, not a degenerate scan
    finally:
        reader.close()


# --------------------------------------------------------------------------
# Q5 vs grid deviation on real streams (GAP-4 quantities, real data)
# --------------------------------------------------------------------------


def test_real_q5_and_grid_quantities(contexts):
    # pair_013 back: nominal-grid error is enormous (Phase 12C measured
    # ~2.5 s), while true nearest-frame lookup error over the overlap stays
    # within presentation bounds.
    ctx13 = contexts["pair_013"]
    grid13 = dcp.nominal_grid_deviation_ms(ctx13.back_index)
    assert grid13["max"] > 500.0

    ctx15 = contexts["pair_015"]
    q5 = dcp.compute_q5_overlap_stats(ctx15, 60.0)
    assert q5["n_samples"] > 100
    # Back stream is 29.97 fps: worst presentation error ~ half its interval.
    assert q5["delta_pts_back_ms"]["max"] < 20.0
    assert q5["delta_pts_front_ms"]["max"] < 20.0
    # Q5 is not Q4: the pair's sync residual (37 ms) is a different quantity
    # and is carried separately.
    assert ctx15.epsilon_sync_ms == pytest.approx(37.0)
