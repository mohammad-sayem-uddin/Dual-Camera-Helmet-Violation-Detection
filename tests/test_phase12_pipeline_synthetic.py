"""Phase 12 Pipeline (Stage G) — Stage H Synthetic Battery Test Suite.

Implements the synthetic half of the Stage H test matrix (design
INTEGRATED_DUAL_CAMERA_SYNCHRONIZATION_ENGINE_DESIGN.md §7.13):

  S-6060         60/60 fps aligned pair (primary loop + tick mode)
  S-6030         60/30 fps pair — primary loop covers physical frames exactly once
  S-VFR          mixed VFR intervals (injected FramePTSIndex, 013/014-style bursts)
  S-DROP         dropped frames / PTS gaps (injected FramePTSIndex)
  S-DUP-PTS      duplicated timestamps retained, not deduplicated (GAP-5)
  S-OFFSET-POS   small positive offset semantics
  S-OFFSET-NEG   negative offset semantics (pair_019-style, front leads)
  S-OFFSET-013   the design §2.2 worked example (offset_sec = +6.695)
  S-OVERLAP-EDGE ticks exactly at overlap start/end +/- 1 frame
  S-MISSING-CAM  missing camera -> FAILED_PRECONDITION, no artifacts, batch isolation
  S-DET-1        determinism (§7.8) — content-scoped comparison (see normalizer)
  S-NEW-1..4     new-frame-only rule (design §7.5) at gate/tracker/pipeline level

Synthetic pairs are generated with known PTS tables (mp4v files + pinned
metadata JSON). Full-pipeline tests patch a scripted fake detector into the
pipeline module's PerCameraDetector seam; tracker, association, reader and
PTS indexes are the REAL approved components (Stage G smoke pattern).
The one exception is the run_all batch-isolation test, which by design runs
in subprocesses and therefore uses the REAL detectors end to end.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.pipeline.dual_camera_pipeline as dcp
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.synchronization.pts_index import FramePTSIndex
from src.synchronization.synchronized_reader import DeduplicatedTemporalGate

REPO = Path(__file__).resolve().parents[1]
W, H = 320, 180  # small synthetic frames (tracker/association are resolution-agnostic)


# --------------------------------------------------------------------------
# Synthetic dataset construction
# --------------------------------------------------------------------------


def _make_video(path: Path, n_frames: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert vw.isOpened(), f"cannot open writer for {path}"
    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        x = 20 + (i * 3) % 120
        cv2.rectangle(frame, (x, 60), (x + 90, 130), (0, 180, 255), -1)
        cv2.circle(frame, (x + 45, 42), 16, (200, 200, 200), -1)
        vw.write(frame)
    vw.release()


def _meta_item(pair_id: str, *, offset: float, front_dur: float, back_dur: float,
               drift: float = 0.0, residual: float = 3.0, quality: str = "HIGH",
               initial: float | None = None, final: float | None = None) -> dict:
    return {
        "pair_id": pair_id,
        "offset_sec": offset,
        "initial_offset_sec": offset if initial is None else initial,
        "final_offset_sec": offset if final is None else final,
        "drift_ms": drift,
        "drift_status": "DRIFT_PRESENT" if drift else "NO_DRIFT",
        "residual_error_ms": residual,
        "quality": quality,
        "overlap_duration": 0.0,  # reader recomputes/tightens from PTS indexes
        "back_duration": back_dur,
        "front_duration": front_dur,
        "status": "CONFIRMED",
    }


def _build_dataset(data_root: Path, specs: list[dict]) -> Path:
    """Each spec: pair_id, front (n, fps), back (n, fps) or back=None (missing),
    plus _meta_item kwargs. Durations are set to the PTS-index durations
    ((n-1)/fps) so metadata and index agree exactly."""
    items = []
    for spec in specs:
        pid = spec["pair_id"]
        nf, ffps = spec["front"]
        _make_video(data_root / "front" / f"{pid}_front.mp4", nf, ffps)
        front_dur = (nf - 1) / ffps
        back = spec.get("back")
        if back is not None:
            nb, bfps = back
            _make_video(data_root / "back" / f"{pid}_back.mp4", nb, bfps)
            back_dur = (nb - 1) / bfps
        else:
            back_dur = spec.get("back_duration", 1.0)
        items.append(_meta_item(pid, front_dur=front_dur, back_dur=back_dur,
                                **spec.get("meta", {})))
    meta_dir = data_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "synchronization_metadata.json"
    meta_path.write_text(json.dumps(items, indent=1))
    return meta_path


def _config(data_root: Path, metadata_path: Path, out_root: Path,
            resume: bool = True) -> dcp.PipelineConfig:
    return dcp.PipelineConfig(
        data_root=data_root,
        metadata_path=metadata_path,
        tracking_config_path=REPO / "configs" / "tracking_config.yaml",
        association_config_path=REPO / "configs" / "cross_camera_association_config.yaml",
        output_root=out_root,
        device="cpu",
        resume=resume,
    )


# --------------------------------------------------------------------------
# Scripted fake detector (module seam) — deterministic in (camera, frame_index)
# --------------------------------------------------------------------------


def _sd(cam: str, idx: int, pts: float, bbox, conf, cid, cname, cat, src) -> StructuredDetection:
    return StructuredDetection(
        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], confidence=conf,
        class_id=cid, class_name=cname, category=cat, source_model=src,
        camera_id=cam, frame_index=idx, pts_timestamp_sec=pts,
    )


class FakePerCameraDetector:
    """PerCameraDetector-shaped stand-in. Emits one motorcycle + rider + bare
    head on frames 5..40 (both cameras), nothing elsewhere. Records every call."""

    calls: list[tuple[str, int]] = []

    def __init__(self, camera_id, device=None, vehicle_conf_threshold=0.18,
                 helmet_conf_threshold=0.25, vehicle_detector=None, helmet_detector=None):
        self.camera_id = camera_id

    def detect_frame(self, frame_bgr, frame_index, pts_timestamp_sec):
        FakePerCameraDetector.calls.append((self.camera_id, int(frame_index)))
        motos, persons, nohelm = [], [], []
        if 5 <= frame_index <= 40:
            motos.append(_sd(self.camera_id, frame_index, pts_timestamp_sec,
                             (100.0, 55.0, 200.0, 120.0), 0.9, 3, "motorcycle",
                             "motorcycle", "yolov8n_coco"))
            persons.append(_sd(self.camera_id, frame_index, pts_timestamp_sec,
                               (125.0, 20.0, 175.0, 115.0), 0.9, 0, "person",
                               "person", "yolov8n_coco"))
            nohelm.append(_sd(self.camera_id, frame_index, pts_timestamp_sec,
                              (135.0, 18.0, 165.0, 48.0), 0.9, 0, "no_helmet",
                              "head", "phase9_best_helmet"))
        return FrameDetectionBundle(
            camera_id=self.camera_id, frame_index=int(frame_index),
            pts_timestamp_sec=float(pts_timestamp_sec),
            motorcycles=motos, persons=persons, helmet_boxes=[], no_helmet_boxes=nohelm,
            all_heads=nohelm, vehicle_latency_ms=0.0, helmet_latency_ms=0.0,
            total_latency_ms=0.0, device="fake",
        )


@pytest.fixture()
def fake_detector(monkeypatch):
    FakePerCameraDetector.calls = []
    monkeypatch.setattr(dcp, "PerCameraDetector", FakePerCameraDetector)
    return FakePerCameraDetector


# --------------------------------------------------------------------------
# Content normalizer for determinism comparisons (S-DET-1 / S-NEW-4).
#
# Per Stage G interpretation #9, the approved to_dict() payloads embed
# wall-clock latency fields (*_latency_ms) that are content-exempt; and Phase
# 10 alert IDs embed a uuid4 suffix (violation_decision.py: VIOLATION-{track}-
# {frame}-{UUID6}), so alert references are compared with the suffix masked.
# Everything else must match exactly.
# --------------------------------------------------------------------------

_ALERT_RE = re.compile(r"(VIOLATION-[A-Za-z0-9_]+-\d+)-[0-9A-Fa-f]{6}\b")
_DROP_KEYS = {"runtime_ms", "elapsed_ms", "wall_time_sec", "peak_rss_mb"}


def _normalize(obj):
    if isinstance(obj, dict):
        return {
            k: _normalize(v)
            for k, v in obj.items()
            if k not in _DROP_KEYS and not k.endswith("_latency_ms")
        }
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, str):
        return _ALERT_RE.sub(r"\1-XXXXXX", obj)
    return obj


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Shared 60/30 dataset (S-6030 geometry = Stage G smoke geometry):
# front 90 frames @60, back 45 frames @30, offset +0.5 s.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair6030(tmp_path_factory):
    root = tmp_path_factory.mktemp("s6030")
    data_root = root / "data"
    meta = _build_dataset(data_root, [{
        "pair_id": "syn6030", "front": (90, 60.0), "back": (45, 30.0),
        "meta": {"offset": 0.5},
    }])
    return {"data_root": data_root, "metadata": meta, "root": root}


@pytest.fixture(scope="module")
def pair6030_run(pair6030, tmp_path_factory):
    """One primary-loop run of the 60/30 pair, fake detector at the seam."""
    out_root = tmp_path_factory.mktemp("s6030_out") / "out"
    original = dcp.PerCameraDetector
    FakePerCameraDetector.calls = []
    dcp.PerCameraDetector = FakePerCameraDetector
    try:
        result = dcp.run_pair("syn6030", _config(pair6030["data_root"],
                                                 pair6030["metadata"], out_root))
        calls = list(FakePerCameraDetector.calls)
    finally:
        dcp.PerCameraDetector = original
    return {"result": result, "out_root": out_root, "calls": calls,
            "pair_dir": out_root / "syn6030"}


# --------------------------------------------------------------------------
# S-6060 — 60/60 aligned pair
# --------------------------------------------------------------------------


def test_s6060_aligned_pair_primary_and_tick(tmp_path, fake_detector):
    data_root = tmp_path / "data"
    meta = _build_dataset(data_root, [{
        "pair_id": "syn6060", "front": (60, 60.0), "back": (60, 60.0),
        "meta": {"offset": 0.0},
    }])
    out_root = tmp_path / "out"
    result = dcp.run_pair("syn6060", _config(data_root, meta, out_root))
    assert result.status == dcp.PairStatus.COMPLETE, result.error
    pair_dir = out_root / "syn6060"

    # Primary loop: every physical frame detected exactly once, per camera.
    front_calls = sorted(i for c, i in FakePerCameraDetector.calls if c == "front")
    back_calls = sorted(i for c, i in FakePerCameraDetector.calls if c == "back")
    assert front_calls == list(range(60))
    assert back_calls == list(range(60))
    assert len(_read_jsonl(pair_dir / "syn6060_front_detections.jsonl")) == 60
    assert len(_read_jsonl(pair_dir / "syn6060_back_detections.jsonl")) == 60

    summary = json.loads((pair_dir / "pair_summary.json").read_text())
    assert summary["validation"]["all_passed"] is True
    assert summary["sync"]["offset_sec"] == 0.0

    # Tick mode on an aligned 60/60 pair: no frame is ever repeated, so the
    # gate must execute every update and skip nothing.
    FakePerCameraDetector.calls = []
    tick = dcp.run_pair_tick_mode("syn6060", _config(data_root, meta, out_root))
    gc_ = tick["gate_counters"]
    assert tick["ticks"] >= 60
    assert gc_["front_unique_updates"] == 60
    assert gc_["back_unique_updates"] == 60
    assert gc_["front_skipped_duplicates"] == 0
    assert gc_["back_skipped_duplicates"] == 0
    assert tick["front"]["frames_updated"] == 60
    assert tick["back"]["frames_updated"] == 60


# --------------------------------------------------------------------------
# S-6030 — 60/30 pair, primary loop covers physical frames exactly once
# --------------------------------------------------------------------------


def test_s6030_primary_loop_physical_frame_coverage(pair6030_run):
    result = pair6030_run["result"]
    assert result.status == dcp.PairStatus.COMPLETE, result.error
    pair_dir = pair6030_run["pair_dir"]

    # The back camera contributes its 45 PHYSICAL frames — not the ~90
    # presentations a 60 Hz tick grid would show.
    assert len(_read_jsonl(pair_dir / "syn6030_back_detections.jsonl")) == 45
    assert len(_read_jsonl(pair_dir / "syn6030_front_detections.jsonl")) == 90
    back_calls = sorted(i for c, i in pair6030_run["calls"] if c == "back")
    front_calls = sorted(i for c, i in pair6030_run["calls"] if c == "front")
    assert back_calls == list(range(45))
    assert front_calls == list(range(90))

    summary = json.loads((pair_dir / "pair_summary.json").read_text())
    assert summary["coverage"]["front"]["frames_decoded"] == 90
    assert summary["coverage"]["back"]["frames_decoded"] == 45
    assert summary["coverage"]["front"]["coverage_complete"] is True
    assert summary["coverage"]["back"]["coverage_complete"] is True


# --------------------------------------------------------------------------
# S-NEW-1 — tick mode on the 60/30 pair: unique updates == physical frames
# in the overlap; the repeated 30 fps presentations never re-trigger work.
# --------------------------------------------------------------------------


def test_s_new_1_tick_mode_60_30_unique_updates(pair6030, tmp_path, fake_detector):
    tick = dcp.run_pair_tick_mode(
        "syn6030", _config(pair6030["data_root"], pair6030["metadata"], tmp_path / "o"))
    gc_ = tick["gate_counters"]

    # Independent expectation from the PTS tables themselves.
    fidx = FramePTSIndex.from_video(pair6030["data_root"] / "front" / "syn6030_front.mp4")
    bidx = FramePTSIndex.from_video(pair6030["data_root"] / "back" / "syn6030_back.mp4")
    offset = 0.5
    overlap_start = max(0.0, offset)
    overlap_end = min(float(bidx.pts_norm_sec[-1]), offset + float(fidx.pts_norm_sec[-1]))
    exp_back_unique = int(np.sum((bidx.pts_norm_sec >= overlap_start - 1e-9)
                                 & (bidx.pts_norm_sec <= overlap_end + 1e-9)))
    exp_front_unique = int(np.sum((fidx.pts_norm_sec + offset >= overlap_start - 1e-9)
                                  & (fidx.pts_norm_sec + offset <= overlap_end + 1e-9)))
    assert exp_back_unique == 30  # back frames 15..44 fall in [0.5, 1.4667]
    assert exp_front_unique == 59

    assert gc_["back_unique_updates"] == exp_back_unique
    assert gc_["front_unique_updates"] == exp_front_unique
    assert gc_["back_skipped_duplicates"] > 0, "30 fps repeats on the 60 Hz grid must be gated"
    assert gc_["back_skipped_duplicates"] == gc_["back_ticks_seen"] - exp_back_unique
    # Front runs at tick rate: exactly one unique update per tick.
    assert gc_["front_unique_updates"] == tick["ticks"]

    # Tracker advanced exactly once per unique frame, per camera.
    assert tick["back"]["frames_updated"] == exp_back_unique
    assert tick["front"]["frames_updated"] == exp_front_unique

    # Detector ran only on genuinely new frames, exactly the recorded set.
    back_calls = sorted(i for c, i in FakePerCameraDetector.calls if c == "back")
    front_calls = sorted(i for c, i in FakePerCameraDetector.calls if c == "front")
    assert back_calls == sorted(tick["back"]["detections"].keys())
    assert front_calls == sorted(tick["front"]["detections"].keys())
    assert len(back_calls) == exp_back_unique < gc_["back_ticks_seen"]


# --------------------------------------------------------------------------
# S-NEW-2 — frozen-frame injection at the tracker guard (D3 layer 3):
# the same frame_index re-presented 5x advances state exactly once.
# --------------------------------------------------------------------------


def test_s_new_2_frozen_frame_injection_tracker_guard(fake_detector):
    from src.tracking.camera_tracker import CameraTrackingPipeline

    pipe = CameraTrackingPipeline.from_yaml("front", REPO / "configs" / "tracking_config.yaml")
    det = FakePerCameraDetector("front")
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    bundle = det.detect_frame(frame, 10, 10 / 60.0)

    first = pipe.step(bundle, should_update_state=True)
    assert first["did_advance_state"] is True
    updated_after_first = pipe.frames_updated
    state_hash_first = json.dumps(first["active_moto_tracks"], sort_keys=True, default=str)

    last = first
    for _ in range(4):  # same index re-presented: 5 presentations total
        last = pipe.step(bundle, should_update_state=True)
        assert last["did_advance_state"] is False
    assert pipe.duplicate_ticks_skipped == 4
    assert pipe.frames_updated == updated_after_first
    state_hash_last = json.dumps(last["active_moto_tracks"], sort_keys=True, default=str)
    assert state_hash_last == state_hash_first, "track state must not change on repeats"


# --------------------------------------------------------------------------
# S-NEW-3 — VFR burst at the gate: updates == distinct indices presented,
# repeats return the cached result object without re-executing the update.
# --------------------------------------------------------------------------


def test_s_new_3_vfr_burst_gate_updates_equal_distinct_indices():
    gate = DeduplicatedTemporalGate()
    executed: list[int] = []

    # Irregular (burst) presentation: long holds, bursts of new indices,
    # with repeats interleaved — flags computed as the reader computes them
    # (is_new == index differs from the last presented index).
    presented = [0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9]
    last_idx = -1
    results = []
    for idx in presented:
        is_new = idx != last_idx
        last_idx = idx

        def _update(idx=idx):
            executed.append(idx)
            return {"frame_index": idx}

        res, did = gate.step_camera("back", is_new, _update)
        results.append((idx, res, did))

    distinct = sorted(set(presented))
    assert executed == distinct
    assert gate.back_unique_updates == len(distinct)
    assert gate.back_skipped_duplicates == len(presented) - len(distinct)
    assert gate.back_ticks_seen == len(presented)
    for (idx, res, did), expect_did in zip(results, [p != q for p, q in
                                                    zip(presented, [-1] + presented[:-1])]):
        assert did == expect_did
        assert res["frame_index"] == idx  # repeats return the cached record for that index


# --------------------------------------------------------------------------
# S-NEW-4 — primary-loop vs tick-mode detection equivalence by
# (camera, frame_index) over the overlap (design §7.5; content-scoped
# comparison per Stage G interp #9 / OQ-1).
# --------------------------------------------------------------------------


def test_s_new_4_tick_primary_detection_equivalence(pair6030, pair6030_run, tmp_path,
                                                    fake_detector):
    tick = dcp.run_pair_tick_mode(
        "syn6030", _config(pair6030["data_root"], pair6030["metadata"], tmp_path / "o"))
    pair_dir = pair6030_run["pair_dir"]
    for cam in ("front", "back"):
        primary = {rec["frame_index"]: rec for rec in
                   _read_jsonl(pair_dir / f"syn6030_{cam}_detections.jsonl")}
        tick_recs = tick[cam]["detections"]
        assert tick_recs, f"tick mode produced no {cam} detections"
        assert set(tick_recs) <= set(primary), "tick detections must be a subset (overlap only)"
        for idx, rec in tick_recs.items():
            assert _normalize(rec) == _normalize(primary[idx]), \
                f"{cam} frame {idx}: tick-mode detection content != primary-loop content"


# --------------------------------------------------------------------------
# S-VFR / S-DROP / S-DUP-PTS — injected PTS tables (design §7.13 allows
# injected index objects for exactly these stream-shape cases).
# --------------------------------------------------------------------------


def _index_from_intervals(intervals_ms: list[float]) -> FramePTSIndex:
    ticks = [0]
    for iv in intervals_ms:
        ticks.append(ticks[-1] + int(round(iv * 90.0)))  # time_base 1/90000
    return FramePTSIndex("synthetic.mp4", np.array(ticks, dtype=np.int64), 1, 90000,
                         "60/1", "60/1")


def test_s_vfr_mixed_intervals_index():
    # 013/014-style burst pattern: 60 fps runs, a 30 fps stretch, an 120 fps burst.
    idx = _index_from_intervals([16.667] * 10 + [33.333] * 10 + [8.333] * 10)
    assert idx.frame_count == 31
    assert idx.is_vfr is True
    assert idx.count_dt_60fps >= 10
    assert idx.count_dt_30fps >= 10
    assert idx.interval_spread_ms > 20.0
    # Nearest-frame lookup stays monotone over the whole stream.
    targets = np.linspace(0.0, idx.duration_norm_sec, 200)
    chosen = [idx.nearest_frame(float(t)).frame_index for t in targets]
    assert chosen == sorted(chosen)
    # The nominal uniform grid is a poor clock for this stream (the Phase 12C
    # nominal-error phenomenon): grid deviation is materially non-zero.
    grid_dev = dcp.nominal_grid_deviation_ms(idx)
    assert grid_dev["max"] > 5.0


def test_s_drop_pts_gap_index():
    full = _index_from_intervals([16.667] * 61)  # 62 frames
    # Frames 20 and 21 dropped: intervals 19..21 merge into one 50 ms gap,
    # leaving 59 intervals / 60 frames.
    dropped = _index_from_intervals([16.667] * 19 + [50.0] + [16.667] * 39)
    assert dropped.frame_count == full.frame_count - 2  # two packets absent
    assert dropped.max_interval_ms == pytest.approx(50.0, abs=0.1)
    # A target at a dropped frame's nominal time resolves to a neighbour with
    # delta == half the gap (16.67 ms), never to a phantom frame.
    missing_t = 20 * (1000.0 / 60.0) / 1000.0
    res = dropped.nearest_frame(missing_t)
    assert res.delta_pts_ms == pytest.approx(1000.0 / 60.0, abs=0.1)
    assert res.frame_index in (19, 20)


def test_s_dup_pts_duplicate_ticks_not_deduplicated():
    # GAP-5: FramePTSIndex sorts but does NOT deduplicate packet ticks; the
    # duplicate is counted as a packet. Stage G's coverage gate compares
    # decoded vs this packet count, which is what makes divergence visible.
    ticks = np.array([0, 1500, 1500, 3000, 4500], dtype=np.int64)  # 16.667 ms steps
    idx = FramePTSIndex("dup.mp4", ticks, 1, 90000, "60/1", "60/1")
    assert idx.frame_count == 5
    assert len(np.unique(idx.pts_norm_sec)) == 4  # duplicate timestamp retained
    res = idx.nearest_frame(float(idx.pts_norm_sec[1]))
    assert res.delta_pts_ms == 0.0
    assert res.frame_index in (1, 2)


# --------------------------------------------------------------------------
# S-OFFSET-013 / S-OFFSET-POS / S-OFFSET-NEG — offset sign semantics via the
# real reader + SyncContext on synthetic streams (no decoding beyond a few
# get_synchronized_pair calls).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offset_pairs(tmp_path_factory):
    root = tmp_path_factory.mktemp("offsets")
    data_root = root / "data"
    meta = _build_dataset(data_root, [
        {"pair_id": "off013", "front": (30, 10.0), "back": (100, 10.0),
         "meta": {"offset": 6.695, "initial": 8.357, "final": 4.595}},
        {"pair_id": "offpos", "front": (60, 60.0), "back": (60, 60.0),
         "meta": {"offset": 0.25}},
        {"pair_id": "offneg", "front": (30, 10.0), "back": (20, 10.0),
         "meta": {"offset": -2.380}},
    ])
    return {"data_root": data_root, "metadata": meta, "root": root}


def _ctx(pair_id: str, ds: dict) -> dcp.SyncContext:
    return dcp.SyncContext.from_reader(
        pair_id, _config(ds["data_root"], ds["metadata"], ds["root"] / "out"))


def test_s_offset_013_worked_example(offset_pairs):
    ctx = _ctx("off013", offset_pairs)
    try:
        assert ctx.offset_sec == pytest.approx(6.695)
        # Operative offset is offset_sec — NOT initial (+8.357) or final (+4.595).
        assert ctx.offset_sec != pytest.approx(8.357)
        assert ctx.canonical_shift_sec("front") == pytest.approx(6.695)
        assert ctx.canonical_shift_sec("back") == 0.0
        info = ctx.sync_info
        assert info.overlap_start_back_sec == pytest.approx(6.695, abs=0.01)
        assert info.overlap_end_back_sec == pytest.approx(6.695 + 2.9, abs=0.01)

        # Design §2.2: front frame at pts_norm 0.500 -> canonical 7.195;
        # lookup for t_ref 7.195 targets 0.500 in front-stream time.
        fb, ff, status = ctx.reader.get_synchronized_pair(7.195)
        assert status["t_front_target_sec"] == pytest.approx(0.5, abs=0.01)
        assert status["frame_idx_front"] == 5
        assert status["t_front_actual_pts_sec"] == pytest.approx(0.5, abs=0.01)
        assert status["t_front_actual_pts_sec"] + ctx.offset_sec == pytest.approx(7.195,
                                                                                  abs=0.01)
        assert status["is_in_overlap"] is True
    finally:
        ctx.close()


def test_s_offset_pos_small_positive(offset_pairs):
    ctx = _ctx("offpos", offset_pairs)
    try:
        assert ctx.offset_sec == pytest.approx(0.25)
        assert ctx.sync_info.overlap_start_back_sec == pytest.approx(0.25, abs=0.01)
        assert ctx.canonical_shift_sec("front") == pytest.approx(0.25)
    finally:
        ctx.close()


def test_s_offset_neg_front_leads(offset_pairs):
    ctx = _ctx("offneg", offset_pairs)
    try:
        assert ctx.offset_sec == pytest.approx(-2.380)
        info = ctx.sync_info
        # Front content leads: overlap starts at back time 0 and ends when the
        # front stream (shifted by -2.380) runs out: -2.380 + 2.9 = 0.52.
        assert info.overlap_start_back_sec == pytest.approx(0.0, abs=0.01)
        assert info.overlap_end_back_sec == pytest.approx(0.52, abs=0.01)
        # At t_ref = 0 the front lookup target is 0 - (-2.380) = +2.380 s.
        fb, ff, status = ctx.reader.get_synchronized_pair(0.0)
        assert status["t_front_target_sec"] == pytest.approx(2.380, abs=0.01)
        assert status["frame_idx_front"] == 24  # nearest 10 fps frame to 2.380 (2.4 s)
        assert status["t_front_actual_pts_sec"] == pytest.approx(2.4, abs=0.01)
    finally:
        ctx.close()


# --------------------------------------------------------------------------
# S-OVERLAP-EDGE — ticks exactly at overlap start/end +/- 1 frame
# --------------------------------------------------------------------------


def test_s_overlap_edge_boundaries(offset_pairs):
    ctx = _ctx("off013", offset_pairs)
    try:
        info = ctx.sync_info
        start, end = info.overlap_start_back_sec, info.overlap_end_back_sec
        frame = 0.1  # one 10 fps frame interval — the synthetic streams' quantum

        _, _, s_at_start = ctx.reader.get_synchronized_pair(start)
        assert s_at_start["is_in_overlap"] is True
        assert s_at_start["valid_pair"] is True

        _, _, s_before = ctx.reader.get_synchronized_pair(start - frame)
        assert s_before["is_in_overlap"] is False

        _, _, s_at_end = ctx.reader.get_synchronized_pair(end)
        assert s_at_end["is_in_overlap"] is True

        _, _, s_after = ctx.reader.get_synchronized_pair(end + frame)
        assert s_after["is_in_overlap"] is False
    finally:
        ctx.close()


# --------------------------------------------------------------------------
# S-MISSING-CAM — FAILED_PRECONDITION, no artifacts (run_pair level)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def missing_cam_ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("missing")
    data_root = root / "data"
    meta = _build_dataset(data_root, [
        {"pair_id": "brokencam", "front": (30, 30.0), "back": None,
         "back_duration": 1.0, "meta": {"offset": 0.1}},
        {"pair_id": "wholecam", "front": (30, 30.0), "back": (18, 30.0),
         "meta": {"offset": 0.1}},
    ])
    return {"data_root": data_root, "metadata": meta, "root": root}


def test_s_missing_cam_failed_precondition_no_artifacts(missing_cam_ds, tmp_path):
    out_root = tmp_path / "out"
    result = dcp.run_pair("brokencam", _config(missing_cam_ds["data_root"],
                                               missing_cam_ds["metadata"], out_root))
    assert result.status == dcp.PairStatus.FAILED_PRECONDITION
    assert result.error is not None and result.error.code == dcp.ERR_PRECONDITION
    assert result.summary is None
    # Design §7.11: a precondition failure writes NO per-pair artifacts —
    # the pair directory is never even created.
    assert not (out_root / "brokencam").exists()


def test_s_missing_cam_batch_isolation_run_all(missing_cam_ds, tmp_path):
    """run_all (subprocess isolation, REAL detectors): the alphabetically
    first pair is missing its back camera and must fail FAILED_PRECONDITION
    while the batch continues and the intact pair completes."""
    out_root = tmp_path / "out_all"
    results = dcp.run_all(["wholecam", "brokencam"],
                          _config(missing_cam_ds["data_root"],
                                  missing_cam_ds["metadata"], out_root))
    by_id = {r.pair_id: r for r in results}
    assert by_id["brokencam"].status == dcp.PairStatus.FAILED_PRECONDITION
    assert by_id["brokencam"].error is not None
    assert by_id["brokencam"].error.code == dcp.ERR_PRECONDITION
    assert not (out_root / "brokencam").exists()
    assert by_id["wholecam"].status == dcp.PairStatus.COMPLETE, by_id["wholecam"].error

    manifest = json.loads((out_root / "run_manifest.json").read_text())
    statuses = {p["pair_id"]: p["status"] for p in manifest["pairs"]}
    assert statuses == {"brokencam": "FAILED_PRECONDITION", "wholecam": "COMPLETE"}


# --------------------------------------------------------------------------
# S-DET-1 — determinism (design §7.8), content-scoped per Stage G interp #9:
# coverage CSVs byte-identical; JSON artifacts content-identical after
# dropping approved wall-clock latency fields and masking the uuid suffix
# of Phase 10 alert IDs; summary identical outside its runtime fields.
# --------------------------------------------------------------------------


def test_s_det_1_determinism_content_scoped(pair6030, pair6030_run, tmp_path, fake_detector):
    out_b = tmp_path / "out_b"
    result_b = dcp.run_pair("syn6030", _config(pair6030["data_root"],
                                               pair6030["metadata"], out_b))
    assert result_b.status == dcp.PairStatus.COMPLETE, result_b.error
    dir_a = pair6030_run["pair_dir"]
    dir_b = out_b / "syn6030"

    # Coverage CSVs carry no wall-clock or random content: byte-identical.
    for cam in ("front", "back"):
        assert (dir_a / f"syn6030_{cam}_coverage.csv").read_bytes() == \
               (dir_b / f"syn6030_{cam}_coverage.csv").read_bytes()

    # JSONL artifacts: content-identical after normalization.
    for name in ("syn6030_front_detections.jsonl", "syn6030_back_detections.jsonl",
                 "syn6030_front_track_frames.jsonl", "syn6030_back_track_frames.jsonl"):
        assert _normalize(_read_jsonl(dir_a / name)) == _normalize(_read_jsonl(dir_b / name)), name

    # JSON artifacts: content-identical after normalization.
    for name in ("syn6030_association.json", "syn6030_events.json",
                 "syn6030_front_tracks_final.json", "syn6030_back_tracks_final.json",
                 "syn6030_sync_diagnostics.json"):
        a = _normalize(json.loads((dir_a / name).read_text()))
        b = _normalize(json.loads((dir_b / name).read_text()))
        assert a == b, name

    # Summary: identical outside the design-exempt runtime fields.
    sa = _normalize(json.loads((dir_a / "pair_summary.json").read_text()))
    sb = _normalize(json.loads((dir_b / "pair_summary.json").read_text()))
    assert sa == sb
