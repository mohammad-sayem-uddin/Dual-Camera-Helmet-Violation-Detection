"""Phase 12 Pipeline (Stage G) — Stage H Unit-Semantics Test Suite.

Covers the pipeline semantics obligations of the Stage H matrix (design
INTEGRATED_DUAL_CAMERA_SYNCHRONIZATION_ENGINE_DESIGN.md §7 + GAP register):

  * D1 canonical-time mapping: t_canonical(front) = pts_norm + offset_sec,
    back unshifted — asserted on the coverage artifacts themselves.
  * GAP-1: drift_ms is REPORTED (summary + diagnostics + SyncContext) but
    NEVER applied — the canonical shift stays the constant offset_sec for the
    first and the last frame alike, and the operative offset is offset_sec,
    not initial/final_offset_sec.
  * GAP-4: Q5 (true nearest-frame lookup error) vs nominal-grid deviation
    are different quantities: Q5 stays bounded by the local frame interval
    on a stream whose grid deviation exceeds 500 ms; the grid quantity is
    emitted only under the name nominal_grid_deviation_ms.
  * GAP-6: residual (epsilon_sync_ms) and manifest quality are separate
    fields; the reader's hard-coded accuracy_classification label is not
    reused in pipeline diagnostics.
  * §7.7 checkpoint/resume lifecycle: skip only on COMPLETE + validation
    passed + all input hashes matching; any tamper re-runs the pair.
  * §7.11 error taxonomy: every reachable code is asserted at its failure
    point (ERR-PRECONDITION variants, ERR-DETECTION, ERR-COVERAGE,
    ERR-TRACKING, ERR-ASSOCIATION, ERR-RESOURCE, ERR-INTERNAL, ERR-DECODE).
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.pipeline.dual_camera_pipeline as dcp
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.synchronization.pts_index import FramePTSIndex

REPO = Path(__file__).resolve().parents[1]
W, H = 320, 180


# --------------------------------------------------------------------------
# Synthetic dataset helpers (same pattern as the synthetic battery file)
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


def _build_dataset(data_root: Path, pair_id: str, front: tuple[int, float],
                   back: tuple[int, float], meta: dict) -> Path:
    nf, ffps = front
    nb, bfps = back
    _make_video(data_root / "front" / f"{pair_id}_front.mp4", nf, ffps)
    _make_video(data_root / "back" / f"{pair_id}_back.mp4", nb, bfps)
    item = {
        "pair_id": pair_id,
        "offset_sec": meta["offset"],
        "initial_offset_sec": meta.get("initial", meta["offset"]),
        "final_offset_sec": meta.get("final", meta["offset"]),
        "drift_ms": meta.get("drift", 0.0),
        "drift_status": "DRIFT_PRESENT" if meta.get("drift") else "NO_DRIFT",
        "residual_error_ms": meta.get("residual", 3.0),
        "quality": meta.get("quality", "HIGH"),
        "overlap_duration": 0.0,
        "back_duration": (nb - 1) / bfps,
        "front_duration": (nf - 1) / ffps,
        "status": "CONFIRMED",
    }
    meta_dir = data_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "synchronization_metadata.json"
    meta_path.write_text(json.dumps([item], indent=1))
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


def _sd(cam: str, idx: int, pts: float, bbox, conf, cid, cname, cat, src) -> StructuredDetection:
    return StructuredDetection(
        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], confidence=conf,
        class_id=cid, class_name=cname, category=cat, source_model=src,
        camera_id=cam, frame_index=idx, pts_timestamp_sec=pts,
    )


class FakePerCameraDetector:
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
# Shared run: 60/30 pair, offset +0.5, drift +74 ms, initial/final offsets
# deliberately different from the operative offset (pair_015-style metadata).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sem_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sem")
    data_root = root / "data"
    meta = _build_dataset(data_root, "sem6030", (90, 60.0), (45, 30.0),
                          {"offset": 0.5, "initial": 0.65, "final": 0.42,
                           "drift": 74.0, "residual": 37.0, "quality": "HIGH"})
    out_root = root / "out"
    original = dcp.PerCameraDetector
    dcp.PerCameraDetector = FakePerCameraDetector
    try:
        result = dcp.run_pair("sem6030", _config(data_root, meta, out_root))
    finally:
        dcp.PerCameraDetector = original
    assert result.status == dcp.PairStatus.COMPLETE, result.error
    return {"data_root": data_root, "metadata": meta, "out_root": out_root,
            "pair_dir": out_root / "sem6030", "result": result}


def _coverage_rows(pair_dir: Path, pair_id: str, cam: str) -> list[dict]:
    with open(pair_dir / f"{pair_id}_{cam}_coverage.csv", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# D1 canonical mapping + GAP-1 drift reported, never applied
# --------------------------------------------------------------------------


def test_canonical_mapping_and_drift_never_applied(sem_run):
    pair_dir = sem_run["pair_dir"]
    front = _coverage_rows(pair_dir, "sem6030", "front")
    back = _coverage_rows(pair_dir, "sem6030", "back")
    assert len(front) == 90 and len(back) == 45

    # Canonicalization direction (D1), on the artifacts: front = pts_norm +
    # offset for EVERY frame — first and last alike. A drift-compensating
    # implementation would let the shift wander by up to drift_ms (74 ms);
    # the observed shift is the constant 0.5 s everywhere.
    for row in front:
        shift = float(row["canonical_pts_sec"]) - float(row["pts_norm_sec"])
        assert shift == pytest.approx(0.5, abs=1e-6)
    for row in back:
        shift = float(row["canonical_pts_sec"]) - float(row["pts_norm_sec"])
        assert shift == pytest.approx(0.0, abs=1e-6)

    summary = json.loads((pair_dir / "pair_summary.json").read_text())
    assert summary["sync"]["offset_sec"] == pytest.approx(0.5)
    assert summary["sync"]["drift_ms"] == pytest.approx(74.0)  # reported...
    assert summary["sync"]["epsilon_sync_ms"] == pytest.approx(37.0)

    # ...and the operative offset is offset_sec, not initial/final.
    ctx = dcp.SyncContext.from_reader(
        "sem6030", _config(sem_run["data_root"], sem_run["metadata"],
                           sem_run["out_root"]))
    try:
        assert ctx.offset_sec == pytest.approx(0.5)
        assert ctx.offset_sec != pytest.approx(0.65)
        assert ctx.offset_sec != pytest.approx(0.42)
        assert ctx.drift_ms == pytest.approx(74.0)
        assert ctx.canonical_shift_sec("front") == pytest.approx(0.5)
        assert ctx.canonical_shift_sec("back") == 0.0
    finally:
        ctx.close()


def test_diagnostics_field_separation_gap4_gap6(sem_run):
    diag = json.loads((sem_run["pair_dir"] / "sem6030_sync_diagnostics.json").read_text())

    # Q3/Q4/drift/quality are four separate fields (GAP-1, GAP-6).
    assert diag["offset_sec"] == pytest.approx(0.5)
    assert diag["epsilon_sync_ms"] == pytest.approx(37.0)
    assert diag["drift_ms"] == pytest.approx(74.0)
    assert diag["sync_quality"] == "HIGH"
    # The reader's hard-coded band label is deliberately not reused (GAP-6).
    assert "accuracy_classification" not in diag

    # Q5 lives under its own name with a definition; the grid quantity only
    # under nominal_grid_deviation_ms inside the per-stream blocks (GAP-4).
    q5 = diag["q5_overlap_lookup_stats"]
    assert "delta_pts_front_ms" in q5 and "delta_pts_back_ms" in q5
    assert q5["n_samples"] > 0
    for cam in ("front", "back"):
        assert "nominal_grid_deviation_ms" in diag[f"{cam}_stream"]
    assert "NOT Q5" in diag["delta_pts_field_definitions"]["nominal_grid_deviation_ms"]
    # Q7 disclaimer propagates verbatim.
    assert diag["exposure_sync_model"] == \
        "container_pts_alignment_not_physical_exposure_genlock"


# --------------------------------------------------------------------------
# GAP-4 at the function level: true Q5 vs nominal-grid deviation are
# different quantities, proven on a stream whose grid deviation is huge.
# --------------------------------------------------------------------------


def _index_from_intervals(intervals_ms: list[float]) -> FramePTSIndex:
    ticks = [0]
    for iv in intervals_ms:
        ticks.append(ticks[-1] + int(round(iv * 90.0)))  # time_base 1/90000
    return FramePTSIndex("synthetic.mp4", np.array(ticks, dtype=np.int64), 1, 90000,
                         "60/1", "60/1")


def _stub_ctx(front: FramePTSIndex, back: FramePTSIndex, offset: float,
              start: float, end: float) -> dcp.SyncContext:
    ctx = object.__new__(dcp.SyncContext)
    ctx.sync_info = SimpleNamespace(
        global_offset_sec=offset,
        overlap_start_back_sec=start,
        overlap_end_back_sec=end,
    )
    ctx.front_index = front
    ctx.back_index = back
    return ctx


def test_q5_true_lookup_vs_nominal_grid_gap4():
    # Two-rate stream: 30 fps stretch then 60 fps stretch. The uniform grid
    # (idx / effective_fps) drifts > 500 ms from true PTS mid-stream, while
    # true nearest-frame lookup error stays within half the local interval.
    idx = _index_from_intervals([33.333] * 100 + [16.667] * 200)
    grid = dcp.nominal_grid_deviation_ms(idx)
    assert grid["max"] > 500.0

    ctx = _stub_ctx(idx, idx, 0.0, 0.0, float(idx.pts_norm_sec[-1]))
    q5 = dcp.compute_q5_overlap_stats(ctx, 60.0)
    assert q5["n_samples"] > 100
    assert q5["delta_pts_front_ms"]["max"] < 20.0  # <= half of 33.3 ms + rounding
    assert q5["delta_pts_back_ms"]["max"] < 20.0
    # And the two quantities are demonstrably not the same number.
    assert q5["delta_pts_front_ms"]["max"] < grid["max"] / 10.0


def test_q5_lookup_target_uses_offset():
    # Uniform 60 fps streams; offset = half a frame. Reference ticks start at
    # the overlap start (= offset), so front targets (t_ref - offset) land
    # exactly on front frames while back targets land mid-frame.
    idx = _index_from_intervals([16.667] * 120)
    offset = 1.0 / 120.0
    ctx = _stub_ctx(idx, idx, offset, offset, float(idx.pts_norm_sec[-1]))
    q5 = dcp.compute_q5_overlap_stats(ctx, 60.0)
    assert q5["delta_pts_front_ms"]["mean"] == pytest.approx(0.0, abs=0.05)
    assert q5["delta_pts_back_ms"]["mean"] == pytest.approx(1000.0 / 120.0, abs=0.05)


# --------------------------------------------------------------------------
# §7.7 checkpoint / resume lifecycle
# --------------------------------------------------------------------------


def test_checkpoint_resume_lifecycle(tmp_path, fake_detector):
    data_root = tmp_path / "data"
    meta = _build_dataset(data_root, "resm", (48, 60.0), (24, 30.0), {"offset": 0.25})
    out_root = tmp_path / "out"
    cfg = _config(data_root, meta, out_root)
    n_frames = 48 + 24

    r1 = dcp.run_pair("resm", cfg)
    assert r1.status == dcp.PairStatus.COMPLETE, r1.error
    assert len(FakePerCameraDetector.calls) == n_frames

    # Valid checkpoint + matching hashes -> skip, zero new detector calls.
    FakePerCameraDetector.calls = []
    r2 = dcp.run_pair("resm", cfg)
    assert r2.status == dcp.PairStatus.SKIPPED_COMPLETE
    assert FakePerCameraDetector.calls == []

    # Tampered recorded hash -> full re-run.
    summary_path = out_root / "resm" / "pair_summary.json"
    s = json.loads(summary_path.read_text())
    s["inputs_sha256"]["tracking_config_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(s, indent=1, sort_keys=True))
    FakePerCameraDetector.calls = []
    r3 = dcp.run_pair("resm", cfg)
    assert r3.status == dcp.PairStatus.COMPLETE, r3.error
    assert len(FakePerCameraDetector.calls) == n_frames

    # Summary no longer claims COMPLETE -> not skipped.
    s = json.loads(summary_path.read_text())
    s["status"] = "FAILED"
    summary_path.write_text(json.dumps(s, indent=1, sort_keys=True))
    FakePerCameraDetector.calls = []
    r4 = dcp.run_pair("resm", cfg)
    assert r4.status == dcp.PairStatus.COMPLETE, r4.error
    assert len(FakePerCameraDetector.calls) == n_frames

    # resume=False -> never skipped.
    FakePerCameraDetector.calls = []
    r5 = dcp.run_pair("resm", _config(data_root, meta, out_root, resume=False))
    assert r5.status == dcp.PairStatus.COMPLETE, r5.error
    assert len(FakePerCameraDetector.calls) == n_frames

    # Unparseable summary -> treated as no checkpoint, pair re-runs cleanly.
    summary_path.write_text("{ not json")
    FakePerCameraDetector.calls = []
    r6 = dcp.run_pair("resm", cfg)
    assert r6.status == dcp.PairStatus.COMPLETE, r6.error
    assert len(FakePerCameraDetector.calls) == n_frames


# --------------------------------------------------------------------------
# §7.11 error taxonomy at its failure points
# --------------------------------------------------------------------------


def test_error_taxonomy_surface():
    assert set(dcp.ERROR_CODES) == {
        "ERR-PRECONDITION", "ERR-DECODE", "ERR-COVERAGE", "ERR-DETECTION",
        "ERR-TRACKING", "ERR-ASSOCIATION", "ERR-RESOURCE", "ERR-INTERNAL",
    }
    with pytest.raises(ValueError):
        dcp.PipelineError("ERR-BOGUS", "nope")
    assert {s.name for s in dcp.PairStatus} == {
        "PENDING", "RUNNING", "COMPLETE", "FAILED_PRECONDITION",
        "FAILED_RUNTIME", "SKIPPED_COMPLETE",
    }


def test_err_precondition_variants(tmp_path, fake_detector):
    data_root = tmp_path / "data"
    meta = _build_dataset(data_root, "prev", (30, 30.0), (18, 30.0), {"offset": 0.1})
    out_root = tmp_path / "out"

    cases = []
    # (a) pinned metadata file missing
    cases.append(_config(data_root, tmp_path / "nope" / "synchronization_metadata.json",
                         out_root))
    # (b) metadata file with the wrong name (GAP-7 pinning)
    wrong_name = tmp_path / "meta.json"
    wrong_name.write_text(meta.read_text())
    cases.append(_config(data_root, wrong_name, out_root))
    # (c) tracking config missing
    bad_track = dataclasses.replace(
        _config(data_root, meta, out_root),
        tracking_config_path=tmp_path / "missing_tracking.yaml")
    cases.append(bad_track)
    # (d) association config missing
    bad_assoc = dataclasses.replace(
        _config(data_root, meta, out_root),
        association_config_path=tmp_path / "missing_assoc.yaml")
    cases.append(bad_assoc)

    for cfg in cases:
        r = dcp.run_pair("prev", cfg)
        assert r.status == dcp.PairStatus.FAILED_PRECONDITION, r
        assert r.error is not None and r.error.code == dcp.ERR_PRECONDITION
        assert not (out_root / "prev").exists()

    # (e) unknown pair id (not in metadata)
    r = dcp.run_pair("pair_zzz", _config(data_root, meta, out_root))
    assert r.status == dcp.PairStatus.FAILED_PRECONDITION
    assert r.error is not None and r.error.code == dcp.ERR_PRECONDITION
    assert not (out_root / "pair_zzz").exists()


class _BoomDetector(FakePerCameraDetector):
    def detect_frame(self, frame_bgr, frame_index, pts_timestamp_sec):
        raise RuntimeError("synthetic detector explosion")


class _ShiftedPTSDetector(FakePerCameraDetector):
    def detect_frame(self, frame_bgr, frame_index, pts_timestamp_sec):
        bundle = super().detect_frame(frame_bgr, frame_index, pts_timestamp_sec)
        return dataclasses.replace(bundle,
                                   pts_timestamp_sec=bundle.pts_timestamp_sec + 0.5)


def _small_pair(tmp_path, pair_id="errp", truncate_back=False):
    data_root = tmp_path / "data"
    meta = _build_dataset(data_root, pair_id, (30, 30.0), (18, 30.0), {"offset": 0.1})
    if truncate_back:
        bp = data_root / "back" / f"{pair_id}_back.mp4"
        blob = bp.read_bytes()
        bp.write_bytes(blob[: int(len(blob) * 0.55)])
    return data_root, meta


def test_err_detection_on_detector_exception(tmp_path, monkeypatch):
    data_root, meta = _small_pair(tmp_path)
    monkeypatch.setattr(dcp, "PerCameraDetector", _BoomDetector)
    out_root = tmp_path / "out"
    r = dcp.run_pair("errp", _config(data_root, meta, out_root))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_DETECTION
    status = json.loads((out_root / "errp" / "pair_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["code"] == dcp.ERR_DETECTION


def test_err_coverage_on_bundle_pts_mismatch(tmp_path, monkeypatch):
    data_root, meta = _small_pair(tmp_path)
    monkeypatch.setattr(dcp, "PerCameraDetector", _ShiftedPTSDetector)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_COVERAGE


def test_err_tracking_on_tracker_init_failure(tmp_path, monkeypatch, fake_detector):
    data_root, meta = _small_pair(tmp_path)

    class BoomTracker:
        @classmethod
        def from_yaml(cls, *a, **k):
            raise RuntimeError("synthetic tracker init failure")

    monkeypatch.setattr(dcp, "CameraTrackingPipeline", BoomTracker)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_TRACKING


def test_err_association_on_engine_failure(tmp_path, monkeypatch, fake_detector):
    data_root, meta = _small_pair(tmp_path)

    class BoomEngine:
        @classmethod
        def from_yaml(cls, *a, **k):
            return cls()

        def associate_pipelines(self, **kwargs):
            raise RuntimeError("synthetic association failure")

    monkeypatch.setattr(dcp, "CrossCameraAssociationEngine", BoomEngine)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_ASSOCIATION


def test_err_resource_ceiling(tmp_path, monkeypatch, fake_detector):
    data_root, meta = _small_pair(tmp_path)
    monkeypatch.setattr(dcp, "DEFAULT_RSS_CEILING_MB", 0.0)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_RESOURCE


def test_err_internal_on_unexpected_exception(tmp_path, monkeypatch, fake_detector):
    data_root, meta = _small_pair(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("synthetic diagnostics explosion")

    monkeypatch.setattr(dcp, "build_sync_diagnostics", boom)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_INTERNAL
    assert r.error.traceback_str and "synthetic diagnostics explosion" in r.error.traceback_str


def test_err_decode_on_truncated_stream(tmp_path, fake_detector):
    # Truncating to 55% removes the mp4 moov atom; opening the stream fails
    # and the failure must surface as ERR-DECODE (probed on this stack:
    # "Failed to open video file with OpenCV" -> FAILED_RUNTIME/ERR-DECODE).
    data_root, meta = _small_pair(tmp_path, truncate_back=True)
    r = dcp.run_pair("errp", _config(data_root, meta, tmp_path / "out"))
    assert r.status == dcp.PairStatus.FAILED_RUNTIME
    assert r.error is not None and r.error.code == dcp.ERR_DECODE
