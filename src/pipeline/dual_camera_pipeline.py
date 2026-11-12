"""Integrated Dual-Camera Helmet-Violation Pipeline (Stage G implementation).

Orchestration layer around the approved Phase 12C synchronization engine, built
exactly to the Stage F design specification
(`docs/phase12/INTEGRATED_DUAL_CAMERA_SYNCHRONIZATION_ENGINE_DESIGN.md`).
This module re-implements NOTHING: synchronization comes from Phase 12C
(`FramePTSIndex`, `VideoReader`, `SynchronizedVideoReader`), detection from
Phase 12D (`PerCameraDetector`), tracking + the embedded Phase 10 adapter from
`CameraTrackingPipeline`, and association from Phase 12E
(`CrossCameraAssociationEngine`). The standalone Phase 10
`TemporalVerificationEngine` is never instantiated (design §5.4 / GAP-8).

Binding design decisions implemented here (design §1, §7):

- D1 — Code is authoritative for all time conventions. The operative offset is
  the Phase 11 metadata `offset_sec` (== `pair_manifest.csv:temporal_offset_sec`),
  never recomputed, re-estimated, or re-signed. The two sanctioned equations
  (design §2.3), with `Δt_offset := offset_sec`:
    * Lookup direction:        `t_front_target = t_ref - Δt_offset`
      (what `SynchronizedVideoReader.get_synchronized_pair` computes).
    * Canonicalization:        `t_canonical(front frame) = pts_norm_front + Δt_offset`
      (what Pass 1 used and what Phase 12E tracklets consume).
  The back camera is the reference timeline and needs no transform:
  `t_canonical(back frame) = pts_norm_back`.
- D2 — Primary loop is per-camera sequential decode (the Pass 1 pattern);
  synchronization is applied through canonical PTS maps and Phase 12C
  diagnostics. Tick mode is retained as a secondary path (`run_pair_tick_mode`)
  for overlap diagnostics and Stage H equivalence tests.
- D3 — New-frame-only processing, keyed purely on `(camera, frame_index)`
  identity, enforced at three layers: primary-loop construction, the reader's
  `is_new_*` flags + `DeduplicatedTemporalGate` (tick mode), and the tracker
  `step()` guard. A repeated frame never re-runs detection and never advances
  temporal state.
- D4 — One pair per process, `pair_summary.json` checkpoint with hash-validated
  resume, per-pair failure isolation (`run_all` driver).
- D5 — Every emitted time quantity carries its design §3 name. TRUE nearest-frame
  lookup error (Q5) is computed by a diagnostic pass over the overlap window
  using only `FramePTSIndex.nearest_frame`; Pass 1's nominal-grid deviation is
  reported only under the name `nominal_grid_deviation_ms`, never `delta_pts`
  (GAP-4).
- D6 — No drift compensation. Phase 11 `drift_ms` is measured and reported per
  pair (sync diagnostics + summary); it is never applied (GAP-1).

Memory discipline (design §7.6, reference machine Apple M1 8 GB): pairs are
processed one at a time in subprocesses; frames are consumed forward-only,
exactly once, and never accumulated (JSONL/CSV artifacts are streamed per
frame); appearance keyframes are bounded by `keyframe_max_per_camera`; models
are shared process-wide via `SharedDetectorRegistry`.

Determinism (design §7.8): fixed iteration order (pair id ascending, front
before back), sorted JSON keys, and no wall-clock values inside per-pair
artifacts except the explicitly exempt runtime fields of `pair_summary.json`
(`wall_time_sec`, `peak_rss_mb`). Two inherited exceptions, documented in the
Stage G progress file: the approved `FrameDetectionBundle.to_dict()` and
`CrossCameraAssociationBatchResult.to_dict()` embed detector/association
latency measurements (`*_latency_ms`, `runtime_ms`); those approved payloads
are written verbatim per design §7.9.

Process working directory must be the repository root: the approved model
adapters resolve their default checkpoints (`yolov8n.pt`, Phase 9 `best.pt`)
relative to the CWD. `run_all` enforces this for its worker subprocesses.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.association.cross_camera_association import CrossCameraAssociationEngine
from src.association.tracklet_models import (
    CrossCameraAssociationBatchResult,
    TimestampQualityRecord,
)
from src.detection.camera_detector import PerCameraDetector
from src.models.helmet_detector_adapter import HelmetDetectorAdapter
from src.models.vehicle_detector_adapter import VehiclePersonDetectorAdapter
from src.synchronization.pts_index import FramePTSIndex
from src.synchronization.synchronized_reader import (
    DeduplicatedTemporalGate,
    SynchronizedVideoReader,
)
from src.tracking.camera_tracker import CameraTrackingPipeline
from src.video.video_reader import VideoReader


# --------------------------------------------------------------------------
# Error taxonomy (design §7.11) — exactly eight codes.
# --------------------------------------------------------------------------

ERR_PRECONDITION = "ERR-PRECONDITION"
ERR_DECODE = "ERR-DECODE"
ERR_COVERAGE = "ERR-COVERAGE"
ERR_DETECTION = "ERR-DETECTION"
ERR_TRACKING = "ERR-TRACKING"
ERR_ASSOCIATION = "ERR-ASSOCIATION"
ERR_RESOURCE = "ERR-RESOURCE"
ERR_INTERNAL = "ERR-INTERNAL"

ERROR_CODES = (
    ERR_PRECONDITION,
    ERR_DECODE,
    ERR_COVERAGE,
    ERR_DETECTION,
    ERR_TRACKING,
    ERR_ASSOCIATION,
    ERR_RESOURCE,
    ERR_INTERNAL,
)

#: Q7 disclaimer string, propagated verbatim from the Phase 12C reader status
#: dict (design §3, Q7): container PTS alignment is not physical exposure sync.
EXPOSURE_SYNC_MODEL = "container_pts_alignment_not_physical_exposure_genlock"

#: Post-pair RSS ceiling for ERR-RESOURCE (design §7.11). The design names a
#: "configured ceiling" without giving PipelineConfig a field for it; 6 GB sits
#: under the 8 GB reference machine with OS/runtime headroom (design §7.6).
DEFAULT_RSS_CEILING_MB = 6144.0

PIPELINE_ARTIFACT_SCHEMA = "dual_camera_pipeline/1.0"


class PipelineError(Exception):
    """Pipeline failure carrying one of the eight design §7.11 error codes."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown pipeline error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PairError:
    """Structured per-pair failure record (design §7.3 / §7.11)."""

    code: str
    message: str
    traceback_str: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "traceback": self.traceback_str,
        }


# --------------------------------------------------------------------------
# Configuration and result types (design §7.3 — implemented as specified).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path  # dataset root (front/, back/, metadata/)
    metadata_path: Path  # pinned synchronization_metadata.json (GAP-7)
    tracking_config_path: Path  # configs/tracking_config.yaml
    association_config_path: Path  # configs/cross_camera_association_config.yaml
    output_root: Path  # Stage J artifact root
    device: str | None = None  # detector device; models via SharedDetectorRegistry
    vehicle_conf_threshold: float = 0.18
    helmet_conf_threshold: float = 0.25
    keyframe_max_per_camera: int = 24  # appearance-frame bound (design §7.6)
    keyframe_stride: int = 3
    keyframe_max_width: int = 1280
    tick_hz: float = 60.0  # tick mode / diagnostics only
    resume: bool = True


class PairStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    FAILED_RUNTIME = "FAILED_RUNTIME"
    SKIPPED_COMPLETE = "SKIPPED_COMPLETE"


@dataclass
class PairRunResult:
    pair_id: str
    status: PairStatus
    summary: dict | None  # parsed pair_summary.json when COMPLETE
    error: PairError | None  # design §7.11 when failed


# --------------------------------------------------------------------------
# Small deterministic I/O helpers.
# --------------------------------------------------------------------------


def _repo_root() -> Path:
    """Repository root = two levels above `src/` (this file lives in src/pipeline/)."""
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(path: Path, obj: Any) -> None:
    """Write a JSON artifact with sorted keys (design §7.8)."""
    path.write_text(
        json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _dump_json_atomic(path: Path, obj: Any) -> None:
    """Atomic JSON write (temp file + rename) — used for the pair checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    _dump_json(tmp, obj)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _peak_rss_mb() -> float:
    # ru_maxrss reports in bytes on macOS (Darwin) and in KiB on Linux (the Stage J venue).
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale,
        1,
    )


# --------------------------------------------------------------------------
# SyncContext — everything Phase 12C knows about one pair (design §7.2/§7.3).
# --------------------------------------------------------------------------


class SyncContext:
    """Per-pair synchronization context built from the approved Phase 12C reader.

    Consumes exactly the Phase 12C services the design sanctions (§7.2):
    `PairSynchronizationInfo`, one `FramePTSIndex` per stream, the two
    `VideoReader`s for decode, and (in tick mode) `get_synchronized_pair` /
    `iter_synchronized_frames`. It never re-derives an offset or a residual.
    """

    def __init__(
        self,
        pair_id: str,
        config: PipelineConfig,
        reader: SynchronizedVideoReader,
        metadata_item: dict[str, Any],
        metadata_sha256: str,
    ) -> None:
        self.pair_id = pair_id
        self.config = config
        self.reader = reader
        self.sync_info = reader.sync_info
        self.metadata_item = metadata_item
        self.metadata_sha256 = metadata_sha256
        if reader.reader_front.pts_index is None or reader.reader_back.pts_index is None:
            raise PipelineError(
                ERR_DECODE,
                f"{pair_id}: VideoReader built without a PTS index (use_pts_index=False)",
            )
        self.front_index: FramePTSIndex = reader.reader_front.pts_index
        self.back_index: FramePTSIndex = reader.reader_back.pts_index

    @classmethod
    def from_reader(cls, pair_id: str, config: PipelineConfig) -> SyncContext:
        """Design §7.3: `SyncContext.from_reader(pair_id, config)`.

        The metadata path is pinned by config (GAP-7): the reader is pointed at
        the pinned file's directory and the file's SHA-256 is recorded, so the
        reader's silent fallback path can never engage unnoticed — the pinned
        file's existence is a precondition checked before construction.
        """
        metadata_sha256 = _sha256_file(config.metadata_path)
        items = json.loads(config.metadata_path.read_text(encoding="utf-8"))
        item = next((x for x in items if x.get("pair_id") == pair_id), None)
        if item is None:
            raise PipelineError(
                ERR_PRECONDITION,
                f"Pair ID '{pair_id}' not found in pinned metadata {config.metadata_path}",
            )
        try:
            reader = SynchronizedVideoReader(
                pair_id,
                data_root=config.data_root,
                metadata_dir=config.metadata_path.parent,
            )
        except FileNotFoundError as e:
            raise PipelineError(ERR_PRECONDITION, str(e)) from e
        except ValueError as e:
            raise PipelineError(ERR_PRECONDITION, str(e)) from e
        except RuntimeError as e:
            raise PipelineError(ERR_DECODE, str(e)) from e
        return cls(pair_id, config, reader, item, metadata_sha256)

    # -- sanctioned quantities (design §3) --------------------------------
    @property
    def offset_sec(self) -> float:
        """Q3: Δt_offset := offset_sec (metadata, via the reader, D1)."""
        return float(self.sync_info.global_offset_sec)

    @property
    def epsilon_sync_ms(self) -> float:
        """Q4: inter-camera sync residual uncertainty (Phase 11 estimate)."""
        return float(self.sync_info.epsilon_sync_ms)

    @property
    def drift_ms(self) -> float:
        """Eighth quantity (design §3.8): reported, never applied (D6/GAP-1)."""
        return float(self.sync_info.measured_drift_ms)

    @property
    def quality(self) -> str:
        """Phase 11 manifest quality judgement — a separate axis from Q4 (GAP-6)."""
        return str(self.sync_info.quality)

    def index_for(self, camera_id: str) -> FramePTSIndex:
        return self.front_index if camera_id == "front" else self.back_index

    def reader_for(self, camera_id: str) -> VideoReader:
        return self.reader.reader_front if camera_id == "front" else self.reader.reader_back

    def video_path_for(self, camera_id: str) -> Path:
        return (
            self.sync_info.front_path if camera_id == "front" else self.sync_info.back_path
        )

    def canonical_shift_sec(self, camera_id: str) -> float:
        """Canonicalization direction (D1): front frames shift by +Δt_offset."""
        return self.offset_sec if camera_id == "front" else 0.0

    def close(self) -> None:
        self.reader.close()


# --------------------------------------------------------------------------
# Q5 — TRUE nearest-frame (presentation) error over the overlap window.
# --------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(np.mean(arr)), 4), "max": round(float(np.max(arr)), 4)}


def compute_q5_overlap_stats(ctx: SyncContext, tick_hz: float) -> dict[str, Any]:
    """True Q5 statistics from genuine lookups over the overlap window.

    Steps the reference (back) timeline at `tick_hz` across the tightened
    overlap window and performs the reader's own lookup primitive —
    `FramePTSIndex.nearest_frame` — on each stream: back against `t_ref`,
    front against the sanctioned lookup target `t_ref - Δt_offset` (D1).
    This is the same computation `SynchronizedVideoReader.get_synchronized_pair`
    performs (minus frame decoding, which diagnostics does not need); no second
    lookup implementation exists here. The resulting mean/max are what the
    pipeline passes to Phase 12E in `TimestampQualityRecord` (GAP-4) — never
    the nominal-grid deviation.
    """
    start = float(ctx.sync_info.overlap_start_back_sec)
    end = float(ctx.sync_info.overlap_end_back_sec)
    d_front: list[float] = []
    d_back: list[float] = []
    s_front: list[float] = []
    s_back: list[float] = []
    skew: list[float] = []
    if end > start and tick_hz > 0:
        dt = 1.0 / float(tick_hz)
        n_steps = int(np.floor((end - start) / dt)) + 1
        for k in range(n_steps):
            t_ref = start + k * dt
            if t_ref > end + 1e-6:
                break
            lb = ctx.back_index.nearest_frame(t_ref)
            lf = ctx.front_index.nearest_frame(t_ref - ctx.offset_sec)
            d_back.append(float(lb.delta_pts_ms))
            s_back.append(float(lb.signed_delta_pts_ms))
            d_front.append(float(lf.delta_pts_ms))
            s_front.append(float(lf.signed_delta_pts_ms))
            # Reader's skew definition (synchronized_reader.py, get_synchronized_pair):
            # |(t_back_actual - t_front_actual) - offset_sec| * 1000, stream-normalized PTS.
            skew.append(
                abs((lb.actual_pts_norm_sec - lf.actual_pts_norm_sec) - ctx.offset_sec)
                * 1000.0
            )
    return {
        "n_samples": len(d_back),
        "tick_hz": float(tick_hz),
        "delta_pts_front_ms": _stats(d_front),
        "delta_pts_back_ms": _stats(d_back),
        "signed_delta_pts_front_ms_mean": round(float(np.mean(s_front)), 4) if s_front else 0.0,
        "signed_delta_pts_back_ms_mean": round(float(np.mean(s_back)), 4) if s_back else 0.0,
        "inter_camera_pts_skew_ms": _stats(skew),
        "definition": (
            "Q5 true nearest-frame (presentation) error: |actual_pts_norm - target| "
            "per FramePTSIndex.nearest_frame lookup over the overlap window; front "
            "target = t_ref - offset_sec, back target = t_ref (design S3/S7.9). "
            "Distinct from epsilon_sync_ms (Q4) and from nominal_grid_deviation_ms."
        ),
    }


def nominal_grid_deviation_ms(index: FramePTSIndex) -> dict[str, float]:
    """Deviation of true PTS from a uniform nominal grid, `|pts_norm - idx/effective_fps|`.

    This is the quantity Pass 1 mislabelled as `delta_pts` (design §3.9, GAP-4).
    It is reported ONLY under this name, in diagnostics, and is never fed to
    Phase 12E as if it were Q5.
    """
    n = index.frame_count
    if n == 0 or index.effective_fps <= 0:
        return {"mean": 0.0, "max": 0.0}
    grid = np.arange(n, dtype=np.float64) / float(index.effective_fps)
    dev = np.abs(index.pts_norm_sec[:n] - grid) * 1000.0
    return {"mean": round(float(np.mean(dev)), 4), "max": round(float(np.max(dev)), 4)}


# --------------------------------------------------------------------------
# Per-camera primary loop (design §7.4).
# --------------------------------------------------------------------------


@dataclass
class CameraRunOutput:
    """Result of one camera's exhaustive sequential pass (design §7.3).

    Coverage rows are streamed to `coverage_csv_path` per frame (design §7.6),
    so this object carries the row count, not an in-memory copy of the rows.
    `delta_pts_by_frame` is None by design: in the primary loop no per-frame
    lookup exists, and `TrackletBuilder`'s designed fallback substitutes the
    `TimestampQualityRecord` (true-Q5) mean per camera (GAP-4).
    """

    camera_id: str
    pipeline: CameraTrackingPipeline
    coverage_csv_path: Path
    coverage_rows: int
    detections_jsonl_path: Path
    track_frames_jsonl_path: Path
    tracks_final_path: Path
    frames_by_index: dict[int, np.ndarray]
    canonical_pts_by_frame: dict[int, float]
    delta_pts_by_frame: dict[int, float] | None
    frames_decoded: int
    frames_expected: int
    pts_assertions: int
    subtrack_state_counts: dict[str, Counter]
    canonical_shift_sec: float
    frame_dimensions: tuple[int, int] | None = None


def _observed_track_ids_at_step(step: dict[str, Any]) -> list[str]:
    """Return deterministic list of track/subtrack/orphan IDs updated on this step (`time_since_update == 0`)."""
    observed: list[str] = []
    for m in step.get("active_moto_tracks", []) or []:
        if int(m.get("time_since_update", 1)) == 0 and m.get("moto_track_id"):
            observed.append(str(m["moto_track_id"]))
        subs = m.get("rider_subtracks", {}) or {}
        for sub in (subs.values() if isinstance(subs, dict) else subs):
            if int(sub.get("time_since_update", 1)) == 0 and sub.get("subtrack_id"):
                observed.append(str(sub["subtrack_id"]))
    for o in step.get("active_orphan_tracks", []) or []:
        if int(o.get("time_since_update", 1)) == 0 and o.get("orphan_track_id"):
            observed.append(str(o["orphan_track_id"]))
    return observed


def should_retain_keyframe(
    frame_index: int,
    has_detections: bool,
    observed_track_ids: list[str],
    stored_indices: set[int],
    track_to_keyframes: dict[str, list[int]],
    keyframe_stride: int,
    keyframe_max_per_camera: int,
) -> tuple[bool, int | None]:
    """Deterministic GAP-B1 keyframe selection and bounded eviction policy.

    Returns `(should_store, evict_frame_index)`:
    1. A frame is a mandatory track-anchor keyframe whenever any track updated on
       `frame_index` currently has zero stored keyframes in `stored_indices` (so
       short tracklets off the stride grid and late-entering tracks are covered).
    2. Otherwise, a detection-bearing frame on `frame_index % max(1, keyframe_stride) == 0`
       is stored while `len(stored_indices) < keyframe_max_per_camera`.
    3. If `needs_anchor` is True and `len(stored_indices) >= keyframe_max_per_camera`,
       deterministically evict either the oldest uncoupled keyframe (not referenced by
       any track) or an interior keyframe of the track holding the most keyframes (> 1),
       strictly preserving the `keyframe_max_per_camera` memory bound.
    """
    if keyframe_max_per_camera <= 0:
        return False, None
    needs_anchor = any(
        len([f for f in track_to_keyframes.get(tid, ()) if f in stored_indices]) == 0
        for tid in observed_track_ids
    )
    stride = max(1, int(keyframe_stride))
    is_stride = bool(has_detections and (frame_index % stride == 0))

    if len(stored_indices) < keyframe_max_per_camera:
        return (needs_anchor or is_stride), None

    if not needs_anchor:
        return False, None

    # Buffer is at capacity, but a newly observed track has 0 keyframes: find an evictable frame
    referenced_counts: Counter[int] = Counter()
    for tid, f_list in track_to_keyframes.items():
        active_f = [f for f in f_list if f in stored_indices]
        for f in active_f:
            referenced_counts[f] += 1

    uncoupled = sorted(f for f in stored_indices if referenced_counts[f] == 0)
    if uncoupled:
        return True, uncoupled[0]

    # Find the track with the largest number of stored keyframes (> 1)
    best_tid: str | None = None
    best_frames: list[int] = []
    for tid in sorted(track_to_keyframes.keys()):
        active_f = sorted(f for f in track_to_keyframes[tid] if f in stored_indices)
        if len(active_f) > len(best_frames):
            best_tid = tid
            best_frames = active_f

    if best_tid is not None and len(best_frames) > 1:
        # Evict the second-most-recent frame of the most-represented track so its first and latest remain
        return True, best_frames[-2]

    return False, None


def _derived_state(aggregated_confidence: dict[str, float] | None) -> str:
    """Per-frame rider state from a Phase 10 aggregator snapshot.

    Mirrors the frozen-GT package derivation exactly (`build_package.py`):
    argmax class when its aggregated confidence is >= 0.5, else UNCERTAIN.
    """
    ac = aggregated_confidence or {}
    if not ac:
        return "UNCERTAIN"
    k = max(ac, key=ac.get)
    return k.upper() if ac[k] >= 0.5 else "UNCERTAIN"


def _track_level_state(counts: Counter) -> str:
    """Modal confident state over a subtrack's per-frame derived states.

    Mirrors the package derivation: UNCERTAIN if no confident state exists or
    HELMET / NO_HELMET tie. A violation event is never derived from a single
    ambiguous frame.
    """
    conf = {s: counts.get(s, 0) for s in ("HELMET", "NO_HELMET")}
    conf = {s: c for s, c in conf.items() if c > 0}
    if not conf:
        return "UNCERTAIN"
    top = sorted(conf.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "UNCERTAIN"
    return top[0][0]


def _tracks_final_dict(pipe: CameraTrackingPipeline) -> dict[str, Any]:
    """Terminated + still-active track objects with PTS histories.

    Same serialization as the Pass 1 runner (the pattern that produced the
    frozen GT): track `to_dict()` plus per-subtrack / per-orphan PTS histories
    of `(frame_index, pts_sec, head_bbox, head_class_name, head_confidence)`.
    """

    def moto_full(m: Any) -> dict[str, Any]:
        d = m.to_dict()
        d["rider_pts_history"] = {
            str(s_idx): [
                [
                    int(f),
                    round(float(p), 6),
                    [round(float(c), 2) for c in bb],
                    str(st),
                    round(float(cf_), 4),
                ]
                for (f, p, bb, st, cf_) in sub.pts_history
            ]
            for s_idx, sub in sorted(m.rider_subtracks.items())
        }
        return d

    def orphan_full(o: Any) -> dict[str, Any]:
        d = o.to_dict()
        d["pts_history"] = [
            [
                int(f),
                round(float(p), 6),
                [round(float(c), 2) for c in bb],
                str(st),
                round(float(cf_), 4),
            ]
            for (f, p, bb, st, cf_) in o.pts_history
        ]
        return d

    return {
        "moto_tracks": [
            moto_full(m)
            for m in list(pipe.terminated_moto_tracks)
            + list(pipe.active_moto_tracks.values())
        ],
        "orphan_tracks": [
            orphan_full(o)
            for o in list(pipe.terminated_orphan_tracks)
            + list(pipe.active_orphan_tracks.values())
        ],
        "frames_updated": pipe.frames_updated,
        "duplicate_ticks_skipped": pipe.duplicate_ticks_skipped,
    }


def run_camera(
    ctx: SyncContext,
    camera_id: str,
    detector: PerCameraDetector,
    out_dir: Path,
) -> CameraRunOutput:
    """Design §7.3/§7.4: exhaustive per-camera sequential decode + detect + track.

    Invariants (design §7.4): I1 frame ordinal == presentation index, validated
    against the container PTS count at end of stream; I2 downstream timestamps
    are always Q2 (normalized PTS), never Q1; I3 forward-only (ascending
    `get_frame_at_index` — the VideoReader cursor never resets); I4 detection
    runs at most once per `(camera, frame_index)` — by loop construction every
    physical frame is new (D3 layer 1), so `should_update_state=True` is
    always correct here; I5 per-camera state is independent until association.
    """
    config = ctx.config
    pair_id = ctx.pair_id
    index = ctx.index_for(camera_id)
    reader = ctx.reader_for(camera_id)
    shift = ctx.canonical_shift_sec(camera_id)

    try:
        pipe = CameraTrackingPipeline.from_yaml(camera_id, config.tracking_config_path)
    except Exception as e:
        raise PipelineError(ERR_TRACKING, f"{pair_id}/{camera_id}: tracker init: {e}") from e

    coverage_path = out_dir / f"{pair_id}_{camera_id}_coverage.csv"
    detections_path = out_dir / f"{pair_id}_{camera_id}_detections.jsonl"
    track_frames_path = out_dir / f"{pair_id}_{camera_id}_track_frames.jsonl"
    tracks_final_path = out_dir / f"{pair_id}_{camera_id}_tracks_final.json"

    frames_by_index: dict[int, np.ndarray] = {}
    track_to_keyframes: dict[str, list[int]] = {}
    native_frame_dimensions: tuple[int, int] | None = None
    subtrack_state_counts: dict[str, Counter] = {}
    pts_assertions = 0
    n = 0

    with open(coverage_path, "w", newline="", encoding="utf-8") as cf, open(
        detections_path, "w", encoding="utf-8"
    ) as df, open(track_frames_path, "w", encoding="utf-8") as tf:
        cw = csv.writer(cf)
        cw.writerow(
            [
                "frame_index",
                "pts_raw_sec",
                "pts_norm_sec",
                "canonical_pts_sec",
                "n_motorcycle_det",
                "n_person_det",
                "n_head_det",
                "detector_ran",
                "tracked_moto_active",
                "tracked_riders_active",
                "orphans_active",
            ]
        )
        for n in range(index.frame_count):
            frame = reader.get_frame_at_index(n)
            if frame is None:
                raise PipelineError(
                    ERR_DECODE,
                    f"{pair_id}/{camera_id}: decode failed at frame {n} "
                    f"of {index.frame_count} (VideoReader returned None)",
                )
            if native_frame_dimensions is None and frame.ndim >= 2:
                native_frame_dimensions = (int(frame.shape[1]), int(frame.shape[0]))
            pts_norm = float(index.pts_norm_sec[n])  # Q2
            pts_raw = float(index.pts_raw_sec[n])  # Q1 (coverage/forensics only)
            try:
                bundle = detector.detect_frame(frame, n, pts_norm)
            except Exception as e:
                raise PipelineError(
                    ERR_DETECTION, f"{pair_id}/{camera_id} frame {n}: {e}"
                ) from e
            if bundle.pts_timestamp_sec != pts_norm or bundle.frame_index != n:
                raise PipelineError(
                    ERR_COVERAGE,
                    f"{pair_id}/{camera_id} frame {n}: bundle PTS/index does not "
                    "match the PTS table value",
                )
            pts_assertions += 1
            try:
                step = pipe.step(bundle, should_update_state=True)
            except Exception as e:
                raise PipelineError(
                    ERR_TRACKING, f"{pair_id}/{camera_id} frame {n}: {e}"
                ) from e
            if not step.get("did_advance_state", False):
                raise PipelineError(
                    ERR_TRACKING,
                    f"{pair_id}/{camera_id} frame {n}: tracker refused a genuinely "
                    "new frame (did_advance_state=False in the primary loop)",
                )

            motos = step.get("active_moto_tracks", [])
            orphans = step.get("active_orphan_tracks", [])
            n_riders = 0
            for m in motos:
                subs = m.get("rider_subtracks", {}) or {}
                n_riders += len(subs)
                for sub in subs.values():
                    sid = sub["subtrack_id"]
                    # Genuine coupled observation only: this subtrack received a
                    # matched head on this advanced step (time_since_update == 0).
                    # Propagated / unmatched / abstained frames add no evidence.
                    if int(sub.get("time_since_update", 1)) != 0:
                        continue
                    state = _derived_state(sub.get("aggregated_confidence"))
                    subtrack_state_counts.setdefault(sid, Counter())[state] += 1

            cw.writerow(
                [
                    n,
                    f"{pts_raw:.6f}",
                    f"{pts_norm:.6f}",
                    f"{pts_norm + shift:.6f}",
                    len(bundle.motorcycles),
                    len(bundle.persons),
                    len(bundle.all_heads),
                    1,
                    len(motos),
                    n_riders,
                    len(orphans),
                ]
            )
            cf.flush()
            # Design §7.9: per physical frame, FrameDetectionBundle.to_dict() verbatim.
            df.write(json.dumps(bundle.to_dict(), sort_keys=True, default=str) + "\n")
            df.flush()
            tf.write(json.dumps(step, sort_keys=True, default=str) + "\n")
            tf.flush()

            # Appearance keyframes — Pass 1 policy + GAP-B1 track-anchor guarantee:
            # detection-bearing frames (motorcycle or head) every keyframe_stride-th
            # frame PLUS first observed frame of any track lacking a keyframe, capped
            # at keyframe_max_per_camera and downscaled to keyframe_max_width.
            obs_tids = _observed_track_ids_at_step(step)
            store_kf, evict_kf = should_retain_keyframe(
                frame_index=n,
                has_detections=bool(bundle.motorcycles or bundle.all_heads),
                observed_track_ids=obs_tids,
                stored_indices=set(frames_by_index.keys()),
                track_to_keyframes=track_to_keyframes,
                keyframe_stride=config.keyframe_stride,
                keyframe_max_per_camera=config.keyframe_max_per_camera,
            )
            if store_kf:
                if evict_kf is not None and evict_kf in frames_by_index:
                    frames_by_index.pop(evict_kf, None)
                if frame.shape[1] > config.keyframe_max_width:
                    scale = config.keyframe_max_width / float(frame.shape[1])
                    small = cv2.resize(
                        frame,
                        (
                            int(config.keyframe_max_width),
                            max(1, int(round(frame.shape[0] * scale))),
                        ),
                    )
                else:
                    small = frame.copy()
                frames_by_index[n] = small
                for tid in obs_tids:
                    track_to_keyframes.setdefault(tid, []).append(n)
            elif n in frames_by_index:
                for tid in obs_tids:
                    track_to_keyframes.setdefault(tid, []).append(n)

            if (n + 1) % 250 == 0:
                _log(f"[{pair_id}] {camera_id}: frame {n + 1}/{index.frame_count}")
            n += 1

    frames_decoded = n
    if frames_decoded != index.frame_count:
        raise PipelineError(
            ERR_COVERAGE,
            f"{pair_id}/{camera_id}: decoded {frames_decoded} != container "
            f"{index.frame_count} (design C4/I1 — possible duplicate-PTS divergence)",
        )

    _dump_json(tracks_final_path, _tracks_final_dict(pipe))
    canonical = {
        i: float(index.pts_norm_sec[i]) + shift for i in range(index.frame_count)
    }
    return CameraRunOutput(
        camera_id=camera_id,
        pipeline=pipe,
        coverage_csv_path=coverage_path,
        coverage_rows=frames_decoded,
        detections_jsonl_path=detections_path,
        track_frames_jsonl_path=track_frames_path,
        tracks_final_path=tracks_final_path,
        frames_by_index=frames_by_index,
        canonical_pts_by_frame=canonical,
        delta_pts_by_frame=None,
        frames_decoded=frames_decoded,
        frames_expected=index.frame_count,
        pts_assertions=pts_assertions,
        subtrack_state_counts=subtrack_state_counts,
        canonical_shift_sec=shift,
        frame_dimensions=native_frame_dimensions,
    )


# --------------------------------------------------------------------------
# Association (design §7.2 AssociationRun).
# --------------------------------------------------------------------------


def run_association(
    ctx: SyncContext,
    front: CameraRunOutput,
    back: CameraRunOutput,
    timestamp_quality: TimestampQualityRecord,
    engine: CrossCameraAssociationEngine,
) -> CrossCameraAssociationBatchResult:
    """Run Phase 12E over the two finished per-camera pipelines.

    Canonical PTS maps use the D1 canonicalization equation; the per-frame
    delta maps are None so the tracklet builder falls back to the true-Q5
    means carried in `timestamp_quality` (GAP-4; see CameraRunOutput).
    """
    try:
        return engine.associate_pipelines(
            pair_id=ctx.pair_id,
            front_pipeline=front.pipeline,
            back_pipeline=back.pipeline,
            timestamp_quality=timestamp_quality,
            front_frames_by_index=front.frames_by_index,
            back_frames_by_index=back.frames_by_index,
            front_canonical_pts_by_frame=front.canonical_pts_by_frame,
            back_canonical_pts_by_frame=back.canonical_pts_by_frame,
            front_delta_pts_ms_by_frame=None,
            back_delta_pts_ms_by_frame=None,
            calibrated_homography_available=False,
            front_frame_dimensions=front.frame_dimensions,
            back_frame_dimensions=back.frame_dimensions,
        )
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(ERR_ASSOCIATION, f"{ctx.pair_id}: {e}") from e


def build_association_engine(config: PipelineConfig) -> CrossCameraAssociationEngine:
    """Construct the Phase 12E engine (used for its transit-model config in diagnostics)."""
    try:
        return CrossCameraAssociationEngine.from_yaml(config.association_config_path)
    except Exception as e:
        raise PipelineError(ERR_ASSOCIATION, f"association engine init: {e}") from e


# --------------------------------------------------------------------------
# Event derivation (design §7.3 derive_pair_events, §7.9 events artifact).
#
# Semantics mirror the frozen-GT package derivation (build_package.py Part 2)
# so Stage K compares like with like: one event per association motorcycle
# result, plus one synthetic UNMATCHED single-view event per moto track that
# no result referenced; rider states merged per slot across views (conflict
# -> UNCERTAIN); event type from driver/pillion states; deterministic
# numbering by canonical start time. Stage G output is uncorrected inference,
# so provenance is AI_DIRECT/UNCERTAIN only (never MANUAL_CORRECTION), and
# plates are NOT_ASSESSED (no Phase 12F/OCR, standing restriction).
# --------------------------------------------------------------------------

_PLATE_BLOCK = {
    "plate_visibility_tier": "NOT_ASSESSED",
    "gt_plate_text_normalized": None,
    "note": (
        "Plate assessment is out of scope for the integrated pipeline "
        "(no Phase 12F/OCR); no plate text is asserted."
    ),
}


def _track_records(cam_out: CameraRunOutput) -> dict[str, dict[str, Any]]:
    """Per-moto-track event records from a finished camera pipeline."""
    records: dict[str, dict[str, Any]] = {}
    pipe = cam_out.pipeline
    tracks = list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values())
    for m in tracks:
        subs: dict[int, dict[str, Any]] = {}
        for slot, sub in sorted(m.rider_subtracks.items()):
            subs[int(slot)] = {
                "subtrack_id": sub.subtrack_id,
                "slot_role": sub.slot_role,
                # Canonical strict majority over genuine coupled observations
                # (RiderSubTrack.helmet_evidence); see TASK-03/TASK-04 design.
                "track_helmet_state": sub.track_helmet_state,
                "alert_ids": [a.alert_id for a in sub.alerts],
            }
        records[m.moto_track_id] = {
            "camera_id": cam_out.camera_id,
            "track_id": m.moto_track_id,
            "t_start_ref_sec": round(float(m.first_pts_sec) + cam_out.canonical_shift_sec, 6),
            "t_end_ref_sec": round(float(m.last_pts_sec) + cam_out.canonical_shift_sec, 6),
            "rider_subtracks": subs,
        }
    return records


def _merged_riders(
    obs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Merge per-slot rider states across views (package `merged_riders` semantics)."""
    slots: dict[int, list[tuple[str, str, dict[str, Any], dict[str, Any]]]] = {}
    for cam, rec in obs.items():
        for slot, sub in rec["rider_subtracks"].items():
            slots.setdefault(slot, []).append((cam, sub["track_helmet_state"], sub, rec))
    riders: list[dict[str, Any]] = []
    detail: dict[int, dict[str, Any]] = {}
    for slot in sorted(slots):
        entries = slots[slot]
        conf = {s for (_, s, _, _) in entries if s in ("HELMET", "NO_HELMET")}
        if conf == {"HELMET"}:
            state, prov = "HELMET", "AI_DIRECT"
        elif conf == {"NO_HELMET"}:
            state, prov = "NO_HELMET", "AI_DIRECT"
        else:
            state, prov = "UNCERTAIN", "UNCERTAIN"
        win = None
        if state == "NO_HELMET":
            starts = [
                t["t_start_ref_sec"] for (_, s, _, t) in entries if s == "NO_HELMET"
            ]
            ends = [t["t_end_ref_sec"] for (_, s, _, t) in entries if s == "NO_HELMET"]
            if starts and ends:
                win = [min(starts), max(ends)]
        role = {0: "driver", 1: "pillion_1", 2: "pillion_2"}.get(slot, f"slot_{slot}")
        riders.append(
            {
                "rider_slot": slot,
                "role": role,
                "recorded_slot_roles": sorted({e[2]["slot_role"] for e in entries}),
                "helmet_state": state,
                "helmet_state_provenance": prov,
                "violation_window_ref_sec": win,
            }
        )
        detail[slot] = {
            "state": state,
            "window": win,
            "per_view": {cam: s for (cam, s, _, _) in entries},
        }
    return riders, detail


def _event_type_for(detail: dict[int, dict[str, Any]]) -> tuple[str, bool | None]:
    """Package `event_type_for` semantics, verbatim."""
    if not detail:
        return "INSUFFICIENT_VISUAL_EVIDENCE", None
    driver = detail.get(0, {}).get("state", "UNCERTAIN")
    pillions = [d["state"] for s, d in detail.items() if s > 0]
    d_nh = driver == "NO_HELMET"
    p_nh = any(s == "NO_HELMET" for s in pillions)
    if d_nh and p_nh:
        return "BOTH_NO_HELMET", True
    if d_nh:
        return "DRIVER_NO_HELMET", True
    if p_nh:
        return "PILLION_NO_HELMET", True
    known = [driver] + pillions
    if all(s == "HELMET" for s in known):
        return "NO_VIOLATION", False
    return "INSUFFICIENT_VISUAL_EVIDENCE", None


def derive_pair_events(
    pair_id: str,
    front: CameraRunOutput,
    back: CameraRunOutput,
    batch: CrossCameraAssociationBatchResult,
) -> list[dict[str, Any]]:
    """Pure derivation of per-pair events from finished pipelines + batch result.

    Reuses the Phase 10 alerts already emitted inside tracker steps (read from
    the finished subtracks' alert lists). The standalone
    `TemporalVerificationEngine` is never instantiated (design §5.4/GAP-8).
    """
    records = {"front": _track_records(front), "back": _track_records(back)}
    covered: set[tuple[str, str]] = set()
    raw: list[dict[str, Any]] = []

    def _make_event(
        obs: dict[str, dict[str, Any]],
        outcome: str,
        score: float | None,
        reasons: list[str],
        link_status: str,
    ) -> dict[str, Any]:
        riders, detail = _merged_riders(obs)
        etype, is_viol = _event_type_for(detail)
        wins = [d["window"] for d in detail.values() if d["window"]]
        win = [min(w[0] for w in wins), max(w[1] for w in wins)] if wins else None
        alert_ids = sorted(
            {
                aid
                for rec in obs.values()
                for sub in rec["rider_subtracks"].values()
                for aid in sub["alert_ids"]
            }
        )
        return {
            "global_moto_id": None,  # numbered after deterministic sort
            "cross_camera_link_status": link_status,
            "association_outcome": outcome,
            "association_score": round(float(score), 4) if score is not None else None,
            "decision_reasons": list(reasons),
            "rider_count": len(riders),
            "event_is_violation": is_viol,
            "violation_event_type": etype,
            "violation_window_ref_sec": win,
            "riders": riders,
            "front_track_id": obs["front"]["track_id"] if "front" in obs else None,
            "back_track_id": obs["back"]["track_id"] if "back" in obs else None,
            "phase10_alert_ids": alert_ids,
            "plate_ground_truth": dict(_PLATE_BLOCK),
            "_sort": min(
                [t["t_start_ref_sec"] for t in obs.values()], default=0.0
            ),
        }

    for r in batch.motorcycle_results:
        f = records["front"].get(r.front_moto_track_id) if r.front_moto_track_id else None
        b = records["back"].get(r.back_moto_track_id) if r.back_moto_track_id else None
        if f is not None:
            covered.add(("front", f["track_id"]))
        if b is not None:
            covered.add(("back", b["track_id"]))
        obs = {c: t for c, t in (("front", f), ("back", b)) if t is not None}
        oc = r.outcome.value
        if oc == "LINKED":
            link_status = "LINKED_BOTH_VIEWS"
        elif oc == "AMBIGUOUS_ABSTAIN":
            link_status = "AMBIGUOUS_HUMAN_UNCERTAIN"
        elif f is not None and b is None:
            link_status = "FRONT_VIEW_ONLY"
        elif b is not None and f is None:
            link_status = "BACK_VIEW_ONLY"
        else:
            link_status = "AMBIGUOUS_HUMAN_UNCERTAIN"
        raw.append(
            _make_event(
                obs,
                outcome=oc,
                score=r.association_score,
                reasons=list(r.decision_reasons),
                link_status=link_status,
            )
        )

    # Moto tracks no association result referenced (e.g. filtered by the 12E
    # tracklet builder's minimum-observation rules): exactly one synthetic
    # UNMATCHED single-view event each, mirroring the package's retained-track
    # pass, so every moto track is accounted for in the events artifact.
    for cam in ("front", "back"):
        for tid, rec in records[cam].items():
            if (cam, tid) in covered:
                continue
            raw.append(
                _make_event(
                    {cam: rec},
                    outcome="UNMATCHED",
                    score=None,
                    reasons=["NO_ASSOCIATION_RESULT_FOR_TRACK"],
                    link_status="FRONT_VIEW_ONLY" if cam == "front" else "BACK_VIEW_ONLY",
                )
            )

    raw.sort(
        key=lambda e: (
            e["_sort"],
            e["front_track_id"] or "",
            e["back_track_id"] or "",
        )
    )
    events: list[dict[str, Any]] = []
    for i, ev in enumerate(raw, 1):
        ev["global_moto_id"] = f"{pair_id}_moto_{i:02d}"
        ev.pop("_sort")
        events.append(ev)
    return events


# --------------------------------------------------------------------------
# Sync diagnostics (design §7.9) — all seven §3 quantities, kept distinct.
# --------------------------------------------------------------------------


def _count_in_overlap(index: FramePTSIndex, shift: float, start: float, end: float) -> int:
    pts = index.pts_norm_sec + shift
    return int(np.sum((pts >= start - 1e-9) & (pts <= end + 1e-9)))


def build_sync_diagnostics(
    ctx: SyncContext,
    q5: dict[str, Any],
    engine: CrossCameraAssociationEngine,
) -> dict[str, Any]:
    """Per-pair synchronization diagnostics with every §3 quantity populated.

    Q1 raw PTS / Q2 normalized PTS (per-stream summaries), Q3 `offset_sec`,
    Q4 `epsilon_sync_ms`, Q5 true lookup statistics, Q6 transit-delay model
    parameters (a scene property consumed only inside Phase 12E — never a
    clock correction), Q7 exposure disclaimer verbatim. `drift_ms` is
    reported alongside Q4 (GAP-1/D6) and manifest `quality` is a separate
    field from the residual (GAP-6); the reader's hard-coded
    `accuracy_classification` label is deliberately not reused.
    """
    tc = engine.config.transit_config
    start = float(ctx.sync_info.overlap_start_back_sec)
    end = float(ctx.sync_info.overlap_end_back_sec)
    diag: dict[str, Any] = {
        "pair_id": ctx.pair_id,
        "reference_clock_anchor": "back_stream_pts",
        # Q3 / Q4 (+ the eighth quantity, drift — reported, never applied):
        "offset_sec": round(ctx.offset_sec, 6),
        "epsilon_sync_ms": round(ctx.epsilon_sync_ms, 4),
        "drift_ms": round(ctx.drift_ms, 4),
        "drift_status": ctx.metadata_item.get("drift_status"),
        "initial_offset_sec": round(float(ctx.sync_info.initial_offset_sec), 6),
        "final_offset_sec": round(float(ctx.sync_info.final_offset_sec), 6),
        "sync_quality": ctx.quality,
        "overlap": {
            "start_back_sec": round(start, 6),
            "end_back_sec": round(end, 6),
            "duration_sec": round(float(ctx.sync_info.overlap_duration_sec), 6),
            "front_frames_in_overlap": _count_in_overlap(
                ctx.front_index, ctx.offset_sec, start, end
            ),
            "back_frames_in_overlap": _count_in_overlap(ctx.back_index, 0.0, start, end),
        },
        # Q5 from true lookups (GAP-4):
        "q5_overlap_lookup_stats": q5,
        "delta_pts_field_definitions": {
            "timestamp_quality_record.mean/max_delta_pts_*_ms": (
                "Q5 true nearest-frame lookup statistics over the overlap "
                "window (see q5_overlap_lookup_stats)"
            ),
            "observed_frame_record.delta_pts_ms": (
                "TrackletBuilder fallback = the record's Q5 mean for that "
                "camera (no per-frame lookup exists in the primary loop)"
            ),
            "nominal_grid_deviation_ms": (
                "Deviation of true PTS from a uniform idx/effective_fps grid. "
                "NOT Q5; reported under its own name only (GAP-4)."
            ),
        },
        # Q6 — transit delay model (scene property, Phase 12E-internal):
        "transit_time_model": {
            "expected_transit_delay_sec": float(tc.expected_transit_delay_sec),
            "min_transit_delay_sec": float(tc.min_transit_delay_sec),
            "max_transit_delay_sec": float(tc.max_transit_delay_sec),
            "transit_sigma_sec": float(tc.transit_sigma_sec),
            "transit_uncertainty_sec": float(tc.transit_uncertainty_sec),
            "note": (
                "Q6 delta_transit_sec is a physical FOV-transit phenomenon "
                "modelled inside Phase 12E candidate generation/scoring; it is "
                "never a clock error and is never corrected by shifting "
                "timestamps (design S3)."
            ),
        },
        # Q7 — exposure synchronization is NOT established by anything in the stack:
        "exposure_sync_model": EXPOSURE_SYNC_MODEL,
        "reader_counters": {
            "front_decode_call_count": ctx.reader_for("front").decode_call_count,
            "front_cache_hit_count": ctx.reader_for("front").cache_hit_count,
            "back_decode_call_count": ctx.reader_for("back").decode_call_count,
            "back_cache_hit_count": ctx.reader_for("back").cache_hit_count,
        },
    }
    for cam in ("front", "back"):
        index = ctx.index_for(cam)
        summary = index.get_summary()  # Q1/Q2 stream timing forensics
        summary["nominal_grid_deviation_ms"] = nominal_grid_deviation_ms(index)
        summary["canonical_shift_sec"] = round(ctx.canonical_shift_sec(cam), 6)
        diag[f"{cam}_stream"] = summary
    return diag


# --------------------------------------------------------------------------
# Validation gate (design §7.9) — all checks must pass before COMPLETE.
# --------------------------------------------------------------------------


def validate_pair(
    ctx: SyncContext,
    front: CameraRunOutput,
    back: CameraRunOutput,
    batch: CrossCameraAssociationBatchResult,
    events: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    outs = {"front": front, "back": back}
    checks: dict[str, bool] = {}
    checks["coverage_complete_front"] = (
        front.frames_decoded == ctx.front_index.frame_count
    )
    checks["coverage_complete_back"] = back.frames_decoded == ctx.back_index.frame_count
    checks["detections_pts_match_table"] = all(
        o.pts_assertions == o.frames_decoded for o in outs.values()
    )
    front_tracklet_ids = {t.moto_track_id for t in batch.front_motorcycle_tracklets}
    back_tracklet_ids = {t.moto_track_id for t in batch.back_motorcycle_tracklets}
    checks["association_results_reference_known_tracklets"] = all(
        (r.front_moto_track_id is None or r.front_moto_track_id in front_tracklet_ids)
        and (r.back_moto_track_id is None or r.back_moto_track_id in back_tracklet_ids)
        for r in batch.motorcycle_results
    )
    checks["association_outcome_counts_reconcile"] = (
        sum(batch.outcome_counts.values()) == len(batch.motorcycle_results)
        and sum(batch.rider_outcome_counts.values())
        == sum(len(r.rider_associations) for r in batch.motorcycle_results)
        and sum(batch.orphan_outcome_counts.values()) == len(batch.orphan_results)
    )
    all_moto_ids = set(_track_records(front)) | set(_track_records(back))
    event_track_ids = {
        tid
        for e in events
        for tid in (e["front_track_id"], e["back_track_id"])
        if tid
    }
    checks["events_cover_all_moto_tracks"] = event_track_ids == all_moto_ids
    q5 = diagnostics.get("q5_overlap_lookup_stats", {})
    checks["sync_diagnostics_all_q_fields"] = all(
        k in diagnostics
        for k in (
            "offset_sec",
            "epsilon_sync_ms",
            "drift_ms",
            "transit_time_model",
            "exposure_sync_model",
            "front_stream",
            "back_stream",
        )
    ) and "delta_pts_front_ms" in q5 and "delta_pts_back_ms" in q5
    checks["all_passed"] = all(checks.values())
    return checks


def _validation_error_code(validation: dict[str, Any]) -> str:
    if not (
        validation["coverage_complete_front"]
        and validation["coverage_complete_back"]
        and validation["detections_pts_match_table"]
    ):
        return ERR_COVERAGE
    if not (
        validation["association_results_reference_known_tracklets"]
        and validation["association_outcome_counts_reconcile"]
        and validation["events_cover_all_moto_tracks"]
    ):
        return ERR_ASSOCIATION
    return ERR_INTERNAL


# --------------------------------------------------------------------------
# Preconditions, input hashes, checkpoint/resume (design §7.7/§7.8/§7.10).
# --------------------------------------------------------------------------


def _weights_paths(repo_root: Path) -> tuple[Path, Path]:
    """Default checkpoint paths exactly as the approved adapters resolve them."""
    veh = repo_root / VehiclePersonDetectorAdapter.DEFAULT_CHECKPOINT
    helm = repo_root / HelmetDetectorAdapter.DEFAULT_CHECKPOINT
    if not helm.exists():
        helm = repo_root / "results" / "exp_combined" / "weights" / "best.pt"
    return veh, helm


def _check_preconditions(
    pair_id: str, config: PipelineConfig, repo_root: Path
) -> PairError | None:
    """ERR-PRECONDITION gate (design §7.11): inputs invalid before any processing.

    Runs before the pair directory is created — a precondition failure writes
    NO per-pair artifacts; it is recorded in the PairRunResult / run manifest.
    """
    if config.metadata_path.name != "synchronization_metadata.json":
        return PairError(
            ERR_PRECONDITION,
            f"Pinned metadata path must be a synchronization_metadata.json file, "
            f"got {config.metadata_path}",
        )
    if not config.metadata_path.exists():
        return PairError(
            ERR_PRECONDITION, f"Pinned metadata file missing: {config.metadata_path}"
        )
    if not config.tracking_config_path.exists():
        return PairError(
            ERR_PRECONDITION, f"Tracking config missing: {config.tracking_config_path}"
        )
    if not config.association_config_path.exists():
        return PairError(
            ERR_PRECONDITION,
            f"Association config missing: {config.association_config_path}",
        )
    try:
        items = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        return PairError(ERR_PRECONDITION, f"Metadata file unparseable: {e}")
    if not any(x.get("pair_id") == pair_id for x in items):
        return PairError(
            ERR_PRECONDITION, f"Pair ID '{pair_id}' not found in synchronization metadata"
        )
    # GAP-2: there is no degraded single-camera mode — a missing camera file
    # is a per-pair fatal precondition failure, isolated to this pair.
    for cam in ("front", "back"):
        video = config.data_root / cam / f"{pair_id}_{cam}.mp4"
        if not video.exists():
            return PairError(
                ERR_PRECONDITION, f"Missing {cam} camera file for {pair_id}: {video}"
            )
    veh, helm = _weights_paths(repo_root)
    if not veh.exists() or not helm.exists():
        return PairError(
            ERR_PRECONDITION,
            f"Model weights missing (vehicle={veh}, helmet={helm}); "
            "weight verification itself is Stage I scope",
        )
    return None


def _input_hashes(pair_id: str, config: PipelineConfig, repo_root: Path) -> dict[str, str]:
    """SHA-256 of every input whose change invalidates a resume skip (§7.10)."""
    veh, helm = _weights_paths(repo_root)
    return {
        "front_video_sha256": _sha256_file(
            config.data_root / "front" / f"{pair_id}_front.mp4"
        ),
        "back_video_sha256": _sha256_file(
            config.data_root / "back" / f"{pair_id}_back.mp4"
        ),
        "metadata_sha256": _sha256_file(config.metadata_path),
        "tracking_config_sha256": _sha256_file(config.tracking_config_path),
        "association_config_sha256": _sha256_file(config.association_config_path),
        "weights_yolov8n_sha256": _sha256_file(veh),
        "weights_phase9_best_sha256": _sha256_file(helm),
    }


def _try_resume(
    pair_id: str, config: PipelineConfig, hashes: dict[str, str]
) -> PairRunResult | None:
    """Hash-validated checkpoint skip (design §7.7, hardened from Pass 1).

    Skip iff `pair_summary.json` exists, parses, says COMPLETE, passed its
    validation gate, and its recorded input hashes all match the current
    inputs; any mismatch re-runs the pair (overwriting its own directory only).
    """
    summary_path = config.output_root / pair_id / "pair_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if summary.get("status") != "COMPLETE":
        return None
    if not summary.get("validation", {}).get("all_passed", False):
        return None
    recorded = summary.get("inputs_sha256", {})
    if any(recorded.get(k) != v for k, v in hashes.items()):
        return None
    return PairRunResult(
        pair_id=pair_id, status=PairStatus.SKIPPED_COMPLETE, summary=summary, error=None
    )


def _write_status(
    out_dir: Path, pair_id: str, status: str, error: PairError | None
) -> None:
    _dump_json(
        out_dir / "pair_status.json",
        {
            "pair_id": pair_id,
            "status": status,
            "error": error.to_dict() if error is not None else None,
        },
    )


# --------------------------------------------------------------------------
# run_pair — the per-pair orchestration (design §7.2/§7.3).
# --------------------------------------------------------------------------


def run_pair(pair_id: str, config: PipelineConfig) -> PairRunResult:
    """Process one pair end to end. Never raises for pair-level failures:
    they are returned as `PairRunResult` with a design §7.11 `PairError`."""
    repo_root = _repo_root()

    pre = _check_preconditions(pair_id, config, repo_root)
    if pre is not None:
        return PairRunResult(
            pair_id=pair_id,
            status=PairStatus.FAILED_PRECONDITION,
            summary=None,
            error=pre,
        )

    hashes = _input_hashes(pair_id, config, repo_root)
    if config.resume:
        skipped = _try_resume(pair_id, config, hashes)
        if skipped is not None:
            _log(f"[{pair_id}] SKIPPED_COMPLETE (checkpoint + hashes valid)")
            return skipped

    out_dir = config.output_root / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_status(out_dir, pair_id, PairStatus.RUNNING.value, None)
    t0 = time.perf_counter()

    ctx: SyncContext | None = None
    try:
        ctx = SyncContext.from_reader(pair_id, config)

        # True Q5 over the overlap — computed before association so the
        # TimestampQualityRecord can carry it (GAP-4).
        q5 = compute_q5_overlap_stats(ctx, config.tick_hz)
        timestamp_quality = TimestampQualityRecord(
            pair_id=pair_id,
            reference_clock_anchor="back_stream_pts",
            global_offset_sec=ctx.offset_sec,
            epsilon_sync_ms=ctx.epsilon_sync_ms,
            mean_delta_pts_front_ms=q5["delta_pts_front_ms"]["mean"],
            max_delta_pts_front_ms=q5["delta_pts_front_ms"]["max"],
            mean_delta_pts_back_ms=q5["delta_pts_back_ms"]["mean"],
            max_delta_pts_back_ms=q5["delta_pts_back_ms"]["max"],
            is_in_overlap=bool(q5["n_samples"] > 0),
        )

        # Detectors share model singletons process-wide via the registry (§7.6).
        try:
            detectors = {
                cam: PerCameraDetector(
                    cam,
                    device=config.device,
                    vehicle_conf_threshold=config.vehicle_conf_threshold,
                    helmet_conf_threshold=config.helmet_conf_threshold,
                )
                for cam in ("front", "back")
            }
        except FileNotFoundError as e:
            raise PipelineError(ERR_PRECONDITION, f"detector weights: {e}") from e
        except ValueError as e:
            raise PipelineError(ERR_DETECTION, f"detector init: {e}") from e
        except Exception as e:
            raise PipelineError(ERR_DETECTION, f"detector init: {e}") from e

        # Per-camera primary loops: front before back (Pass 1 order, §7.8).
        cam_outs: dict[str, CameraRunOutput] = {}
        for cam in ("front", "back"):
            _log(f"[{pair_id}] {cam}: start")
            cam_outs[cam] = run_camera(ctx, cam, detectors[cam], out_dir)
            _log(
                f"[{pair_id}] {cam}: decoded {cam_outs[cam].frames_decoded} frames, "
                f"moto={len(cam_outs[cam].pipeline.terminated_moto_tracks) + len(cam_outs[cam].pipeline.active_moto_tracks)}"
            )
            gc.collect()

        engine = build_association_engine(config)
        batch = run_association(
            ctx, cam_outs["front"], cam_outs["back"], timestamp_quality, engine
        )
        _dump_json(out_dir / f"{pair_id}_association.json", batch.to_dict())

        events = derive_pair_events(pair_id, cam_outs["front"], cam_outs["back"], batch)
        _dump_json(out_dir / f"{pair_id}_events.json", events)

        diagnostics = build_sync_diagnostics(ctx, q5, engine)
        _dump_json(out_dir / f"{pair_id}_sync_diagnostics.json", diagnostics)

        validation = validate_pair(
            ctx, cam_outs["front"], cam_outs["back"], batch, events, diagnostics
        )
        if not validation["all_passed"]:
            failed = [k for k, v in validation.items() if v is False]
            raise PipelineError(
                _validation_error_code(validation),
                f"{pair_id}: validation gate failed: {failed}",
            )

        peak_rss = _peak_rss_mb()
        if peak_rss > DEFAULT_RSS_CEILING_MB:
            raise PipelineError(
                ERR_RESOURCE,
                f"{pair_id}: peak RSS {peak_rss} MB exceeds ceiling "
                f"{DEFAULT_RSS_CEILING_MB} MB (design §7.11)",
            )

        wall = round(time.perf_counter() - t0, 2)
        counts = {
            "front_moto_tracks": len(cam_outs["front"].pipeline.terminated_moto_tracks)
            + len(cam_outs["front"].pipeline.active_moto_tracks),
            "back_moto_tracks": len(cam_outs["back"].pipeline.terminated_moto_tracks)
            + len(cam_outs["back"].pipeline.active_moto_tracks),
            "front_orphan_tracks": len(
                cam_outs["front"].pipeline.terminated_orphan_tracks
            )
            + len(cam_outs["front"].pipeline.active_orphan_tracks),
            "back_orphan_tracks": len(cam_outs["back"].pipeline.terminated_orphan_tracks)
            + len(cam_outs["back"].pipeline.active_orphan_tracks),
            "association_outcome_counts": dict(batch.outcome_counts),
            "rider_outcome_counts": dict(batch.rider_outcome_counts),
            "orphan_outcome_counts": dict(batch.orphan_outcome_counts),
            "events_total": len(events),
            "events_by_type": dict(
                Counter(e["violation_event_type"] for e in events)
            ),
        }
        artifacts = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        summary = {
            "pair_id": pair_id,
            "status": PairStatus.COMPLETE.value,
            "artifact_schema": PIPELINE_ARTIFACT_SCHEMA,
            "validation": validation,
            "coverage": {
                cam: {
                    "frames_expected_container": cam_outs[cam].frames_expected,
                    "frames_decoded": cam_outs[cam].frames_decoded,
                    "coverage_complete": bool(
                        cam_outs[cam].frames_decoded == cam_outs[cam].frames_expected
                    ),
                }
                for cam in ("front", "back")
            },
            "counts": counts,
            "sync": {
                "offset_sec": round(ctx.offset_sec, 6),
                "epsilon_sync_ms": round(ctx.epsilon_sync_ms, 4),
                "drift_ms": round(ctx.drift_ms, 4),
                "quality": ctx.quality,
            },
            "inputs_sha256": hashes,
            "metadata_sha256": ctx.metadata_sha256,
            "artifacts": artifacts,
            # Runtime fields — the design §7.8 determinism exemption:
            "wall_time_sec": wall,
            "peak_rss_mb": peak_rss,
        }
        _dump_json_atomic(out_dir / "pair_summary.json", summary)
        _write_status(out_dir, pair_id, PairStatus.COMPLETE.value, None)
        _log(f"[{pair_id}] COMPLETE wall={wall}s peak_rss={peak_rss}MB")
        return PairRunResult(
            pair_id=pair_id, status=PairStatus.COMPLETE, summary=summary, error=None
        )
    except PipelineError as e:
        err = PairError(e.code, e.message, traceback.format_exc())
        _write_status(out_dir, pair_id, "FAILED", err)
        status = (
            PairStatus.FAILED_PRECONDITION
            if e.code == ERR_PRECONDITION
            else PairStatus.FAILED_RUNTIME
        )
        return PairRunResult(pair_id=pair_id, status=status, summary=None, error=err)
    except Exception as e:  # ERR-INTERNAL: anything else (design §7.11)
        err = PairError(ERR_INTERNAL, f"{type(e).__name__}: {e}", traceback.format_exc())
        _write_status(out_dir, pair_id, "FAILED", err)
        return PairRunResult(
            pair_id=pair_id, status=PairStatus.FAILED_RUNTIME, summary=None, error=err
        )
    finally:
        if ctx is not None:
            ctx.close()


# --------------------------------------------------------------------------
# Tick mode (secondary; design §7.2/§7.5) — for diagnostics and Stage H
# equivalence tests (S-NEW-1..4). Stage J uses the primary loop.
# --------------------------------------------------------------------------


def run_pair_tick_mode(pair_id: str, config: PipelineConfig) -> dict[str, Any]:
    """Iterate the overlap on the reference tick grid with dedup gating.

    Each camera is routed through `DeduplicatedTemporalGate.step_camera`
    (D3 layer 2): the wrapped update (detect + tracker step) executes ONLY on
    a genuinely new `(camera, frame_index)` observation; on repeated ticks the
    cached result is returned, detection is not re-run, and temporal state
    does not advance. The tracker `step()` guard is layer 3. Returns the
    collected per-camera detection records and gate/tracker counters so
    Stage H can assert unique updates == physical frames.

    Raises PipelineError on precondition failure (diagnostic entry point).
    """
    repo_root = _repo_root()
    pre = _check_preconditions(pair_id, config, repo_root)
    if pre is not None:
        raise PipelineError(pre.code, pre.message)

    ctx = SyncContext.from_reader(pair_id, config)
    try:
        detectors = {
            cam: PerCameraDetector(
                cam,
                device=config.device,
                vehicle_conf_threshold=config.vehicle_conf_threshold,
                helmet_conf_threshold=config.helmet_conf_threshold,
            )
            for cam in ("front", "back")
        }
        pipes = {
            cam: CameraTrackingPipeline.from_yaml(cam, config.tracking_config_path)
            for cam in ("front", "back")
        }
        gate = DeduplicatedTemporalGate()
        detections: dict[str, dict[int, dict[str, Any]]] = {"front": {}, "back": {}}
        ticks = 0
        for frame_back, frame_front, status in ctx.reader.iter_synchronized_frames(
            tick_hz=config.tick_hz, overlap_only=True
        ):
            ticks += 1
            for cam, frame in (("back", frame_back), ("front", frame_front)):
                idx = int(status[f"frame_idx_{cam}"])
                is_new = (
                    bool(status[f"is_new_{cam}_frame"]) and frame is not None and idx >= 0
                )

                def _update(
                    cam: str = cam,
                    frame: Any = frame,
                    idx: int = idx,
                    status: dict[str, Any] = status,
                ) -> dict[str, Any]:
                    pts = float(status[f"t_{cam}_actual_pts_sec"])
                    bundle = detectors[cam].detect_frame(frame, idx, pts)
                    pipes[cam].step(bundle, should_update_state=True)
                    detections[cam][idx] = bundle.to_dict()
                    return detections[cam][idx]

                gate.step_camera(cam, is_new, _update)
        return {
            "pair_id": pair_id,
            "mode": "tick",
            "ticks": ticks,
            "gate_counters": {
                "front_ticks_seen": gate.front_ticks_seen,
                "back_ticks_seen": gate.back_ticks_seen,
                "front_unique_updates": gate.front_unique_updates,
                "back_unique_updates": gate.back_unique_updates,
                "front_skipped_duplicates": gate.front_skipped_duplicates,
                "back_skipped_duplicates": gate.back_skipped_duplicates,
            },
            "front": {
                "detections": detections["front"],
                "frames_updated": pipes["front"].frames_updated,
                "duplicate_ticks_skipped": pipes["front"].duplicate_ticks_skipped,
            },
            "back": {
                "detections": detections["back"],
                "frames_updated": pipes["back"].frames_updated,
                "duplicate_ticks_skipped": pipes["back"].duplicate_ticks_skipped,
            },
        }
    finally:
        ctx.close()


# --------------------------------------------------------------------------
# run_all — batch driver with per-pair subprocess isolation (design §7.7).
# --------------------------------------------------------------------------


def _serialize_config(config: PipelineConfig) -> dict[str, Any]:
    return {
        "data_root": str(config.data_root),
        "metadata_path": str(config.metadata_path),
        "tracking_config_path": str(config.tracking_config_path),
        "association_config_path": str(config.association_config_path),
        "output_root": str(config.output_root),
        "device": config.device,
        "vehicle_conf_threshold": config.vehicle_conf_threshold,
        "helmet_conf_threshold": config.helmet_conf_threshold,
        "keyframe_max_per_camera": config.keyframe_max_per_camera,
        "keyframe_stride": config.keyframe_stride,
        "keyframe_max_width": config.keyframe_max_width,
        "tick_hz": config.tick_hz,
        "resume": config.resume,
    }


def _config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        data_root=Path(d["data_root"]),
        metadata_path=Path(d["metadata_path"]),
        tracking_config_path=Path(d["tracking_config_path"]),
        association_config_path=Path(d["association_config_path"]),
        output_root=Path(d["output_root"]),
        device=d.get("device"),
        vehicle_conf_threshold=float(d.get("vehicle_conf_threshold", 0.18)),
        helmet_conf_threshold=float(d.get("helmet_conf_threshold", 0.25)),
        keyframe_max_per_camera=int(d.get("keyframe_max_per_camera", 24)),
        keyframe_stride=int(d.get("keyframe_stride", 3)),
        keyframe_max_width=int(d.get("keyframe_max_width", 1280)),
        tick_hz=float(d.get("tick_hz", 60.0)),
        resume=bool(d.get("resume", True)),
    )


def run_all(pair_ids: list[str], config: PipelineConfig) -> list[PairRunResult]:
    """Batch driver: one subprocess per pair, failure isolation (design §7.7).

    Pairs run in ascending id order. A pair's exception, OOM-kill, or
    precondition failure is recorded and the batch proceeds. Writes
    `run_manifest.json` (the one artifact allowed wall-clock values) and a
    Pass 1-format driver log under `output_root`.
    """
    repo_root = _repo_root()
    config.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = config.output_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_json = config.output_root / "run_config.json"
    _dump_json(config_json, _serialize_config(config))
    log_path = config.output_root / "run_log.txt"

    def _driver_log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        _log(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    results: list[PairRunResult] = []
    for pair_id in sorted(pair_ids):
        _driver_log(f"{pair_id}: START")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.pipeline.dual_camera_pipeline",
                "--worker",
                pair_id,
                "--config-json",
                str(config_json),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        (logs_dir / f"{pair_id}_stdout.log").write_text(
            proc.stdout[-20000:], encoding="utf-8"
        )
        (logs_dir / f"{pair_id}_stderr.log").write_text(
            proc.stderr[-20000:], encoding="utf-8"
        )
        result: PairRunResult | None = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("PAIR_RESULT_JSON "):
                payload = json.loads(line[len("PAIR_RESULT_JSON ") :])
                err = None
                if payload.get("error"):
                    err = PairError(
                        payload["error"]["code"], payload["error"]["message"], None
                    )
                result = PairRunResult(
                    pair_id=pair_id,
                    status=PairStatus(payload["status"]),
                    summary=None,
                    error=err,
                )
                break
        if result is None:
            # Worker died without reporting (e.g. OOM-kill): synthesize from
            # the sidecar when present, else classify by exit condition.
            status_path = config.output_root / pair_id / "pair_status.json"
            sidecar_err: PairError | None = None
            if status_path.exists():
                try:
                    sidecar = json.loads(status_path.read_text(encoding="utf-8"))
                    if sidecar.get("error"):
                        sidecar_err = PairError(
                            sidecar["error"]["code"],
                            sidecar["error"]["message"],
                            sidecar["error"].get("traceback"),
                        )
                except Exception:
                    sidecar_err = None
            if sidecar_err is not None:
                st = (
                    PairStatus.FAILED_PRECONDITION
                    if sidecar_err.code == ERR_PRECONDITION
                    else PairStatus.FAILED_RUNTIME
                )
                result = PairRunResult(pair_id, st, None, sidecar_err)
            elif proc.returncode in (-9, 137):
                result = PairRunResult(
                    pair_id,
                    PairStatus.FAILED_RUNTIME,
                    None,
                    PairError(
                        ERR_RESOURCE,
                        f"worker killed (returncode {proc.returncode}); "
                        "probable resource exhaustion",
                    ),
                )
            else:
                result = PairRunResult(
                    pair_id,
                    PairStatus.FAILED_RUNTIME,
                    None,
                    PairError(
                        ERR_INTERNAL,
                        f"worker exited rc={proc.returncode} without a result; "
                        f"see _logs/{pair_id}_stderr.log",
                    ),
                )
        if result.status in (PairStatus.COMPLETE, PairStatus.SKIPPED_COMPLETE):
            summary_path = config.output_root / pair_id / "pair_summary.json"
            if summary_path.exists():
                try:
                    result.summary = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    result.summary = None
        results.append(result)
        _driver_log(f"{pair_id}: {result.status.value}")

    veh, helm = _weights_paths(repo_root)
    manifest = {
        "pipeline": "src.pipeline.dual_camera_pipeline",
        "artifact_schema": PIPELINE_ARTIFACT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": _serialize_config(config),
        "input_hashes": {
            "metadata_sha256": _sha256_file(config.metadata_path)
            if config.metadata_path.exists()
            else None,
            "tracking_config_sha256": _sha256_file(config.tracking_config_path)
            if config.tracking_config_path.exists()
            else None,
            "association_config_sha256": _sha256_file(config.association_config_path)
            if config.association_config_path.exists()
            else None,
            "weights_yolov8n_sha256": _sha256_file(veh) if veh.exists() else None,
            "weights_phase9_best_sha256": _sha256_file(helm) if helm.exists() else None,
        },
        "pairs": [
            {
                "pair_id": r.pair_id,
                "status": r.status.value,
                "error_code": r.error.code if r.error else None,
            }
            for r in results
        ],
        "counts": dict(Counter(r.status.value for r in results)),
    }
    _dump_json_atomic(config.output_root / "run_manifest.json", manifest)
    _driver_log(
        "DRIVER FINISHED "
        + " ".join(f"{k}={v}" for k, v in sorted(manifest["counts"].items()))
    )
    return results


# --------------------------------------------------------------------------
# CLI: `python -m src.pipeline.dual_camera_pipeline ...`
#   --worker PAIR --config-json PATH   (internal; used by run_all)
#   run mode: --pairs ... --output-root ... (defaults = the full corpus)
# --------------------------------------------------------------------------


def _default_config(output_root: Path, device: str | None, resume: bool) -> PipelineConfig:
    repo_root = _repo_root()
    data_root = repo_root / "data" / "dual_camera_test_v1"
    return PipelineConfig(
        data_root=data_root,
        metadata_path=data_root / "metadata" / "synchronization_metadata.json",
        tracking_config_path=repo_root / "configs" / "tracking_config.yaml",
        association_config_path=repo_root
        / "configs"
        / "cross_camera_association_config.yaml",
        output_root=output_root,
        device=device,
        resume=resume,
    )


def _all_pair_ids(config: PipelineConfig) -> list[str]:
    items = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    return sorted(x["pair_id"] for x in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dual_camera_pipeline",
        description="Integrated dual-camera helmet-violation pipeline (Stage G).",
    )
    parser.add_argument("--worker", metavar="PAIR_ID", default=None)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--tracking-config", default=None)
    parser.add_argument("--association-config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    if args.worker:
        if not args.config_json:
            print("PAIR_RESULT_JSON " + json.dumps({
                "pair_id": args.worker,
                "status": PairStatus.FAILED_RUNTIME.value,
                "error": {"code": ERR_INTERNAL, "message": "--config-json required"},
            }))
            return 2
        config = _config_from_dict(
            json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        )
        result = run_pair(args.worker, config)
        print(
            "PAIR_RESULT_JSON "
            + json.dumps(
                {
                    "pair_id": result.pair_id,
                    "status": result.status.value,
                    "error": result.error.to_dict() if result.error else None,
                }
            ),
            flush=True,
        )
        return 0

    repo_root = _repo_root()
    output_root = (
        Path(args.output_root)
        if args.output_root
        else repo_root / "results" / "phase12_integrated"
    )
    config = _default_config(output_root, args.device, resume=not args.no_resume)
    if args.data_root or args.metadata or args.tracking_config or args.association_config:
        config = PipelineConfig(
            data_root=Path(args.data_root) if args.data_root else config.data_root,
            metadata_path=Path(args.metadata) if args.metadata else config.metadata_path,
            tracking_config_path=Path(args.tracking_config)
            if args.tracking_config
            else config.tracking_config_path,
            association_config_path=Path(args.association_config)
            if args.association_config
            else config.association_config_path,
            output_root=config.output_root,
            device=config.device,
            vehicle_conf_threshold=config.vehicle_conf_threshold,
            helmet_conf_threshold=config.helmet_conf_threshold,
            keyframe_max_per_camera=config.keyframe_max_per_camera,
            keyframe_stride=config.keyframe_stride,
            keyframe_max_width=config.keyframe_max_width,
            tick_hz=config.tick_hz,
            resume=config.resume,
        )
    pair_ids = args.pairs if args.pairs else _all_pair_ids(config)
    results = run_all(pair_ids, config)
    n_ok = sum(
        1
        for r in results
        if r.status in (PairStatus.COMPLETE, PairStatus.SKIPPED_COMPLETE)
    )
    _log(f"BATCH FINISHED ok={n_ok}/{len(results)}")
    # Batch exit status (design §7.7): nonzero only if ZERO pairs completed
    # (or the run-level config itself was invalid, handled by argparse above).
    return 0 if n_ok > 0 else 1


__all__ = [
    "PipelineConfig",
    "PairStatus",
    "PairRunResult",
    "PairError",
    "PipelineError",
    "SyncContext",
    "CameraRunOutput",
    "run_pair",
    "run_all",
    "run_pair_tick_mode",
    "run_camera",
    "run_association",
    "derive_pair_events",
    "compute_q5_overlap_stats",
    "nominal_grid_deviation_ms",
    "build_sync_diagnostics",
    "validate_pair",
    "ERROR_CODES",
    "ERR_PRECONDITION",
    "ERR_DECODE",
    "ERR_COVERAGE",
    "ERR_DETECTION",
    "ERR_TRACKING",
    "ERR_ASSOCIATION",
    "ERR_RESOURCE",
    "ERR_INTERNAL",
    "EXPOSURE_SYNC_MODEL",
]


if __name__ == "__main__":
    raise SystemExit(main())
