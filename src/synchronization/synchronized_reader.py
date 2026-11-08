"""Synchronized Dual-Camera Video Stream Reader (Phase 12C PTS-Indexed Engine).

Provides continuous-time reference timeline stepping, exact container Presentation
Timestamp (PTS) nearest-frame retrieval, VFR/30-60fps frame deduplication gating,
and strict metrological separation between:
  1. epsilon_sync_ms: Inter-camera synchronization residual uncertainty (Phase 11.6).
  2. delta_pts_front_ms: Front-camera container PTS discretization offset |t_front,actual - t_front,target|.
  3. delta_pts_back_ms: Back-camera container PTS discretization offset |t_back,actual - t_back,target|.

Scientific Caveat:
Container timestamp alignment aligns demuxed presentation timestamps to the estimated
recording offset (global_offset_sec). It does NOT establish physical CMOS sensor
exposure genlock or rolling-shutter readout synchronization.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Generator
import cv2
import numpy as np

from src.video.video_reader import VideoReader


@dataclass
class PairSynchronizationInfo:
    """Audited metadata for a synchronized dual-camera pair."""
    pair_id: str
    front_file: str
    back_file: str
    front_path: Path
    back_path: Path
    initial_offset_sec: float
    final_offset_sec: float
    global_offset_sec: float
    measured_drift_ms: float
    epsilon_sync_ms: float
    residual_uncertainty_ms: float  # Backward-compatible alias for epsilon_sync_ms
    accuracy_classification: str
    quality: str
    overlap_duration_sec: float
    overlap_start_back_sec: float
    overlap_end_back_sec: float


@dataclass(frozen=True)
class SyncedFramePair:
    """Strongly typed synchronized dual-camera frame pair with separated uncertainty terms."""
    pair_id: str
    tick_index: int
    t_ref_sec: float
    t_back_target_sec: float
    t_front_target_sec: float
    t_back_actual_pts_sec: float
    t_front_actual_pts_sec: float
    t_back_raw_pts_sec: float
    t_front_raw_pts_sec: float
    frame_idx_back: int
    frame_idx_front: int
    delta_pts_back_ms: float
    delta_pts_front_ms: float
    signed_delta_pts_back_ms: float
    signed_delta_pts_front_ms: float
    inter_camera_pts_skew_ms: float
    epsilon_sync_ms: float
    global_offset_sec: float
    is_new_back_frame: bool
    is_new_front_frame: bool
    should_update_back_state: bool
    should_update_front_state: bool
    is_in_overlap: bool
    valid_pair: bool
    frame_back: np.ndarray | None
    frame_front: np.ndarray | None


class DeduplicatedTemporalGate:
    """
    Guard ensuring that repeated frames from a lower-FPS or VFR stream (e.g., 30 fps
    Back camera stepped on a 60 Hz reference timeline) do NOT trigger duplicate
    object detection or duplicate Phase 10 temporal state-machine updates.
    """

    def __init__(self) -> None:
        self.front_ticks_seen: int = 0
        self.back_ticks_seen: int = 0
        self.front_unique_updates: int = 0
        self.back_unique_updates: int = 0
        self.front_skipped_duplicates: int = 0
        self.back_skipped_duplicates: int = 0
        self._last_front_result: Any = None
        self._last_back_result: Any = None

    def step_camera(
        self,
        camera: str,
        is_new_frame: bool,
        update_fn: Callable[[], Any],
    ) -> tuple[Any, bool]:
        """
        Execute update_fn() ONLY if is_new_frame is True; otherwise return cached
        state from the last unique physical frame without advancing the state machine.
        Returns (result, did_execute_update).
        """
        if camera == "front":
            self.front_ticks_seen += 1
            if is_new_frame:
                self.front_unique_updates += 1
                self._last_front_result = update_fn()
                return self._last_front_result, True
            self.front_skipped_duplicates += 1
            return self._last_front_result, False
        elif camera == "back":
            self.back_ticks_seen += 1
            if is_new_frame:
                self.back_unique_updates += 1
                self._last_back_result = update_fn()
                return self._last_back_result, True
            self.back_skipped_duplicates += 1
            return self._last_back_result, False
        else:
            raise ValueError(f"Unknown camera identifier: {camera}")


class SynchronizedVideoReader:
    """
    PTS-indexed reader for paired dual-camera streams maintaining audited temporal alignment.

    Coordinate conventions:
    - Reference timeline (t_ref): Anchored to Back camera stream-normalized PTS timeline (t_back).
    - Front camera target time: t_front_target = t_ref - global_offset_sec.
    - Common overlap interval on reference timeline: [overlap_start_back_sec, overlap_end_back_sec].
    - Metrological separation:
        * epsilon_sync_ms: Phase 11.6 inter-camera synchronization residual uncertainty.
        * delta_pts_front_ms: |t_front_actual_pts - t_front_target| * 1000.
        * delta_pts_back_ms: |t_back_actual_pts - t_back_target| * 1000.
    """

    def __init__(
        self,
        pair_id: str,
        data_root: str | Path = "data/dual_camera_test_v1",
        metadata_dir: str | Path | None = None,
        use_pts_index: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.pair_id = pair_id
        self.use_pts_index = use_pts_index

        meta_dir = Path(metadata_dir) if metadata_dir else self.data_root / "metadata"
        meta_json = meta_dir / "synchronization_metadata.json"

        if not meta_json.exists():
            meta_json = Path("results/phase11_dual_camera/synchronization_metadata.json")
            if not meta_json.exists():
                raise FileNotFoundError(f"Synchronization metadata not found at {meta_json}")

        self.sync_info = self._load_pair_metadata(meta_json, pair_id)

        # Open PTS-indexed video streams
        self.reader_back = VideoReader(self.sync_info.back_path, use_pts_index=use_pts_index)
        self.reader_front = VideoReader(self.sync_info.front_path, use_pts_index=use_pts_index)

        # Tighten overlap upper bound to actual stream-normalized PTS durations
        true_overlap_end = min(
            self.reader_back.duration_sec,
            self.sync_info.global_offset_sec + self.reader_front.duration_sec,
        )
        self.sync_info.overlap_end_back_sec = max(
            self.sync_info.overlap_start_back_sec,
            min(self.sync_info.overlap_end_back_sec, true_overlap_end),
        )
        self.sync_info.overlap_duration_sec = max(
            0.0,
            self.sync_info.overlap_end_back_sec - self.sync_info.overlap_start_back_sec,
        )

        # Track previous frame indices for duplicate-frame suppression
        self._last_frame_idx_back: int = -1
        self._last_frame_idx_front: int = -1
        self._tick_counter: int = 0

    def _load_pair_metadata(self, meta_json: Path, target_pair_id: str) -> PairSynchronizationInfo:
        with open(meta_json, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)

        match = None
        for item in pairs_data:
            if item.get("pair_id") == target_pair_id:
                match = item
                break

        if match is None:
            raise ValueError(f"Pair ID '{target_pair_id}' not found in synchronization metadata.")

        glob_off = float(match["offset_sec"])
        init_off = float(match.get("initial_offset_sec", glob_off))
        fin_off = float(match.get("final_offset_sec", glob_off))
        drift_ms = float(match.get("drift_ms", 0.0))
        eps_sync_ms = float(match.get("residual_error_ms", 3.0))
        quality = match.get("quality", "HIGH")
        overlap_dur = float(match.get("overlap_duration", 0.0))
        dur_b = float(match.get("back_duration", 0.0))
        dur_f = float(match.get("front_duration", 0.0))

        overlap_start = max(0.0, glob_off)
        overlap_end = min(dur_b, glob_off + dur_f)

        cls = "Sub-Frame (<= 10.0 ms)" if eps_sync_ms <= 10.0 else "Near-Frame (18.0-38.0 ms)"

        front_p = self.data_root / "front" / f"{target_pair_id}_front.mp4"
        back_p = self.data_root / "back" / f"{target_pair_id}_back.mp4"

        return PairSynchronizationInfo(
            pair_id=target_pair_id,
            front_file=f"{target_pair_id}_front.mp4",
            back_file=f"{target_pair_id}_back.mp4",
            front_path=front_p,
            back_path=back_p,
            initial_offset_sec=init_off,
            final_offset_sec=fin_off,
            global_offset_sec=glob_off,
            measured_drift_ms=drift_ms,
            epsilon_sync_ms=eps_sync_ms,
            residual_uncertainty_ms=eps_sync_ms,
            accuracy_classification=cls,
            quality=quality,
            overlap_duration_sec=overlap_dur,
            overlap_start_back_sec=overlap_start,
            overlap_end_back_sec=overlap_end,
        )

    def reset_deduplication_state(self) -> None:
        """Reset tracking of previously yielded frame indices."""
        self._last_frame_idx_back = -1
        self._last_frame_idx_front = -1
        self._tick_counter = 0

    def get_synchronized_pair(
        self,
        t_back_sec: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """
        Extract synchronized frames from Back and Front streams at reference time t_back_sec.
        Maintains strict separation between epsilon_sync_ms, delta_pts_front_ms, and delta_pts_back_ms.
        """
        t_ref_sec = float(t_back_sec)
        t_front_target_sec = t_ref_sec - self.sync_info.global_offset_sec

        frame_b, lookup_b = self.reader_back.get_frame_at_pts(t_ref_sec)
        frame_f, lookup_f = self.reader_front.get_frame_at_pts(t_front_target_sec)

        idx_b = lookup_b.frame_index if lookup_b is not None and frame_b is not None else -1
        idx_f = lookup_f.frame_index if lookup_f is not None and frame_f is not None else -1

        valid_back = (frame_b is not None and lookup_b is not None)
        valid_front = (frame_f is not None and lookup_f is not None)

        eps = 1e-4
        in_overlap = (
            (self.sync_info.overlap_start_back_sec - eps)
            <= t_ref_sec
            <= (self.sync_info.overlap_end_back_sec + eps)
        )

        # Complete synchronized pair validity requires BOTH frames to exist AND t_ref inside common overlap
        valid_pair = bool(valid_back and valid_front and in_overlap)

        # Decoupled per-camera PTS diagnostics and deduplication eligibility:
        # Each camera independently tracks its own new physical frames even if the other camera is unavailable.
        if valid_back and lookup_b is not None:
            t_b_actual = lookup_b.actual_pts_norm_sec
            t_b_raw = lookup_b.actual_pts_raw_sec
            delta_b_ms = lookup_b.delta_pts_ms
            s_delta_b_ms = lookup_b.signed_delta_pts_ms
            is_new_b = (idx_b != self._last_frame_idx_back)
            self._last_frame_idx_back = idx_b
        else:
            t_b_actual = -1.0
            t_b_raw = -1.0
            delta_b_ms = -1.0
            s_delta_b_ms = 0.0
            is_new_b = False

        if valid_front and lookup_f is not None:
            t_f_actual = lookup_f.actual_pts_norm_sec
            t_f_raw = lookup_f.actual_pts_raw_sec
            delta_f_ms = lookup_f.delta_pts_ms
            s_delta_f_ms = lookup_f.signed_delta_pts_ms
            is_new_f = (idx_f != self._last_frame_idx_front)
            self._last_frame_idx_front = idx_f
        else:
            t_f_actual = -1.0
            t_f_raw = -1.0
            delta_f_ms = -1.0
            s_delta_f_ms = 0.0
            is_new_f = False

        if valid_back and valid_front:
            inter_skew_ms = abs((t_b_actual - t_f_actual) - self.sync_info.global_offset_sec) * 1000.0
        else:
            inter_skew_ms = -1.0

        status: dict[str, Any] = {
            "pair_id": self.pair_id,
            "tick_index": self._tick_counter,
            "t_ref_sec": round(t_ref_sec, 6),
            "t_back_sec": round(t_ref_sec, 6),
            "t_front_sec": round(t_front_target_sec, 6),
            "t_back_target_sec": round(t_ref_sec, 6),
            "t_front_target_sec": round(t_front_target_sec, 6),
            "t_back_actual_pts_sec": round(t_b_actual, 6),
            "t_front_actual_pts_sec": round(t_f_actual, 6),
            "t_back_raw_pts_sec": round(t_b_raw, 6),
            "t_front_raw_pts_sec": round(t_f_raw, 6),
            "frame_idx_back": idx_b,
            "frame_idx_front": idx_f,
            "delta_pts_back_ms": round(delta_b_ms, 4),
            "delta_pts_front_ms": round(delta_f_ms, 4),
            "signed_delta_pts_back_ms": round(s_delta_b_ms, 4),
            "signed_delta_pts_front_ms": round(s_delta_f_ms, 4),
            "inter_camera_pts_skew_ms": round(inter_skew_ms, 4),
            "epsilon_sync_ms": self.sync_info.epsilon_sync_ms,
            "residual_uncertainty_ms": self.sync_info.residual_uncertainty_ms,
            "global_offset_sec": self.sync_info.global_offset_sec,
            "accuracy_classification": self.sync_info.accuracy_classification,
            "valid_back_frame": valid_back,
            "valid_front_frame": valid_front,
            "is_new_back_frame": is_new_b,
            "is_new_front_frame": is_new_f,
            "should_update_back_state": bool(is_new_b and valid_back),
            "should_update_front_state": bool(is_new_f and valid_front),
            "exposure_sync_model": "container_pts_alignment_not_physical_exposure_genlock",
            "is_in_overlap": in_overlap,
            "valid_pair": valid_pair,
        }
        self._tick_counter += 1
        return frame_b, frame_f, status

    def iter_synchronized_frames(
        self,
        step_frames: int = 1,
        overlap_only: bool = True,
        tick_hz: float = 60.0,
    ) -> Generator[tuple[np.ndarray, np.ndarray, dict[str, Any]], None, None]:
        """
        Iterate sequentially through synchronized frames across the common overlap interval
        on a canonical reference clock (default 60.0 Hz, matching the 60 fps Front camera rate).

        Repeated frames on lower-rate or VFR streams (e.g., 30 fps Back streams in pair_015-018
        or VFR segments in pair_013-014) are served from VideoReader's zero-cost frame cache
        with `is_new_back_frame=False` and `should_update_back_state=False`.
        """
        self.reset_deduplication_state()
        base_hz = tick_hz if tick_hz > 0 else 60.0
        dt = (1.0 / base_hz) * max(1, step_frames)

        if overlap_only:
            t_start = self.sync_info.overlap_start_back_sec
            t_end = self.sync_info.overlap_end_back_sec
        else:
            t_start = 0.0
            t_end = self.reader_back.duration_sec

        n_steps = int(np.floor((t_end - t_start + 1e-9) / dt)) + 1
        for step_idx in range(max(0, n_steps)):
            t_curr = t_start + step_idx * dt
            if t_curr > t_end + 1e-6:
                break
            fb, ff, status = self.get_synchronized_pair(t_curr)
            if overlap_only:
                if status["valid_pair"] and fb is not None and ff is not None:
                    yield fb, ff, status
            else:
                if status["valid_back_frame"] or status["valid_front_frame"]:
                    yield fb, ff, status

    def create_side_by_side_preview(
        self,
        frame_back: np.ndarray,
        frame_front: np.ndarray,
        status: dict[str, Any],
        target_height: int = 480,
    ) -> np.ndarray:
        """Create an annotated side-by-side composite displaying separated PTS and sync residuals."""
        h_b, w_b = frame_back.shape[:2]
        h_f, w_f = frame_front.shape[:2]

        w_b_new = int(w_b * (target_height / h_b))
        w_f_new = int(w_f * (target_height / h_f))

        b_resized = cv2.resize(frame_back, (w_b_new, target_height))
        f_resized = cv2.resize(frame_front, (w_f_new, target_height))

        banner_h = 50
        comp = np.zeros((target_height + banner_h, w_b_new + w_f_new, 3), dtype=np.uint8)
        comp[banner_h:, :w_b_new] = b_resized
        comp[banner_h:, w_b_new:] = f_resized

        header_text = (
            f"[{self.pair_id.upper()}] B:f#{status['frame_idx_back']}(dPTS={status.get('delta_pts_back_ms', 0.0):.1f}ms) | "
            f"F:f#{status['frame_idx_front']}(dPTS={status.get('delta_pts_front_ms', 0.0):.1f}ms) | "
            f"Off:{status['global_offset_sec']:+.3f}s | eps_sync:±{status.get('epsilon_sync_ms', status['residual_uncertainty_ms']):.1f}ms"
        )
        cv2.putText(comp, header_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        return comp

    def close(self) -> None:
        self.reader_back.close()
        self.reader_front.close()

    def __enter__(self) -> SynchronizedVideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
