"""Synchronized Dual-Camera Video Stream Reader.
Provides temporal alignment access for paired 4K 60fps video streams based on audited Phase 11.6 metadata.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator
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
    residual_uncertainty_ms: float
    accuracy_classification: str
    quality: str
    overlap_duration_sec: float
    overlap_start_back_sec: float
    overlap_end_back_sec: float


class SynchronizedVideoReader:
    """
    Reader for paired dual-camera streams maintaining strictly audited temporal alignment.
    
    Coordinate conventions:
    - Reference timeline: Back camera timeline (t_back).
    - Front camera timeline: t_front = t_back - global_offset_sec.
    - Common overlap interval on Back timeline: [overlap_start_back_sec, overlap_end_back_sec].
    """

    def __init__(
        self,
        pair_id: str,
        data_root: str | Path = "data/dual_camera_test_v1",
        metadata_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.pair_id = pair_id

        # Resolve metadata path
        meta_dir = Path(metadata_dir) if metadata_dir else self.data_root / "metadata"
        meta_json = meta_dir / "synchronization_metadata.json"

        if not meta_json.exists():
            # Fallback to results directory
            meta_json = Path("results/phase11_dual_camera/synchronization_metadata.json")
            if not meta_json.exists():
                raise FileNotFoundError(f"Synchronization metadata not found at {meta_json}")

        self.sync_info = self._load_pair_metadata(meta_json, pair_id)

        # Open video streams
        self.reader_back = VideoReader(self.sync_info.back_path)
        self.reader_front = VideoReader(self.sync_info.front_path)

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
        resid_ms = float(match.get("residual_error_ms", 3.0))
        quality = match.get("quality", "HIGH")
        overlap_dur = float(match.get("overlap_duration", 0.0))
        dur_b = float(match.get("back_duration", 0.0))
        dur_f = float(match.get("front_duration", 0.0))

        # Overlap interval on Back timeline
        overlap_start = max(0.0, glob_off)
        overlap_end = min(dur_b, glob_off + dur_f)

        cls = "Sub-Frame (<= 10.0 ms)" if resid_ms <= 10.0 else "Near-Frame (18.0-38.0 ms)"

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
            residual_uncertainty_ms=resid_ms,
            accuracy_classification=cls,
            quality=quality,
            overlap_duration_sec=overlap_dur,
            overlap_start_back_sec=overlap_start,
            overlap_end_back_sec=overlap_end,
        )

    def get_synchronized_pair(
        self,
        t_back_sec: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """
        Extract synchronized frames from Back and Front streams at time t_back_sec.
        
        Args:
            t_back_sec: Timestamp on the Back camera timeline.
            
        Returns:
            (frame_back, frame_front, sync_status_dict)
        """
        # Front time calculated via global offset
        t_front_sec = t_back_sec - self.sync_info.global_offset_sec

        frame_b, idx_b = self.reader_back.get_frame_at_time(t_back_sec)
        frame_f, idx_f = self.reader_front.get_frame_at_time(t_front_sec)

        in_overlap = (
            self.sync_info.overlap_start_back_sec <= t_back_sec <= self.sync_info.overlap_end_back_sec
        )

        status = {
            "pair_id": self.pair_id,
            "t_back_sec": round(t_back_sec, 4),
            "t_front_sec": round(t_front_sec, 4),
            "frame_idx_back": idx_b,
            "frame_idx_front": idx_f,
            "global_offset_sec": self.sync_info.global_offset_sec,
            "residual_uncertainty_ms": self.sync_info.residual_uncertainty_ms,
            "accuracy_classification": self.sync_info.accuracy_classification,
            "is_in_overlap": in_overlap,
            "valid_pair": (frame_b is not None and frame_f is not None),
        }

        return frame_b, frame_f, status

    def iter_synchronized_frames(
        self,
        step_frames: int = 1,
        overlap_only: bool = True,
    ) -> Generator[tuple[np.ndarray, np.ndarray, dict[str, Any]], None, None]:
        """
        Iterate sequentially through synchronized frames across the common overlap interval.
        Yields (frame_back, frame_front, status_dict).
        """
        fps = self.reader_back.fps
        dt = 1.0 / fps if fps > 0 else 1.0 / 60.0

        if overlap_only:
            t_start = self.sync_info.overlap_start_back_sec
            t_end = self.sync_info.overlap_end_back_sec
        else:
            t_start = 0.0
            t_end = self.reader_back.duration_sec

        t_curr = t_start
        while t_curr <= t_end:
            fb, ff, status = self.get_synchronized_pair(t_curr)
            if fb is not None and ff is not None:
                yield fb, ff, status
            t_curr += dt * step_frames

    def create_side_by_side_preview(
        self,
        frame_back: np.ndarray,
        frame_front: np.ndarray,
        status: dict[str, Any],
        target_height: int = 480,
    ) -> np.ndarray:
        """Create an annotated side-by-side composite for visual verification."""
        # Calculate aspect-ratio preserved width
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

        # Draw banner text
        header_text = (
            f"[{self.pair_id.upper()}] Back: t={status['t_back_sec']:.3f}s (f#{status['frame_idx_back']}) | "
            f"Front: t={status['t_front_sec']:.3f}s (f#{status['frame_idx_front']}) | "
            f"Offset: {status['global_offset_sec']:+.3f}s | Uncertainty: ±{status['residual_uncertainty_ms']:.1f}ms"
        )
        cv2.putText(comp, header_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return comp

    def close(self) -> None:
        self.reader_back.close()
        self.reader_front.close()

    def __enter__(self) -> SynchronizedVideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
