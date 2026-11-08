"""Standardized PTS-Indexed Video Reader using OpenCV and Rational PTS Tables.

Provides frame-accurate sequential forward grabbing, exact Presentation Timestamp (PTS)
lookup via binary search, cached retrieval for repeated frames (e.g., 30 fps streams
stepped at 60 Hz), and VFR-safe random access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generator
import cv2
import numpy as np

from src.synchronization.pts_index import FramePTSIndex, PTSLookupResult


class VideoReader:
    """
    PTS-indexed OpenCV video stream reader with VFR-safe sequential forward grabbing
    and zero-cost frame caching for repeated timestamp queries.
    """

    def __init__(self, video_path: str | Path, use_pts_index: bool = True) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file with OpenCV: {self.video_path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.use_pts_index = use_pts_index

        if use_pts_index:
            self.pts_index = FramePTSIndex.from_video(self.video_path)
            self.frame_count = self.pts_index.frame_count
            self.duration_sec = self.pts_index.duration_norm_sec
            self.effective_fps = self.pts_index.effective_fps
            self.is_vfr = self.pts_index.is_vfr
        else:
            self.pts_index = None  # type: ignore[assignment]
            self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0
            self.effective_fps = self.fps
            self.is_vfr = False

        # Sequential decoder cursor state (eliminates OpenCV CAP_PROP_POS_FRAMES VFR seek bug)
        self._cursor_idx: int = -1
        self._cached_frame_idx: int = -1
        self._cached_frame: np.ndarray | None = None
        self._last_decoded_pos_msec: float = 0.0
        self.decode_call_count: int = 0
        self.cache_hit_count: int = 0

    def _reset_decoder(self) -> None:
        """Re-initialize decoder at stream start (Frame -1 before first grab)."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to reopen video file: {self.video_path}")
        self._cursor_idx = -1
        self._cached_frame_idx = -1
        self._cached_frame = None
        self._last_decoded_pos_msec = 0.0

    def get_metadata(self) -> dict[str, Any]:
        """Return comprehensive video stream metadata including VFR and PTS diagnostics."""
        meta: dict[str, Any] = {
            "path": str(self.video_path),
            "filename": self.video_path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "effective_fps": round(self.effective_fps, 3),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 6),
            "is_vfr": self.is_vfr,
        }
        if self.pts_index is not None:
            meta.update({
                "time_base": f"{self.pts_index.time_base_num}/{self.pts_index.time_base_den}",
                "start_time_raw_sec": round(self.pts_index.start_time_raw_sec, 6),
                "min_interval_ms": round(self.pts_index.min_interval_ms, 3),
                "median_interval_ms": round(self.pts_index.median_interval_ms, 3),
                "max_interval_ms": round(self.pts_index.max_interval_ms, 3),
            })
        return meta

    def get_frame_at_index(self, frame_index: int) -> np.ndarray | None:
        """
        Retrieve frame at specified 0-based presentation frame index using VFR-safe
        sequential forward grabbing and single-frame caching.
        Returns BGR numpy array or None if out of bounds.
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        # Instant cache hit when a 30 fps frame is queried on consecutive 60 Hz ticks
        if frame_index == self._cached_frame_idx and self._cached_frame is not None:
            self.cache_hit_count += 1
            return self._cached_frame

        # If target frame is behind current cursor, reset decoder to start of stream
        if frame_index < self._cursor_idx:
            self._reset_decoder()

        # Advance cursor via lightweight sequential grab() up to target frame_index
        while self._cursor_idx < frame_index:
            ret = self._cap.grab()
            if not ret:
                return None
            self._cursor_idx += 1

        # Decode and retrieve only the target frame
        ret, frame = self._cap.retrieve()
        if not ret or frame is None:
            return None

        self._last_decoded_pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        self._cached_frame_idx = frame_index
        self._cached_frame = frame
        self.decode_call_count += 1
        return frame

    def get_frame_at_pts(self, timestamp_sec: float) -> tuple[np.ndarray | None, PTSLookupResult | None]:
        """
        Retrieve frame whose stream-normalized PTS is nearest to timestamp_sec,
        along with exact PTS lookup diagnostics (delta_pts_ms, actual_pts_norm_sec, etc.).
        """
        eps = 1e-4
        if timestamp_sec < -eps or timestamp_sec > self.duration_sec + eps:
            return None, None

        clamped_t = float(min(max(0.0, timestamp_sec), self.duration_sec))
        if self.pts_index is not None:
            lookup = self.pts_index.nearest_frame(clamped_t)
            # Recompute delta against original requested timestamp_sec
            signed_delta_ms = (lookup.actual_pts_norm_sec - timestamp_sec) * 1000.0
            lookup = PTSLookupResult(
                frame_index=lookup.frame_index,
                target_norm_sec=float(timestamp_sec),
                actual_pts_norm_sec=lookup.actual_pts_norm_sec,
                actual_pts_raw_sec=lookup.actual_pts_raw_sec,
                pts_ticks=lookup.pts_ticks,
                delta_pts_ms=abs(signed_delta_ms),
                signed_delta_pts_ms=signed_delta_ms,
                local_frame_interval_ms=lookup.local_frame_interval_ms,
            )
            frame = self.get_frame_at_index(lookup.frame_index)
            return frame, lookup

        # Fallback if use_pts_index=False
        target_frame = min(max(0, int(round(clamped_t * self.fps))), self.frame_count - 1)
        actual_sec = target_frame / self.fps if self.fps > 0 else 0.0
        signed_delta_ms = (actual_sec - timestamp_sec) * 1000.0
        fallback_lookup = PTSLookupResult(
            frame_index=target_frame,
            target_norm_sec=float(timestamp_sec),
            actual_pts_norm_sec=actual_sec,
            actual_pts_raw_sec=actual_sec,
            pts_ticks=int(round(actual_sec * 1_000_000)),
            delta_pts_ms=abs(signed_delta_ms),
            signed_delta_pts_ms=signed_delta_ms,
            local_frame_interval_ms=1000.0 / self.fps if self.fps > 0 else 16.6667,
        )
        frame = self.get_frame_at_index(target_frame)
        return frame, fallback_lookup

    def get_frame_at_time(self, timestamp_sec: float) -> tuple[np.ndarray | None, int]:
        """
        Retrieve frame nearest to the given stream-normalized timestamp in seconds.
        Returns (frame, frame_index).
        """
        frame, lookup = self.get_frame_at_pts(timestamp_sec)
        if frame is None or lookup is None:
            return None, -1
        return frame, lookup.frame_index

    def iter_frames(
        self,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        step_frames: int = 1,
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Iterate sequentially over unique physical frames in a specified time window
        using exact container PTS timestamps.
        Yields (frame_index, actual_pts_norm_sec, frame_bgr).
        """
        if self.pts_index is not None:
            start_lookup = self.pts_index.nearest_frame(max(0.0, start_sec))
            start_idx = start_lookup.frame_index
            if end_sec is None:
                max_idx = self.frame_count
            else:
                end_lookup = self.pts_index.nearest_frame(min(self.duration_sec, end_sec))
                max_idx = min(self.frame_count, end_lookup.frame_index + 1)
        else:
            start_idx = max(0, int(round(start_sec * self.fps)))
            max_idx = (
                self.frame_count
                if end_sec is None
                else min(self.frame_count, int(round(end_sec * self.fps)))
            )

        for idx in range(start_idx, max_idx, max(1, step_frames)):
            frame = self.get_frame_at_index(idx)
            if frame is None:
                break
            t_sec = (
                float(self.pts_index.pts_norm_sec[idx])
                if self.pts_index is not None
                else (idx / self.fps if self.fps > 0 else 0.0)
            )
            yield idx, t_sec, frame

    def close(self) -> None:
        self._cached_frame = None
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
