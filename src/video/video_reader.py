"""Standardized Video Reader using OpenCV for frame extraction and timestamp mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generator
import cv2
import numpy as np


class VideoReader:
    """
    OpenCV-based video stream reader providing random-access frame retrieval,
    timestamp calculation, and sequential iteration.
    """

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file with OpenCV: {self.video_path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0

    def get_metadata(self) -> dict[str, Any]:
        """Return comprehensive video stream metadata."""
        return {
            "path": str(self.video_path),
            "filename": self.video_path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 3),
        }

    def get_frame_at_index(self, frame_index: int) -> np.ndarray | None:
        """
        Seek to and read frame at specified 0-based frame index.
        Returns BGR numpy array or None if out of bounds.
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        return frame if ret else None

    def get_frame_at_time(self, timestamp_sec: float) -> tuple[np.ndarray | None, int]:
        """
        Seek to frame nearest to the given timestamp in seconds.
        Returns (frame, frame_index).
        """
        if timestamp_sec < 0.0 or timestamp_sec > self.duration_sec:
            return None, -1

        target_frame = int(round(timestamp_sec * self.fps))
        target_frame = min(target_frame, self.frame_count - 1)
        frame = self.get_frame_at_index(target_frame)
        return frame, target_frame

    def iter_frames(
        self,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        step_frames: int = 1,
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Iterate over frames in a specified time window.
        Yields (frame_index, timestamp_sec, frame_bgr).
        """
        start_idx = max(0, int(round(start_sec * self.fps)))
        max_idx = self.frame_count if end_sec is None else min(self.frame_count, int(round(end_sec * self.fps)))

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        current_idx = start_idx

        while current_idx < max_idx:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (current_idx - start_idx) % step_frames == 0:
                t_sec = current_idx / self.fps if self.fps > 0 else 0.0
                yield current_idx, t_sec, frame

            current_idx += 1

    def close(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
