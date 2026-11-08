"""Timestamp and Frame Conversion Utilities."""
from __future__ import annotations


def frame_to_seconds(frame_index: int, fps: float = 60.0) -> float:
    """Convert frame index to seconds timestamp."""
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    return frame_index / fps


def seconds_to_frame(timestamp_sec: float, fps: float = 60.0) -> int:
    """Convert seconds timestamp to nearest frame index."""
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    return int(round(timestamp_sec * fps))


def format_timestamp(timestamp_sec: float) -> str:
    """Format seconds timestamp to MM:SS.mmm string."""
    mins = int(timestamp_sec // 60)
    secs = timestamp_sec % 60
    return f"{mins:02d}:{secs:06.3f}"


def calculate_overlap_interval(
    duration_a: float,
    duration_b: float,
    offset_sec: float,
) -> tuple[float, float, float]:
    """
    Calculate the common overlap interval on stream A's timeline.
    Stream B begins at offset_sec on stream A's timeline.
    
    Returns:
        (overlap_start_sec, overlap_end_sec, overlap_duration_sec) on stream A's timeline.
    """
    overlap_start = max(0.0, offset_sec)
    overlap_end = min(duration_a, offset_sec + duration_b)
    overlap_dur = max(0.0, overlap_end - overlap_start)
    return overlap_start, overlap_end, overlap_dur
