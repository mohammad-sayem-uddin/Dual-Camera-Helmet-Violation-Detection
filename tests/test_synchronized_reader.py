"""Synchronized Video Reader Unit Test Suite."""
from pathlib import Path
import pytest

from src.synchronization.synchronized_reader import SynchronizedVideoReader
from src.video.frame_timestamp import frame_to_seconds, seconds_to_frame, calculate_overlap_interval


def test_frame_timestamp_conversions():
    """Verify frame index and timestamp conversion utilities."""
    assert frame_to_seconds(60, fps=60.0) == 1.0
    assert seconds_to_frame(1.5, fps=60.0) == 90

    start, end, dur = calculate_overlap_interval(duration_a=5.0, duration_b=4.0, offset_sec=1.0)
    assert start == 1.0
    assert end == 5.0
    assert dur == 4.0


def test_synchronized_video_reader_metadata():
    """Verify SynchronizedVideoReader correctly parses metadata for pair_001."""
    with SynchronizedVideoReader("pair_001") as reader:
        info = reader.sync_info
        assert info.pair_id == "pair_001"
        assert info.global_offset_sec == -0.205
        assert info.accuracy_classification == "Sub-Frame (<= 10.0 ms)"
        assert info.overlap_duration_sec == pytest.approx(3.102, abs=0.01)
        assert info.front_path.exists()
        assert info.back_path.exists()


def test_synchronized_frame_extraction():
    """Verify SynchronizedVideoReader extracts valid frames from both streams."""
    with SynchronizedVideoReader("pair_001") as reader:
        t_mid = (reader.sync_info.overlap_start_back_sec + reader.sync_info.overlap_end_back_sec) / 2.0
        fb, ff, status = reader.get_synchronized_pair(t_mid)

        assert fb is not None, "Back frame should not be None."
        assert ff is not None, "Front frame should not be None."
        assert fb.shape[0] == 2160 and fb.shape[1] == 3840
        assert ff.shape[0] == 2160 and ff.shape[1] == 3840
        assert status["is_in_overlap"] is True
        assert status["valid_pair"] is True

        comp = reader.create_side_by_side_preview(fb, ff, status, target_height=240)
        assert comp.shape[0] == 240 + 50  # height + banner
        assert comp.shape[2] == 3
