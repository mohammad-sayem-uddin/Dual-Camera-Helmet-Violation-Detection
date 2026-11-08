"""Phase 12C Unit & Forensic Integration Test Suite:
PTS-Indexed Synchronized Dual-Camera Frame Reader and VFR/30-60 FPS Alignment Engine.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.synchronization.pts_index import (
    FramePTSIndex,
    validate_pts_against_opencv_sequential,
)
from src.synchronization.synchronized_reader import (
    DeduplicatedTemporalGate,
    SynchronizedVideoReader,
)
from src.temporal.confidence_aggregator import EMAAggregator
from src.temporal.state_machine import StateMachineConfig, TemporalState, TemporalStateMachine
from src.video.video_reader import VideoReader


def test_pts_index_vfr_detection_and_nominal_error_on_pair_013_and_014():
    """
    Verify FramePTSIndex detects VFR throttling on pair_013_back and pair_014_back
    and quantifies the massive error of nominal frame_idx / avg_fps calculation.
    """
    idx_13 = FramePTSIndex.from_video("data/dual_camera_test_v1/back/pair_013_back.mp4")
    assert idx_13.is_vfr is True
    assert idx_13.frame_count == 608
    assert idx_13.count_dt_60fps > 100
    assert idx_13.count_dt_30fps > 100
    err_13 = idx_13.nominal_fps_timestamp_error_stats()
    # Nominal frame_idx / avg_fps drifts by ~2,527 ms on pair_013_back
    assert err_13["max_error_ms"] == pytest.approx(2527.07, abs=5.0)

    idx_14 = FramePTSIndex.from_video("data/dual_camera_test_v1/back/pair_014_back.mp4")
    assert idx_14.is_vfr is True
    assert idx_14.frame_count == 276
    err_14 = idx_14.nominal_fps_timestamp_error_stats()
    # Nominal frame_idx / avg_fps drifts by ~826 ms on pair_014_back
    assert err_14["max_error_ms"] == pytest.approx(826.01, abs=5.0)


def test_validate_pos_msec_against_decoded_frames_and_pos_frames_bug():
    """
    Validate that sequential OpenCV CAP_PROP_POS_MSEC matches stream-normalized
    ffprobe rational PTS to < 0.01 ms, whereas random OpenCV CAP_PROP_POS_FRAMES
    seek fails by > 2,800 ms on VFR pair_013_back.mp4.
    """
    val_13 = validate_pts_against_opencv_sequential(
        "data/dual_camera_test_v1/back/pair_013_back.mp4"
    )
    assert val_13["frame_count_exact_match"] is True
    assert val_13["seq_pos_msec_vs_norm_pts_max_diff_ms"] < 0.01
    # OpenCV CAP_PROP_POS_FRAMES seek at midpoint frame 304 lands ~2.8 seconds off!
    assert val_13["pos_frames_seek_error_at_mid_ms"] > 2700.0

    # Verify Android front stream raw start_time offset vs normalized 0.0s anchor
    val_04f = validate_pts_against_opencv_sequential(
        "data/dual_camera_test_v1/front/pair_004_front.mp4"
    )
    assert val_04f["frame_count_exact_match"] is True
    assert val_04f["ffprobe_start_time_raw_sec"] == pytest.approx(0.0725, abs=0.001)
    assert val_04f["opencv_frame0_pos_sec"] == 0.0
    assert val_04f["seq_pos_msec_vs_norm_pts_max_diff_ms"] < 0.01


def test_vfr_video_reader_sequential_grab_eliminates_seek_error():
    """
    Prove that VideoReader.get_frame_at_index(300) on VFR pair_013_back.mp4
    retrieves the exact frame at 5005.0 ms (0.0 ms decoder drift), and caches
    repeated requests without re-decoding.
    """
    with VideoReader("data/dual_camera_test_v1/back/pair_013_back.mp4") as reader:
        frame_300 = reader.get_frame_at_index(300)
        assert frame_300 is not None
        # Decoded POS_MSEC must match true PTS[300] == 5005.0 ms
        assert reader._last_decoded_pos_msec == pytest.approx(5005.0, abs=0.05)
        decodes_after_first = reader.decode_call_count
        assert reader.cache_hit_count == 0

        # Querying frame 300 again must hit the cache without touching the decoder
        frame_300_cached = reader.get_frame_at_index(300)
        assert frame_300_cached is frame_300
        assert reader.decode_call_count == decodes_after_first
        assert reader.cache_hit_count == 1

        # Querying by timestamp 5.005s must retrieve frame 300 with ~0.0 ms delta_pts
        _, lookup = reader.get_frame_at_pts(5.005)
        assert lookup is not None
        assert lookup.frame_index == 300
        assert lookup.delta_pts_ms < 0.01


def test_separated_uncertainty_terms_and_exposure_disclaimer():
    """
    Verify SynchronizedVideoReader keeps epsilon_sync_ms, delta_pts_front_ms,
    and delta_pts_back_ms strictly separate and includes the exposure disclaimer.
    """
    with SynchronizedVideoReader("pair_008") as sync_reader:
        t_mid = (
            sync_reader.sync_info.overlap_start_back_sec
            + sync_reader.sync_info.overlap_end_back_sec
        ) / 2.0
        fb, ff, status = sync_reader.get_synchronized_pair(t_mid)

        assert fb is not None and ff is not None
        # Near-frame pair_008 has epsilon_sync_ms = 38.0 ms
        assert status["epsilon_sync_ms"] == 38.0
        assert "delta_pts_front_ms" in status
        assert "delta_pts_back_ms" in status
        assert "inter_camera_pts_skew_ms" in status
        # Container PTS discretization is bounded by half a 60fps frame (~8.34 ms),
        # completely distinct from epsilon_sync_ms (38.0 ms)
        assert 0.0 <= status["delta_pts_front_ms"] <= 8.5
        assert 0.0 <= status["delta_pts_back_ms"] <= 8.5
        assert status["epsilon_sync_ms"] != status["delta_pts_front_ms"]
        assert (
            status["exposure_sync_model"]
            == "container_pts_alignment_not_physical_exposure_genlock"
        )


def test_measured_non_zero_discretization_errors_on_vfr_and_30fps_pairs():
    """
    Verify that stepping at 60 Hz across a 30 fps Back stream (pair_015) and VFR
    Back stream (pair_013) produces honest, non-zero delta_pts_back_ms bounded by
    half the local frame interval (<= 16.7 ms).
    """
    with SynchronizedVideoReader("pair_015") as reader_15:
        deltas_back = []
        deltas_front = []
        new_back_flags = []
        new_front_flags = []

        # Step first 30 ticks (0.5 sec at 60 Hz) without decoding full 4K pixels via metadata check
        t_start = reader_15.sync_info.overlap_start_back_sec
        for k in range(30):
            t_ref = t_start + k / 60.0
            t_front = t_ref - reader_15.sync_info.global_offset_sec
            lb = reader_15.reader_back.pts_index.nearest_frame(t_ref)
            lf = reader_15.reader_front.pts_index.nearest_frame(t_front)
            deltas_back.append(lb.delta_pts_ms)
            deltas_front.append(lf.delta_pts_ms)
            new_back_flags.append(lb.frame_index)
            new_front_flags.append(lf.frame_index)

        # Back camera is 30 fps, so stepping at 60 Hz MUST have non-zero delta_pts_back_ms
        assert np.mean(deltas_back) > 2.0
        assert np.max(deltas_back) <= 17.0
        # Unique back frames over 30 ticks at 60 Hz should be ~15-16 (half of 30)
        unique_back = len(set(new_back_flags))
        unique_front = len(set(new_front_flags))
        assert 14 <= unique_back <= 17
        assert 29 <= unique_front <= 30


def test_repeated_30fps_frame_does_not_trigger_duplicate_detection_or_temporal_update():
    """
    Mandatory Condition 4 & Review Correction 3:
    Use the production SynchronizedVideoReader.get_synchronized_pair() path,
    production status["is_new_back_frame"], production status["should_update_back_state"],
    DeduplicatedTemporalGate, and Phase 10 TemporalStateMachine on pair_015.
    Verify that 8 canonical 60 Hz ticks produce exactly 4 unique Back-camera
    detector/state-machine updates.
    """
    cfg = StateMachineConfig(min_observation_frames=3, confirm_frames=5, confirm_threshold=0.70)
    sm_gated = TemporalStateMachine(track_id=1, config=cfg)
    agg_gated = EMAAggregator(window_size=10, alpha=0.40)

    sm_ungated_bug = TemporalStateMachine(track_id=1, config=cfg)
    agg_ungated_bug = EMAAggregator(window_size=10, alpha=0.40)

    gate = DeduplicatedTemporalGate()
    detector_calls_back = 0
    detector_calls_front = 0

    with SynchronizedVideoReader("pair_015") as sync_reader:
        t_start = sync_reader.sync_info.overlap_start_back_sec

        # Execute 8 canonical 60 Hz ticks through production get_synchronized_pair()
        for tick in range(8):
            t_ref = t_start + tick / 60.0
            fb, ff, status = sync_reader.get_synchronized_pair(t_ref)

            assert fb is not None and ff is not None
            assert status["is_in_overlap"] is True
            assert status["valid_pair"] is True
            assert status["valid_back_frame"] is True
            assert status["valid_front_frame"] is True
            assert status["should_update_back_state"] == status["is_new_back_frame"]
            assert status["should_update_front_state"] == status["is_new_front_frame"]

            # 1. Production-gated Back camera update
            def _run_back_step():
                nonlocal detector_calls_back
                detector_calls_back += 1
                agg_gated.add_observation("no_helmet", 0.85, is_detected=True)
                conf = agg_gated.get_aggregated_confidence()
                state, _ = sm_gated.update(status["frame_idx_back"], conf, is_detected=True)
                return state

            gated_state, did_update_b = gate.step_camera(
                "back", status["should_update_back_state"], _run_back_step
            )
            assert did_update_b == status["is_new_back_frame"]

            # Production-gated Front camera update (60 fps -> updates on all 8 ticks)
            def _run_front_step():
                nonlocal detector_calls_front
                detector_calls_front += 1
                return status["frame_idx_front"]

            _, did_update_f = gate.step_camera(
                "front", status["should_update_front_state"], _run_front_step
            )
            assert did_update_f is True

            # 2. Ungated bug simulation (updates on every 60 Hz tick even when 30fps frame repeats)
            agg_ungated_bug.add_observation("no_helmet", 0.85, is_detected=True)
            conf_bug = agg_ungated_bug.get_aggregated_confidence()
            ungated_state, _ = sm_ungated_bug.update(tick, conf_bug, is_detected=True)

    # Over 8 canonical 60 Hz ticks, Front (60 fps) updated 8 times,
    # while Back (30 fps) updated EXACTLY 4 times (skipping 4 duplicate ticks):
    assert gate.front_ticks_seen == 8
    assert gate.front_unique_updates == 8
    assert detector_calls_front == 8

    assert gate.back_ticks_seen == 8
    assert gate.back_unique_updates == 4
    assert detector_calls_back == 4
    assert gate.back_skipped_duplicates == 4

    # Because only 4 unique physical Back frames occurred (< confirm_frames=5),
    # the gated state machine remains in NO_HELMET_SUSPECTED,
    # whereas the ungated bug prematurely escalated to NO_HELMET_CONFIRMED!
    assert gated_state == TemporalState.NO_HELMET_SUSPECTED
    assert sm_gated.consecutive_no_helmet_frames == 4
    assert ungated_state == TemporalState.NO_HELMET_CONFIRMED
    assert sm_ungated_bug.consecutive_no_helmet_frames == 8


def test_decoupled_per_camera_update_eligibility_and_overlap_validity():
    """
    Review Corrections 1, 2 & 3:
    1. Verify that when one camera's frame is missing (outside overlap), valid_pair is False,
       the missing camera has should_update_<cam>_state=False, and the available camera still
       has should_update_<cam>_state=True on new physical frames and False on repeated frames.
    2. Verify that if both frames exist at a timestamp outside the common overlap window,
       is_in_overlap is False and valid_pair is False (never treating out-of-overlap frames
       as valid synchronized evidence), while per-camera update eligibility remains active.
    """
    # Case A: pair_003 has global_offset_sec = +0.665s, overlap_start_back_sec = 0.665s.
    # At t_back = 0.10s, Back camera has a valid frame, but Front target time is -0.565s (None).
    with SynchronizedVideoReader("pair_003") as reader:
        fb1, ff1, st1 = reader.get_synchronized_pair(0.10)
        assert fb1 is not None
        assert ff1 is None
        assert st1["is_in_overlap"] is False
        assert st1["valid_pair"] is False
        assert st1["valid_back_frame"] is True
        assert st1["valid_front_frame"] is False
        assert st1["is_new_back_frame"] is True
        assert st1["should_update_back_state"] is True
        assert st1["is_new_front_frame"] is False
        assert st1["should_update_front_state"] is False

        # Querying 1 ms later (0.101s) hits the SAME physical Back frame -> deduplicated!
        fb2, ff2, st2 = reader.get_synchronized_pair(0.101)
        assert fb2 is not None
        assert ff2 is None
        assert st2["frame_idx_back"] == st1["frame_idx_back"]
        assert st2["valid_pair"] is False
        assert st2["valid_back_frame"] is True
        assert st2["is_new_back_frame"] is False
        assert st2["should_update_back_state"] is False

        # Advancing by one 60fps interval (0.10 + 1/60 = 0.1167s) hits the NEXT physical Back frame
        fb3, ff3, st3 = reader.get_synchronized_pair(0.10 + 1.0 / 60.0)
        assert fb3 is not None
        assert st3["frame_idx_back"] == st1["frame_idx_back"] + 1
        assert st3["valid_pair"] is False
        assert st3["is_new_back_frame"] is True
        assert st3["should_update_back_state"] is True

        # Case B: Both frames exist, but timestamp is outside the declared overlap window
        reader.sync_info.overlap_start_back_sec = 1.50
        fb4, ff4, st4 = reader.get_synchronized_pair(1.00)
        assert fb4 is not None and ff4 is not None
        assert st4["valid_back_frame"] is True
        assert st4["valid_front_frame"] is True
        assert st4["is_in_overlap"] is False
        assert st4["valid_pair"] is False
        assert st4["should_update_back_state"] is True
        assert st4["should_update_front_state"] is True


