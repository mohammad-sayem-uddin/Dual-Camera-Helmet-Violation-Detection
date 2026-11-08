"""Phase 12D Unit & Integration Test Suite:
Per-Camera Hierarchical Detection, Rider-Motorcycle Coupling & Tracking.

Explicitly separates synthetic geometry/tracking unit tests from real-video smoke tests.
"""
from __future__ import annotations

import resource
import numpy as np
import pytest

from src.detection.camera_detector import (
    FrameDetectionBundle,
    PerCameraDetector,
    SharedDetectorRegistry,
    StructuredDetection,
)
from src.models.helmet_detector_adapter import HelmetDetectorAdapter
from src.models.vehicle_detector_adapter import VehiclePersonDetectorAdapter
from src.synchronization.synchronized_reader import SynchronizedVideoReader
from src.temporal.state_machine import StateMachineConfig, TemporalState
from src.temporal.violation_decision import DecisionConfig
from src.tracking.camera_tracker import (
    CameraTrackingPipeline,
    Phase10AdapterConfig,
)
from src.tracking.kalman_bytetrack import (
    ByteTrackConfig,
    KalmanBoxFilter,
    TrackLifecycleState,
    two_stage_bytetrack_assign,
)
from src.tracking.moto_rider_coupler import CouplerConfig, MotoRiderCoupler


def _make_det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cls_name: str,
    conf: float = 0.85,
    cam: str = "front",
    f_idx: int = 0,
    pts: float = 0.0,
) -> StructuredDetection:
    mapping = {
        "motorcycle": (3, "motorcycle", "yolov8n_coco"),
        "person": (0, "person", "yolov8n_coco"),
        "no_helmet": (0, "head", "phase9_best_helmet"),
        "helmet": (1, "head", "phase9_best_helmet"),
    }
    cid, cat, src = mapping[cls_name]
    return StructuredDetection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=conf,
        class_id=cid,
        class_name=cls_name,
        category=cat,
        source_model=src,
        camera_id=cam,
        frame_index=f_idx,
        pts_timestamp_sec=pts,
    )


def _make_bundle(
    cam: str,
    f_idx: int,
    pts: float,
    motos: list[StructuredDetection],
    persons: list[StructuredDetection],
    heads: list[StructuredDetection],
) -> FrameDetectionBundle:
    helmets = [h for h in heads if h.class_name == "helmet"]
    no_helmets = [h for h in heads if h.class_name == "no_helmet"]
    return FrameDetectionBundle(
        camera_id=cam,
        frame_index=f_idx,
        pts_timestamp_sec=pts,
        motorcycles=motos,
        persons=persons,
        helmet_boxes=helmets,
        no_helmet_boxes=no_helmets,
        all_heads=heads,
        vehicle_latency_ms=1.0,
        helmet_latency_ms=1.0,
        total_latency_ms=2.0,
        device="cpu",
    )


# ============================================================================
# 1 & 2. Model Class Mapping Validation & Detection Adapter Smoke Tests
# ============================================================================


def test_model_class_mappings_and_sha256_verification():
    """
    Test 1 & 2: Verify actual loaded model metadata and SHA-256 hashes for both
    COCO `yolov8n.pt` and Phase 9 `best.pt`.
    """
    veh_adapter = SharedDetectorRegistry.get_vehicle_detector(device="cpu")
    helm_adapter = SharedDetectorRegistry.get_helmet_detector(device="cpu")

    veh_info = veh_adapter.get_info()
    assert veh_info["verified"] is True
    assert veh_info["target_classes"] == {0: "person", 3: "motorcycle"}
    assert veh_info["total_coco_classes"] == 80
    assert (
        veh_adapter.get_sha256()
        == "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
    )

    helm_info = helm_adapter.get_info()
    assert helm_info["verified"] is True
    # Helmet model must contain ONLY no_helmet (0) and helmet (1)
    assert dict(helm_adapter.class_names) == {0: "no_helmet", 1: "helmet"}


def test_per_camera_detector_smoke_on_blank_frame():
    """Test 1: Verify PerCameraDetector runs both models and returns structured bundle."""
    det_front = PerCameraDetector(camera_id="front", device="cpu")
    det_back = PerCameraDetector(camera_id="back", device="cpu")
    # Singleton sharing check
    assert det_front.vehicle_detector is det_back.vehicle_detector
    assert det_front.helmet_detector is det_back.helmet_detector

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    bundle = det_front.detect_frame(blank, frame_index=0, pts_timestamp_sec=0.0)
    assert bundle.camera_id == "front"
    assert bundle.frame_index == 0
    assert isinstance(bundle.motorcycles, list)
    assert isinstance(bundle.persons, list)
    assert isinstance(bundle.all_heads, list)


# ============================================================================
# 3, 4, 5 & 6. Synthetic Geometry Coupling Tests (MotoRiderCoupler)
# ============================================================================


def test_synthetic_driver_plus_pillion_and_three_rider_coupling():
    """
    Test 3 (Synthetic): Verify MotoRiderCoupler links 1, 2, and 3 riders to a parent
    motorcycle and orders driver, pillion_1, pillion_2 only when separation is sufficient.
    """
    coupler = MotoRiderCoupler()
    moto = _make_det(400, 350, 600, 750, "motorcycle")
    p_driver = _make_det(420, 220, 580, 650, "person")
    p_pillion = _make_det(430, 180, 570, 580, "person")

    # Front view: driver head is lower in frame (cy=260) than pillion_1 (cy=210) and pillion_2 (cy=175)
    h_driver = _make_det(460, 230, 540, 290, "helmet")      # cy = 260
    h_pillion1 = _make_det(465, 185, 535, 235, "no_helmet")  # cy = 210
    h_pillion2 = _make_det(470, 150, 530, 200, "no_helmet")  # cy = 175

    res = coupler.couple_frame(
        camera_id="front",
        frame_index=1,
        pts_timestamp_sec=0.0167,
        motorcycles=[moto],
        persons=[p_driver, p_pillion],
        heads=[h_pillion1, h_driver, h_pillion2],
    )
    assert len(res.coupled_motorcycles) == 1
    assert len(res.orphan_riders) == 0
    cm = res.coupled_motorcycles[0]
    assert cm.rider_count == 3
    assert [r.slot_role for r in cm.riders] == ["driver", "pillion_1", "pillion_2"]
    assert all(r.ordering_confident for r in cm.riders)
    assert cm.riders[0].head_detection.class_name == "helmet"
    assert cm.riders[1].head_detection.class_name == "no_helmet"

    # Now test ambiguous slot ordering when two heads have nearly identical vertical y and height
    h_same_a = _make_det(450, 230, 500, 280, "helmet")      # cy = 255, h = 50
    h_same_b = _make_det(505, 232, 555, 282, "no_helmet")   # cy = 257 (dy=2 < 8px), h = 50
    res_unranked = coupler.couple_frame(
        camera_id="front",
        frame_index=2,
        pts_timestamp_sec=0.0333,
        motorcycles=[moto],
        persons=[p_driver],
        heads=[h_same_a, h_same_b],
    )
    cm_u = res_unranked.coupled_motorcycles[0]
    assert cm_u.rider_count == 2
    assert all(not r.ordering_confident for r in cm_u.riders)
    assert cm_u.riders[0].slot_role == "ambiguous_slot_0"
    assert cm_u.riders[1].slot_role == "ambiguous_slot_1"


def test_synthetic_adjacent_motorcycle_rider_separation():
    """
    Test 4 (Synthetic): Verify two adjacent motorcycles each receive their own rider
    without cross-assignment or duplicate assignment.
    """
    coupler = MotoRiderCoupler()
    moto_left = _make_det(200, 350, 380, 750, "motorcycle")
    moto_right = _make_det(650, 350, 830, 750, "motorcycle")

    head_left = _make_det(255, 220, 325, 290, "helmet")
    head_right = _make_det(705, 220, 775, 290, "no_helmet")

    res = coupler.couple_frame(
        camera_id="front",
        frame_index=1,
        pts_timestamp_sec=0.0167,
        motorcycles=[moto_left, moto_right],
        persons=[],
        heads=[head_left, head_right],
    )
    assert len(res.orphan_riders) == 0
    assert res.coupled_motorcycles[0].rider_count == 1
    assert res.coupled_motorcycles[1].rider_count == 1
    assert res.coupled_motorcycles[0].riders[0].head_detection.class_name == "helmet"
    assert res.coupled_motorcycles[1].riders[0].head_detection.class_name == "no_helmet"


def test_synthetic_orphan_rider_fallback_and_ambiguous_abstention():
    """
    Test 5 & 6 (Synthetic):
    - Verify OrphanRiderObservation when no motorcycle is detected or head is far away.
    - Verify AMBIGUOUS_ADJACENT_MOTORCYCLES abstention when a head sits symmetrically
      between two overlapping motorcycles with score margin < ambiguity_margin_threshold.
    """
    coupler = MotoRiderCoupler(CouplerConfig(ambiguity_margin_threshold=0.08))
    head_mid = _make_det(470, 220, 530, 280, "no_helmet")  # cx = 500

    # Case A: No motorcycle in frame
    res_no_moto = coupler.couple_frame("front", 1, 0.0167, motorcycles=[], persons=[], heads=[head_mid])
    assert len(res_no_moto.orphan_riders) == 1
    assert res_no_moto.orphan_riders[0].rejection_reason == "NO_MOTORCYCLE_IN_FRAME"

    # Case B: Head sits equidistant between two symmetrically overlapping motorcycles (cx=485 and cx=515)
    moto_a = _make_det(385, 350, 585, 750, "motorcycle")  # cx = 485
    moto_b = _make_det(415, 350, 615, 750, "motorcycle")  # cx = 515
    res_ambig = coupler.couple_frame(
        "front", 2, 0.0333, motorcycles=[moto_a, moto_b], persons=[], heads=[head_mid]
    )
    assert len(res_ambig.orphan_riders) == 1
    assert res_ambig.orphan_riders[0].rejection_reason == "AMBIGUOUS_ADJACENT_MOTORCYCLES"
    assert res_ambig.coupled_motorcycles[0].rider_count == 0
    assert res_ambig.coupled_motorcycles[1].rider_count == 0


# ============================================================================
# 7, 8 & 9. Kalman Prediction, ByteTrack Stage 1/2 & Track Lifecycle Recovery
# ============================================================================


def test_kalman_filter_prediction_and_two_stage_bytetrack_matching():
    """
    Test 7 & 8 (Synthetic):
    - Verify KalmanBoxFilter predicts constant-velocity motion accurately.
    - Verify ByteTrack Stage 1 matches high-conf detections (>= 0.45) and Stage 2
      recovers unmatched confirmed tracks using low-conf detections (0.18 <= conf < 0.45).
    """
    kf = KalmanBoxFilter((100.0, 200.0, 200.0, 400.0))
    for step in range(1, 5):
        shift = step * 20.0
        kf.predict(dt_frames=1.0)
        kf.update((100.0 + shift, 200.0, 200.0 + shift, 400.0))

    # Velocity in x should be positive (~15-20 px/frame)
    vx, _ = kf.velocity_xy
    assert vx > 10.0
    pred_next = kf.predict(dt_frames=1.0)
    assert pred_next[0] > 180.0

    # Test Two-Stage ByteTrack matching
    cfg = ByteTrackConfig(high_conf_threshold=0.45, low_conf_threshold=0.18)
    track_boxes = [
        (100.0, 100.0, 200.0, 300.0),  # Track 0
        (500.0, 100.0, 600.0, 300.0),  # Track 1
    ]
    det_boxes = [
        (105.0, 100.0, 205.0, 300.0),  # High-conf detection matching Track 0
        (505.0, 100.0, 605.0, 300.0),  # Low-conf (occluded) detection matching Track 1
    ]
    det_confs = [0.82, 0.24]

    bt_out = two_stage_bytetrack_assign(
        track_boxes, det_boxes, det_confs, cfg, track_can_use_stage2=[True, True]
    )
    assert len(bt_out["stage1_matches"]) == 1
    assert bt_out["stage1_matches"][0][0] == 0 and bt_out["stage1_matches"][0][1] == 0
    assert len(bt_out["stage2_matches"]) == 1
    assert bt_out["stage2_matches"][0][0] == 1 and bt_out["stage2_matches"][0][1] == 1


def test_temporary_occlusion_and_track_lifecycle_recovery():
    """
    Test 9 (Synthetic): Verify MotoTrack transitions through
    TENTATIVE -> CONFIRMED -> LOST -> RECOVERED (via Stage 2 low-conf detection) -> TERMINATED.
    """
    pipeline = CameraTrackingPipeline(
        camera_id="front",
        bytetrack_config=ByteTrackConfig(min_confirm_hits=2, max_lost_frames=2),
    )

    moto_box = (400.0, 350.0, 600.0, 750.0)
    head_box = (460.0, 230.0, 540.0, 290.0)

    # Frame 0: Track initialized -> TENTATIVE (hits=1 < min_confirm_hits=2)
    b0 = _make_bundle(
        "front", 0, 0.0,
        [_make_det(*moto_box, "motorcycle", conf=0.88, f_idx=0)],
        [],
        [_make_det(*head_box, "no_helmet", conf=0.85, f_idx=0)],
    )
    s0 = pipeline.step(b0)
    assert s0["active_moto_tracks"][0]["lifecycle_state"] == TrackLifecycleState.TENTATIVE.value
    assert s0["active_moto_tracks"][0]["moto_track_id"] == "front_moto_0001"

    # Frame 1: Second hit -> CONFIRMED
    b1 = _make_bundle(
        "front", 1, 0.0167,
        [_make_det(*moto_box, "motorcycle", conf=0.88, f_idx=1)],
        [],
        [_make_det(*head_box, "no_helmet", conf=0.85, f_idx=1)],
    )
    s1 = pipeline.step(b1)
    assert s1["active_moto_tracks"][0]["lifecycle_state"] == TrackLifecycleState.CONFIRMED.value

    # Frame 2: Total occlusion (no detections) -> LOST (time_since_update=1 <= max_lost_frames=2)
    b2 = _make_bundle("front", 2, 0.0333, [], [], [])
    s2 = pipeline.step(b2)
    assert s2["active_moto_tracks"][0]["lifecycle_state"] == TrackLifecycleState.LOST.value

    # Frame 3: Re-detected with LOW confidence (0.25, Stage 2 ByteTrack) -> RECOVERED!
    b3 = _make_bundle(
        "front", 3, 0.0500,
        [_make_det(*moto_box, "motorcycle", conf=0.25, f_idx=3)],
        [],
        [_make_det(*head_box, "no_helmet", conf=0.85, f_idx=3)],
    )
    s3 = pipeline.step(b3)
    m_rec = s3["active_moto_tracks"][0]
    assert m_rec["lifecycle_state"] == TrackLifecycleState.RECOVERED.value
    assert m_rec["stage2_low_conf_matches"] == 1
    assert m_rec["recovery_count"] == 1

    # Frames 4, 5, 6: Missing for 3 consecutive frames (> max_lost_frames=2) -> TERMINATED
    for f_i in (4, 5, 6):
        s_miss = pipeline.step(_make_bundle("front", f_i, f_i * 0.0167, [], [], []))
    assert len(s_miss["active_moto_tracks"]) == 0
    assert len(pipeline.terminated_moto_tracks) == 1
    assert pipeline.terminated_moto_tracks[0].lifecycle_state == TrackLifecycleState.TERMINATED


# ============================================================================
# 10 & 11. Multi-Rider Independent States, Hysteresis & Cooldown Regression
# ============================================================================


def test_multiple_riders_independent_helmet_states_and_phase10_cooldown():
    """
    Test 10 & 11 (Synthetic):
    - Verify a motorcycle with 2 riders (Driver wearing a helmet, Pillion 1 without a helmet)
      maintains strictly independent RiderSubTrack temporal states:
        * Driver (slot 0) -> HELMET_CONFIRMED (0 alerts)
        * Pillion 1 (slot 1) -> NO_HELMET_CONFIRMED (emits exactly 1 ViolationAlert)
    - Verify Phase 10 hysteresis and cooldown prevent duplicate alerts on Pillion 1.
    """
    p10_cfg = Phase10AdapterConfig(
        aggregator_method="ema",
        sliding_window_size=10,
        ema_alpha=0.45,
        state_machine_config=StateMachineConfig(
            min_observation_frames=3,
            confirm_frames=5,
            confirm_threshold=0.65,
            helmet_threshold=0.65,
        ),
        decision_config=DecisionConfig(
            violation_alert_threshold=0.65,
            min_confirmation_frames=5,
            suppress_duplicates_per_track=True,
            cooldown_frames=50,
        ),
    )
    pipeline = CameraTrackingPipeline(camera_id="back", phase10_config=p10_cfg)

    moto_box = (400.0, 350.0, 600.0, 750.0)
    # In Back view: driver is higher in frame (smaller cy=200), pillion is lower (larger cy=255)
    driver_head = (465.0, 175.0, 535.0, 225.0)   # cy = 200 (wearing helmet)
    pillion_head = (460.0, 230.0, 540.0, 280.0)  # cy = 255 (no_helmet violation)

    total_alerts = []
    for f_idx in range(10):
        bundle = _make_bundle(
            "back",
            f_idx,
            f_idx * 0.0333,
            [_make_det(*moto_box, "motorcycle", conf=0.90, cam="back", f_idx=f_idx)],
            [],
            [
                _make_det(*driver_head, "helmet", conf=0.92, cam="back", f_idx=f_idx),
                _make_det(*pillion_head, "no_helmet", conf=0.92, cam="back", f_idx=f_idx),
            ],
        )
        step_out = pipeline.step(bundle)
        total_alerts.extend(step_out["new_alerts"])

    m_dict = step_out["active_moto_tracks"][0]
    assert m_dict["moto_track_id"] == "back_moto_0001"
    r0 = m_dict["rider_subtracks"][0]
    r1 = m_dict["rider_subtracks"][1]

    assert r0["slot_role"] == "driver"
    assert r0["temporal_state"] == TemporalState.HELMET_CONFIRMED.value
    assert r0["alerts_emitted"] == 0

    assert r1["slot_role"] == "pillion_1"
    assert r1["temporal_state"] == TemporalState.NO_HELMET_CONFIRMED.value
    # Cooldown suppression ensures exactly 1 alert was emitted across the 10 frames
    assert r1["alerts_emitted"] == 1
    assert len(total_alerts) == 1

    # Hysteresis check: 1 single noisy "helmet" frame on pillion_1 must NOT revoke NO_HELMET_CONFIRMED
    b_noisy = _make_bundle(
        "back",
        10,
        10 * 0.0333,
        [_make_det(*moto_box, "motorcycle", conf=0.90, cam="back", f_idx=10)],
        [],
        [
            _make_det(*driver_head, "helmet", conf=0.92, cam="back", f_idx=10),
            _make_det(*pillion_head, "helmet", conf=0.55, cam="back", f_idx=10),
        ],
    )
    step_noisy = pipeline.step(b_noisy)
    assert (
        step_noisy["active_moto_tracks"][0]["rider_subtracks"][1]["temporal_state"]
        == TemporalState.NO_HELMET_CONFIRMED.value
    )


# ============================================================================
# 12 & 13. Production PTS Gate Integration & Short Real-Video Smoke Test
# ============================================================================


def test_production_pts_update_gate_and_real_video_smoke():
    """
    Test 12 & 13 (Real-Video Smoke Test):
    - Integrate SynchronizedVideoReader + PerCameraDetector + CameraTrackingPipeline
      on `pair_015` (60 fps Front + 30 fps Back) for 6 canonical 60 Hz ticks.
    - Verify Back detector and tracker execute ONLY on the 3 unique physical 30 fps
      Back frames (skipping the 3 repeated ticks), while Front executes on all 6 ticks.
    - Verify Front and Back track namespaces (`front_*` vs. `back_*`) are disjoint.
    - Verify peak RSS memory stays well within the Apple M1 8 GB budget (< 1,850 MB).
    """
    det_front = PerCameraDetector(camera_id="front")
    det_back = PerCameraDetector(camera_id="back")
    trk_front = CameraTrackingPipeline.from_yaml(camera_id="front")
    trk_back = CameraTrackingPipeline.from_yaml(camera_id="back")

    front_det_calls = 0
    back_det_calls = 0

    with SynchronizedVideoReader("pair_015") as sync_reader:
        t_start = sync_reader.sync_info.overlap_start_back_sec
        for tick in range(6):
            t_ref = t_start + tick / 60.0
            fb, ff, status = sync_reader.get_synchronized_pair(t_ref)
            assert status["valid_pair"] is True

            if status["should_update_front_state"] and ff is not None:
                front_det_calls += 1
                bundle_f = det_front.detect_frame(
                    ff, status["frame_idx_front"], status["t_front_actual_pts_sec"]
                )
                out_f = trk_front.step(bundle_f, should_update_state=True)
                assert out_f["did_advance_state"] is True
            else:
                out_f = trk_front.step(bundle_f, should_update_state=False)
                assert out_f["did_advance_state"] is False

            if status["should_update_back_state"] and fb is not None:
                back_det_calls += 1
                bundle_b = det_back.detect_frame(
                    fb, status["frame_idx_back"], status["t_back_actual_pts_sec"]
                )
                out_b = trk_back.step(bundle_b, should_update_state=True)
                assert out_b["did_advance_state"] is True
            else:
                out_b = trk_back.step(bundle_b, should_update_state=False)
                assert out_b["did_advance_state"] is False

    # Over 6 canonical 60 Hz ticks on pair_015, Front (60 fps) updated 6 times,
    # while Back (30 fps) updated exactly 3 times and skipped 3 duplicate ticks!
    assert front_det_calls == 6
    assert trk_front.frames_updated == 6
    assert trk_front.duplicate_ticks_skipped == 0

    assert back_det_calls == 3
    assert trk_back.frames_updated == 3
    assert trk_back.duplicate_ticks_skipped == 3

    # Check namespace separation on any created tracks
    for m in trk_front.active_moto_tracks.values():
        assert m.moto_track_id.startswith("front_moto_")
    for m in trk_back.active_moto_tracks.values():
        assert m.moto_track_id.startswith("back_moto_")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    assert peak_rss_mb < 1850.0
