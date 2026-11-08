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


# ============================================================================
# 14..19. Phase 12D Review Correction Regression Tests (Corrections 1, 2 & 4)
# ============================================================================


def test_rider_apparent_vertical_ordering_exchange_preserves_identity():
    """
    Correction 1 Regression Test A (Synthetic):
    Two riders on a motorcycle exchange apparent vertical ordering in image space
    (causing MotoRiderCoupler to swap their `slot_index` 0 <-> 1 with `ordering_confident=True`).
    Verify:
    - `MotoTrack._sync_riders_on_match` detects the slot-vs-spatial conflict.
    - High-confidence disjoint spatial trajectory matching overrides the swapped slot_index.
    - Driver (`s0`, `helmet`) and Pillion (`s1`, `no_helmet`) preserve their separate
      Phase 10 temporal states and semantic roles without transferring history.
    - No duplicate rider tracks are created.
    """
    pipeline = CameraTrackingPipeline(camera_id="front")
    moto_box = (400.0, 350.0, 620.0, 750.0)

    # Frames 0..4:
    # Driver (s0, helmet) is on left-lower side: (430, 220, 490, 280) -> cy = 250
    # Pillion (s1, no_helmet) is on right-upper side: (530, 200, 590, 260) -> cy = 230
    # In front view, larger cy -> slot 0 (driver), smaller cy -> slot 1 (pillion_1).
    for f_idx in range(5):
        b = _make_bundle(
            "front",
            f_idx,
            f_idx * 0.0167,
            [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=f_idx)],
            [],
            [
                _make_det(430.0, 220.0, 490.0, 280.0, "helmet", conf=0.92, f_idx=f_idx),
                _make_det(530.0, 200.0, 590.0, 260.0, "no_helmet", conf=0.92, f_idx=f_idx),
            ],
        )
        out = pipeline.step(b)

    m = out["active_moto_tracks"][0]
    assert len(m["rider_subtracks"]) == 2
    assert m["rider_subtracks"][0]["temporal_state"] == TemporalState.HELMET_CONFIRMED.value
    assert m["rider_subtracks"][1]["temporal_state"] == TemporalState.NO_HELMET_CONFIRMED.value

    # Frame 5: Apparent vertical ordering exchange!
    # Driver (left, helmet) shifts slightly up to y=(195..255), cy = 225 (IoU with s0 = 35/85 = 0.412)
    # Pillion (right, no_helmet) shifts slightly down to y=(210..270), cy = 240 (IoU with s1 = 50/70 = 0.714)
    # Because cy(Pillion)=240 > cy(Driver)=225 by 15px >= 8px, Coupler assigns:
    #   Pillion (no_helmet) -> slot_index=0 (ordering_confident=True)
    #   Driver (helmet)     -> slot_index=1 (ordering_confident=True)
    b_swap = _make_bundle(
        "front",
        5,
        5 * 0.0167,
        [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=5)],
        [],
        [
            _make_det(430.0, 195.0, 490.0, 255.0, "helmet", conf=0.95, f_idx=5),
            _make_det(530.0, 210.0, 590.0, 270.0, "no_helmet", conf=0.95, f_idx=5),
        ],
    )
    out_swap = pipeline.step(b_swap)
    m_swap = out_swap["active_moto_tracks"][0]

    # Verify conflict was detected and resolved via spatial trajectory continuity
    assert m_swap["slot_spatial_conflicts_detected"] >= 2
    assert m_swap["spatial_identity_overrides"] >= 2
    assert len(m_swap["rider_subtracks"]) == 2

    s0 = m_swap["rider_subtracks"][0]
    s1 = m_swap["rider_subtracks"][1]

    # Driver (s0) must STILL have 6 consecutive helmet frames and 0 no_helmet frames!
    assert s0["slot_role"] == "driver"
    assert s0["hits"] == 6
    assert s0["consecutive_helmet_frames"] == 6
    assert s0["consecutive_no_helmet_frames"] == 0
    assert s0["temporal_state"] == TemporalState.HELMET_CONFIRMED.value

    # Pillion (s1) must STILL have 6 consecutive no_helmet frames and 0 helmet frames!
    assert s1["slot_role"] == "pillion_1"
    assert s1["hits"] == 6
    assert s1["consecutive_no_helmet_frames"] == 6
    assert s1["consecutive_helmet_frames"] == 0
    assert s1["temporal_state"] == TemporalState.NO_HELMET_CONFIRMED.value


def test_rider_slot_and_spatial_trajectory_conflict_abstains_without_contamination_or_duplicates():
    """
    Correction 1 Regression Test B (Synthetic):
    Slot 0 and Slot 1 observations conflict with their previous spatial trajectories
    and do NOT meet the high-confidence disjoint override threshold.
    Verify:
    - Conflict between slot identity and spatial assignment is detected.
    - Tracker conservatively abstains from updating the wrong rider's Phase 10 history.
    - Neither rider's accumulated EMA confidence or consecutive frame counter is contaminated.
    - No duplicate rider tracks (e.g. slot 2) are spawned.
    """
    pipeline = CameraTrackingPipeline(camera_id="front")
    moto_box = (400.0, 350.0, 620.0, 750.0)

    # Establish s0 (driver, helmet) at (430, 230, 490, 290) [cy=260]
    # and s1 (pillion_1, no_helmet) at (530, 180, 590, 240) [cy=210] over 5 frames
    for f_idx in range(5):
        pipeline.step(
            _make_bundle(
                "front",
                f_idx,
                f_idx * 0.0167,
                [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=f_idx)],
                [],
                [
                    _make_det(430.0, 230.0, 490.0, 290.0, "helmet", conf=0.92, f_idx=f_idx),
                    _make_det(530.0, 180.0, 590.0, 240.0, "no_helmet", conf=0.92, f_idx=f_idx),
                ],
            )
        )

    moto_track = pipeline.active_moto_tracks["front_moto_0001"]
    s0_conf_before = dict(moto_track.rider_subtracks[0].aggregated_confidence)
    s1_conf_before = dict(moto_track.rider_subtracks[1].aggregated_confidence)

    # Frame 5: Conflicting observations that disagree with previous trajectories and have
    # low/conflicting spatial overlap (IoU ~ 0.21 < rider_spatial_override_min_iou=0.35 with swapped slots):
    # - Observation near s1's upper-right region (530, 215, 590, 275) [cy=245] has larger cy -> gets slot_index=0!
    #   Its IoU with s1 (530, 180, 590, 240) is 25/95 = 0.263 (< 0.35 override gate), and class is "helmet".
    # - Observation near s0's lower-left region (430, 195, 490, 255) [cy=225] has smaller cy -> gets slot_index=1!
    #   Its IoU with s0 (430, 230, 490, 290) is 25/95 = 0.263 (< 0.35 override gate), and class is "no_helmet".
    b_conflict = _make_bundle(
        "front",
        5,
        5 * 0.0167,
        [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=5)],
        [],
        [
            _make_det(530.0, 215.0, 590.0, 275.0, "helmet", conf=0.95, f_idx=5),
            _make_det(430.0, 195.0, 490.0, 255.0, "no_helmet", conf=0.95, f_idx=5),
        ],
    )
    out_conflict = pipeline.step(b_conflict)
    m_c = out_conflict["active_moto_tracks"][0]

    assert m_c["slot_spatial_conflicts_detected"] >= 2
    assert m_c["rider_abstained_updates"] >= 2
    # Must NOT create a duplicate 3rd rider track
    assert len(m_c["rider_subtracks"]) == 2

    # Both subtracks abstained on frame 5: hits remain 5, time_since_update becomes 1,
    # and Phase 10 EMA confidences are completely uncontaminated!
    assert m_c["rider_subtracks"][0]["hits"] == 5
    assert m_c["rider_subtracks"][1]["hits"] == 5
    assert m_c["rider_subtracks"][0]["time_since_update"] == 1
    assert m_c["rider_subtracks"][1]["time_since_update"] == 1
    assert m_c["rider_subtracks"][0]["aggregated_confidence"] == s0_conf_before
    assert m_c["rider_subtracks"][1]["aggregated_confidence"] == s1_conf_before


def test_partial_rider_occlusion_and_reappearance_preserves_driver_and_pillion_identities():
    """
    Correction 1 Regression Test C (Synthetic):
    Driver (`s0`, `helmet`) and Pillion (`s1`, `no_helmet`) are tracked on a motorcycle.
    Then Driver (`s0`) is temporarily occluded for 3 frames while Pillion (`s1`) remains visible.
    Even though MotoRiderCoupler assigns `slot_index=0` to the single visible Pillion head,
    verify:
    - Pillion's `no_helmet` observations update `s1` (via spatial consistency) and NEVER contaminate `s0`.
    - Driver (`s0`) enters `TEMPORARILY_OCCLUDED` during the 3 missed frames.
    - When Driver (`s0`) reappears at frame 8, `s0` resumes `HELMET_CONFIRMED` with 0 no_helmet frames.
    """
    pipeline = CameraTrackingPipeline(camera_id="front")
    moto_box = (400.0, 350.0, 600.0, 750.0)
    driver_box = (460.0, 240.0, 530.0, 300.0)   # cy = 270 -> slot 0 (driver)
    pillion_box = (465.0, 170.0, 525.0, 225.0)  # cy = 197.5 -> slot 1 (pillion_1)

    # Frames 0..4: Both riders visible
    for f_idx in range(5):
        pipeline.step(
            _make_bundle(
                "front",
                f_idx,
                f_idx * 0.0167,
                [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=f_idx)],
                [],
                [
                    _make_det(*driver_box, "helmet", conf=0.92, f_idx=f_idx),
                    _make_det(*pillion_box, "no_helmet", conf=0.92, f_idx=f_idx),
                ],
            )
        )

    # Frames 5, 6, 7: Driver (s0) is occluded; ONLY Pillion (s1, no_helmet) is detected!
    for f_idx in (5, 6, 7):
        out_occ = pipeline.step(
            _make_bundle(
                "front",
                f_idx,
                f_idx * 0.0167,
                [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=f_idx)],
                [],
                [_make_det(*pillion_box, "no_helmet", conf=0.94, f_idx=f_idx)],
            )
        )

    m_occ = out_occ["active_moto_tracks"][0]
    assert len(m_occ["rider_subtracks"]) == 2
    s0_occ = m_occ["rider_subtracks"][0]
    s1_occ = m_occ["rider_subtracks"][1]

    # s0 (driver) must be TEMPORARILY_OCCLUDED with hits=5 and 0 no_helmet frames
    assert s0_occ["slot_role"] == "driver"
    assert s0_occ["temporal_state"] == TemporalState.TEMPORARILY_OCCLUDED.value
    assert s0_occ["hits"] == 5
    assert s0_occ["time_since_update"] == 3
    assert s0_occ["consecutive_no_helmet_frames"] == 0

    # s1 (pillion_1) received all 3 updates (hits=8) and preserved slot_role="pillion_1"
    assert s1_occ["slot_role"] == "pillion_1"
    assert s1_occ["temporal_state"] == TemporalState.NO_HELMET_CONFIRMED.value
    assert s1_occ["hits"] == 8
    assert s1_occ["time_since_update"] == 0

    # Frame 8: Driver (s0) reappears alongside Pillion (s1)
    out_re = pipeline.step(
        _make_bundle(
            "front",
            8,
            8 * 0.0167,
            [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=8)],
            [],
            [
                _make_det(*driver_box, "helmet", conf=0.92, f_idx=8),
                _make_det(*pillion_box, "no_helmet", conf=0.92, f_idx=8),
            ],
        )
    )
    m_re = out_re["active_moto_tracks"][0]
    assert len(m_re["rider_subtracks"]) == 2
    assert m_re["rider_subtracks"][0]["temporal_state"] == TemporalState.HELMET_CONFIRMED.value
    assert m_re["rider_subtracks"][0]["hits"] == 6
    assert m_re["rider_subtracks"][0]["time_since_update"] == 0
    assert m_re["rider_subtracks"][0]["consecutive_no_helmet_frames"] == 0


def test_ambiguous_inter_rider_observation_abstains_without_contaminating_phase10_history():
    """
    Correction 1 Regression Test D (Synthetic):
    An ambiguous head observation falls symmetrically between two existing RiderSubTracks
    (`|IoU(s0, obs) - IoU(s1, obs)| < rider_ambiguity_iou_margin`).
    Verify the tracker abstains from updating either RiderSubTrack so neither rider's
    Phase 10 history is contaminated, and no duplicate subtrack is spawned.
    """
    pipeline = CameraTrackingPipeline(camera_id="front")
    moto_box = (400.0, 350.0, 600.0, 750.0)
    # s0 at y=(220..280) [cy=250], s1 at y=(190..250) [cy=220] -> dy = 30px >= 8px
    s0_box = (460.0, 220.0, 520.0, 280.0)
    s1_box = (460.0, 190.0, 520.0, 250.0)

    for f_idx in range(4):
        pipeline.step(
            _make_bundle(
                "front",
                f_idx,
                f_idx * 0.0167,
                [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=f_idx)],
                [],
                [
                    _make_det(*s0_box, "helmet", conf=0.92, f_idx=f_idx),
                    _make_det(*s1_box, "no_helmet", conf=0.92, f_idx=f_idx),
                ],
            )
        )

    moto_track = pipeline.active_moto_tracks["front_moto_0001"]
    s0_ema_before = dict(moto_track.rider_subtracks[0].aggregated_confidence)
    s1_ema_before = dict(moto_track.rider_subtracks[1].aggregated_confidence)

    # Frame 4: A single merged/ambiguous head box right at y=(205..265) [cy=235],
    # overlapping s0 (220..280) and s1 (190..250) with identical IoU!
    ambig_head = (460.0, 205.0, 520.0, 265.0)
    out_ambig = pipeline.step(
        _make_bundle(
            "front",
            4,
            4 * 0.0167,
            [_make_det(*moto_box, "motorcycle", conf=0.90, f_idx=4)],
            [],
            [_make_det(*ambig_head, "no_helmet", conf=0.95, f_idx=4)],
        )
    )
    m_ambig = out_ambig["active_moto_tracks"][0]
    assert m_ambig["rider_abstained_updates"] >= 1
    assert len(m_ambig["rider_subtracks"]) == 2
    assert m_ambig["rider_subtracks"][0]["hits"] == 4
    assert m_ambig["rider_subtracks"][1]["hits"] == 4
    assert m_ambig["rider_subtracks"][0]["abstained_updates_count"] >= 1
    assert m_ambig["rider_subtracks"][1]["abstained_updates_count"] >= 1
    assert m_ambig["rider_subtracks"][0]["aggregated_confidence"] == s0_ema_before
    assert m_ambig["rider_subtracks"][1]["aggregated_confidence"] == s1_ema_before


def test_bipartite_head_to_person_one_to_one_and_ambiguity_abstention():
    """
    Correction 2 Regression Test (Synthetic):
    Verify gated one-to-one Hungarian bipartite head-to-person assignment in MotoRiderCoupler:
    1. One person cannot be assigned to multiple heads in the same frame.
    2. One head cannot be assigned to multiple persons.
    3. Unmatched heads remain available for motorcycle coupling (`w_no_person_*`) or orphan handling.
    4. Ambiguous head-to-person assignments abstain (`head_person_abstained=True`, `person_detection=None`)
       rather than creating false associations.
    5. Multiple riders on one motorcycle remain supported.
    """
    coupler = MotoRiderCoupler(CouplerConfig(head_person_ambiguity_margin=0.08))
    moto = _make_det(400.0, 350.0, 600.0, 750.0, "motorcycle")

    # 1. One person box (driver body) and two distinct rider heads (driver + pillion):
    #    Only the closer apex head (driver) may match `p_only`; the second head must have
    #    `person_detection is None` yet still couple to `moto`!
    p_only = _make_det(420.0, 225.0, 580.0, 650.0, "person")
    h_driver = _make_det(460.0, 230.0, 540.0, 290.0, "helmet")      # y1=230 (5px from p_only.y1=225)
    h_pillion = _make_det(465.0, 165.0, 535.0, 215.0, "no_helmet")  # y1=165 (60px above p_only.y1=225)

    res_1p_2h = coupler.couple_frame(
        camera_id="front",
        frame_index=1,
        pts_timestamp_sec=0.0167,
        motorcycles=[moto],
        persons=[p_only],
        heads=[h_driver, h_pillion],
    )
    cm = res_1p_2h.coupled_motorcycles[0]
    assert cm.rider_count == 2
    assigned_persons = [r.person_detection for r in cm.riders if r.person_detection is not None]
    # Strictly at most 1 head can receive the single person box!
    assert len(assigned_persons) == 1
    assert cm.riders[0].person_detection == p_only
    assert cm.riders[1].person_detection is None

    # 2. Ambiguous 2-heads-to-1-person case: two heads at identical vertical apex y1=230 and symmetric x
    #    inside `p_only` (cx=500) -> score difference == 0.0 < head_person_ambiguity_margin (0.08).
    #    Both must abstain from the person bridge while still coupling to `moto`!
    h_sym_left = _make_det(440.0, 230.0, 490.0, 280.0, "helmet")     # cx=465, dx=35 from 500
    h_sym_right = _make_det(510.0, 230.0, 560.0, 280.0, "no_helmet") # cx=535, dx=35 from 500
    hp_map, abstained = coupler.assign_heads_to_persons_bipartite([h_sym_left, h_sym_right], [p_only])
    assert len(hp_map) == 0
    assert abstained == {0, 1}

    res_ambig_hp = coupler.couple_frame(
        camera_id="front",
        frame_index=2,
        pts_timestamp_sec=0.0333,
        motorcycles=[moto],
        persons=[p_only],
        heads=[h_sym_left, h_sym_right],
    )
    cm_ambig = res_ambig_hp.coupled_motorcycles[0]
    assert cm_ambig.rider_count == 2
    assert all(r.person_detection is None for r in cm_ambig.riders)
    assert all(r.geometric_features.head_person_abstained is True for r in cm_ambig.riders)

    # 3. Ambiguous 1-head-to-2-persons case: one head at cx=500 between two symmetric person boxes
    p_left = _make_det(410.0, 220.0, 570.0, 650.0, "person")   # cx=490
    p_right = _make_det(430.0, 220.0, 590.0, 650.0, "person")  # cx=510
    h_center = _make_det(470.0, 230.0, 530.0, 285.0, "helmet") # cx=500 (equidistant to 490 and 510)
    hp_map_2p, abstained_2p = coupler.assign_heads_to_persons_bipartite([h_center], [p_left, p_right])
    assert len(hp_map_2p) == 0
    assert abstained_2p == {0}

    # 4. Orphan handling with 1-to-1 person assignment when no motorcycles are in frame
    res_orphans = coupler.couple_frame(
        camera_id="front",
        frame_index=3,
        pts_timestamp_sec=0.0500,
        motorcycles=[],
        persons=[p_only],
        heads=[h_driver, h_pillion],
    )
    assert len(res_orphans.orphan_riders) == 2
    orphan_persons = [o.person_detection for o in res_orphans.orphan_riders if o.person_detection is not None]
    assert len(orphan_persons) == 1


def test_kalman_noise_and_rider_protection_parameters_loaded_from_yaml(tmp_path):
    """
    Correction 4 Regression Test:
    Verify that Kalman filter noise parameters (`kalman_std_weight_pos`, `kalman_std_weight_vel`)
    and rider identity protection parameters are loaded from `configs/tracking_config.yaml`
    and propagated to `MotoTrack.kalman`, `RiderSubTrack.kalman`, and `OrphanRiderTrack.kalman`.
    """
    trk_default = CameraTrackingPipeline.from_yaml(camera_id="front", yaml_path="configs/tracking_config.yaml")
    assert trk_default.bt_config.kalman_std_weight_pos == pytest.approx(0.05)
    assert trk_default.bt_config.kalman_std_weight_vel == pytest.approx(0.00625)
    assert trk_default.bt_config.rider_match_min_iou == pytest.approx(0.15)
    assert trk_default.bt_config.rider_spatial_override_min_iou == pytest.approx(0.35)
    assert trk_default.bt_config.rider_ambiguity_iou_margin == pytest.approx(0.15)
    assert trk_default.bt_config.rider_max_norm_center_dist == pytest.approx(0.65)
    assert trk_default.coupler_config.head_person_ambiguity_margin == pytest.approx(0.08)

    # Test custom YAML override propagation into actual KalmanBoxFilter instances
    custom_yaml = tmp_path / "custom_tracking.yaml"
    custom_yaml.write_text(
        """
moto_rider_coupler:
  head_person_ambiguity_margin: 0.11
kalman_bytetrack:
  kalman_std_weight_pos: 0.08
  kalman_std_weight_vel: 0.0125
  rider_match_min_iou: 0.22
  rider_spatial_override_min_iou: 0.42
  rider_ambiguity_iou_margin: 0.19
  rider_max_norm_center_dist: 0.50
""",
        encoding="utf-8",
    )
    trk_custom = CameraTrackingPipeline.from_yaml(camera_id="front", yaml_path=custom_yaml)
    assert trk_custom.bt_config.kalman_std_weight_pos == pytest.approx(0.08)
    assert trk_custom.bt_config.kalman_std_weight_vel == pytest.approx(0.0125)
    assert trk_custom.coupler_config.head_person_ambiguity_margin == pytest.approx(0.11)

    b0 = _make_bundle(
        "front",
        0,
        0.0,
        [_make_det(400.0, 350.0, 600.0, 750.0, "motorcycle", conf=0.90)],
        [],
        [
            _make_det(460.0, 230.0, 540.0, 290.0, "helmet", conf=0.90),
            _make_det(100.0, 100.0, 150.0, 150.0, "no_helmet", conf=0.90),  # far orphan head
        ],
    )
    trk_custom.step(b0)
    moto = trk_custom.active_moto_tracks["front_moto_0001"]
    rider = moto.rider_subtracks[0]
    orphan = trk_custom.active_orphan_tracks["front_orphan_0001"]

    assert moto.kalman._std_weight_pos == pytest.approx(0.08)
    assert moto.kalman._std_weight_vel == pytest.approx(0.0125)
    assert rider.kalman._std_weight_pos == pytest.approx(0.08)
    assert rider.kalman._std_weight_vel == pytest.approx(0.0125)
    assert orphan.kalman._std_weight_pos == pytest.approx(0.08)
    assert orphan.kalman._std_weight_vel == pytest.approx(0.0125)

