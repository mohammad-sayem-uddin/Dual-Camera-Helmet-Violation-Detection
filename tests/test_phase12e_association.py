"""Comprehensive Unit and Integration Test Suite for Phase 12E Cross-Camera Tracklet Association.

Covers all 25 required synthetic scenarios from Section 14 plus YAML configuration
override propagation verification.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.association import (
    AppearanceConfig,
    AppearanceDescriptorRecord,
    AssociationAuditor,
    AssociationOutcome,
    CandidatePairScore,
    CrossCameraAssociationConfig,
    CrossCameraAssociationEngine,
    GlobalAssignmentConfig,
    GlobalAssignmentSolver,
    MotorcycleTracklet,
    OrphanAssociationConfig,
    RiderAssociationConfig,
    TimestampQualityRecord,
    TrackletBuilder,
    TrackletBuilderConfig,
    TrackletCompleteness,
    TransitTimeConfig,
)
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.tracking.camera_tracker import CameraTrackingPipeline


def _make_det(
    camera_id: str,
    frame_index: int,
    pts_sec: float,
    bbox: tuple[float, float, float, float],
    class_name: str,
    confidence: float = 0.88,
) -> StructuredDetection:
    cls_map = {"motorcycle": 3, "person": 0, "helmet": 1, "no_helmet": 0}
    cat = "motorcycle" if class_name == "motorcycle" else ("person" if class_name == "person" else "head")
    src = "yolov8n_coco" if class_name in ("motorcycle", "person") else "phase9_best_helmet"
    x1, y1, x2, y2 = bbox
    return StructuredDetection(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        confidence=float(confidence),
        class_id=cls_map.get(class_name, 0),
        class_name=class_name,
        category=cat,
        source_model=src,
        camera_id=camera_id,
        frame_index=int(frame_index),
        pts_timestamp_sec=float(pts_sec),
    )


def _make_bundle(
    camera_id: str,
    frame_index: int,
    pts_sec: float,
    motorcycles: list[StructuredDetection],
    persons: list[StructuredDetection],
    helmets: list[StructuredDetection],
    no_helmets: list[StructuredDetection],
) -> FrameDetectionBundle:
    return FrameDetectionBundle(
        camera_id=camera_id,
        frame_index=int(frame_index),
        pts_timestamp_sec=float(pts_sec),
        motorcycles=motorcycles,
        persons=persons,
        helmet_boxes=helmets,
        no_helmet_boxes=no_helmets,
        all_heads=list(helmets) + list(no_helmets),
        vehicle_latency_ms=0.5,
        helmet_latency_ms=0.5,
        total_latency_ms=1.0,
        device="cpu",
    )


def _feed_moto_sequence(
    pipeline: CameraTrackingPipeline,
    frames_pts: list[tuple[int, float]],
    moto_bbox_start: tuple[float, float, float, float] = (400.0, 400.0, 560.0, 620.0),
    vx: float = 8.0,
    rider_count: int = 1,
    helmet_classes: list[str] | None = None,
    ambiguous_rider_geometry: bool = False,
) -> None:
    """Helper to feed a multi-frame motorcycle + rider sequence into `CameraTrackingPipeline`."""
    cam = pipeline.camera_id
    h_classes = helmet_classes or (["helmet"] * rider_count)
    for step_k, (f_idx, pts_sec) in enumerate(frames_pts):
        dx = step_k * vx
        mx1, my1, mx2, my2 = (
            moto_bbox_start[0] + dx,
            moto_bbox_start[1],
            moto_bbox_start[2] + dx,
            moto_bbox_start[3],
        )
        motos = [_make_det(cam, f_idx, pts_sec, (mx1, my1, mx2, my2), "motorcycle", 0.90)]
        persons: list[StructuredDetection] = []
        heads: list[StructuredDetection] = []

        for r_i in range(rider_count):
            if ambiguous_rider_geometry:
                # Spatially separated horizontally (no 2D box overlap) but nearly identical
                # vertical apex (< 8 px separation) so ordering_confident == False
                hx1 = mx1 + 15.0 + r_i * 55.0
                hy1 = my1 - 65.0 + r_i * 2.0
            else:
                # Distinct vertical apex and lateral offset (> 25 px separation)
                hx1 = mx1 + 25.0 + r_i * 42.0
                hy1 = my1 - 80.0 + r_i * 24.0
            hx2 = hx1 + 38.0
            hy2 = hy1 + 40.0
            px1 = hx1 - 12.0
            py1 = hy1
            px2 = hx2 + 12.0
            py2 = my2 - 25.0
            persons.append(_make_det(cam, f_idx, pts_sec, (px1, py1, px2, py2), "person", 0.86))
            h_cls = h_classes[r_i] if r_i < len(h_classes) else "helmet"
            heads.append(_make_det(cam, f_idx, pts_sec, (hx1, hy1, hx2, hy2), h_cls, 0.88))

        bundle = _make_bundle(
            camera_id=cam,
            frame_index=f_idx,
            pts_sec=pts_sec,
            motorcycles=motos,
            persons=persons,
            helmets=[h for h in heads if h.class_name == "helmet"],
            no_helmets=[h for h in heads if h.class_name == "no_helmet"],
        )
        pipeline.step(bundle, should_update_state=True)


def _make_textured_frame(
    hue_val: int = 20,
    sat_val: int = 200,
    val_val: int = 180,
    height: int = 720,
    width: int = 1280,
) -> np.ndarray:
    """Create a textured synthetic BGR frame that passes Laplacian sharpness checks."""
    import cv2

    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue_val
    hsv[:, :, 1] = sat_val
    # Checkerboard luminance pattern so Laplacian variance >> 12.0
    yy, xx = np.indices((height, width))
    checker = ((xx // 4 + yy // 4) % 2) * 60
    hsv[:, :, 2] = np.clip(val_val - 30 + checker, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_01_single_motorcycle_single_rider_clean_cross_camera_link() -> None:
    """Scenario 1: Single motorcycle, single rider -> clean LINKED outcome across cameras."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10), (3, 1.15)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85), (3, 1.90)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(
        pair_id="syn_pair_01",
        global_offset_sec=0.0,
        epsilon_sync_ms=3.0,
        mean_delta_pts_front_ms=1.2,
        mean_delta_pts_back_ms=0.8,
    )
    res = engine.associate_pipelines("syn_pair_01", front_pipe, back_pipe, tq)

    assert len(res.motorcycle_results) == 1
    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    assert m_res.global_motorcycle_id == "global_moto_0001"
    assert m_res.front_moto_track_id == "front_moto_0001"
    assert m_res.back_moto_track_id == "back_moto_0001"
    assert len(m_res.rider_associations) == 1
    assert m_res.rider_associations[0].outcome == AssociationOutcome.LINKED
    assert m_res.rider_associations[0].front_slot_role == "driver"
    assert m_res.rider_associations[0].back_slot_role == "driver"


def test_02_single_motorcycle_two_riders_driver_and_pillion_link() -> None:
    """Scenario 2: Single motorcycle with 2 riders (driver + pillion_1) -> both riders LINKED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(front_pipe, [(0, 2.00), (1, 2.05), (2, 2.10), (3, 2.15)], rider_count=2)
    _feed_moto_sequence(back_pipe, [(0, 2.70), (1, 2.75), (2, 2.80), (3, 2.85)], rider_count=2)

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(pair_id="syn_pair_02", epsilon_sync_ms=3.0)
    res = engine.associate_pipelines("syn_pair_02", front_pipe, back_pipe, tq)

    assert len(res.motorcycle_results) == 1
    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    assert len(m_res.rider_associations) == 2
    roles = {(r.front_slot_role, r.back_slot_role): r.outcome for r in m_res.rider_associations}
    assert roles[("driver", "driver")] == AssociationOutcome.LINKED
    assert roles[("pillion_1", "pillion_1")] == AssociationOutcome.LINKED


def test_03_single_motorcycle_three_riders_triple_riding_link() -> None:
    """Scenario 3: Single motorcycle with 3 riders (driver, pillion_1, pillion_2) -> all 3 LINKED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(
        front_pipe,
        [(0, 3.00), (1, 3.05), (2, 3.10), (3, 3.15)],
        moto_bbox_start=(300.0, 400.0, 540.0, 640.0),
        rider_count=3,
    )
    _feed_moto_sequence(
        back_pipe,
        [(0, 3.75), (1, 3.80), (2, 3.85), (3, 3.90)],
        moto_bbox_start=(300.0, 400.0, 540.0, 640.0),
        rider_count=3,
    )

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(pair_id="syn_pair_03", epsilon_sync_ms=3.0)
    res = engine.associate_pipelines("syn_pair_03", front_pipe, back_pipe, tq)

    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    assert len(m_res.rider_associations) == 3
    for r_assoc in m_res.rider_associations:
        assert r_assoc.outcome == AssociationOutcome.LINKED
        assert r_assoc.front_slot_role == r_assoc.back_slot_role


def test_04_two_motorcycles_well_separated_in_time() -> None:
    """Scenario 4: Two motorcycles separated by 4.0 seconds -> two unambiguous 1-to-1 links."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    # Moto 1 around t=1.0s (1 rider), Moto 2 around t=5.5s (2 riders, spatially far)
    _feed_moto_sequence(
        front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], moto_bbox_start=(150.0, 400.0, 300.0, 600.0), rider_count=1
    )
    _feed_moto_sequence(
        front_pipe, [(30, 5.50), (31, 5.55), (32, 5.60)], moto_bbox_start=(800.0, 400.0, 980.0, 620.0), rider_count=2
    )

    _feed_moto_sequence(
        back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], moto_bbox_start=(150.0, 400.0, 300.0, 600.0), rider_count=1
    )
    _feed_moto_sequence(
        back_pipe, [(30, 6.25), (31, 6.30), (32, 6.35)], moto_bbox_start=(800.0, 400.0, 980.0, 620.0), rider_count=2
    )

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(pair_id="syn_pair_04", epsilon_sync_ms=3.0)
    res = engine.associate_pipelines("syn_pair_04", front_pipe, back_pipe, tq)

    linked = [r for r in res.motorcycle_results if r.outcome == AssociationOutcome.LINKED]
    assert len(linked) == 2
    pair_map = {r.front_moto_track_id: r.back_moto_track_id for r in linked}
    assert pair_map["front_moto_0001"] == "back_moto_0001"
    assert pair_map["front_moto_0002"] == "back_moto_0002"


def test_05_two_similar_motorcycles_close_in_time_ambiguous_abstain() -> None:
    """Scenario 5: Two nearly identical motorcycles at simultaneous timestamps -> AMBIGUOUS_ABSTAIN."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    # Two motorcycles visible simultaneously in Front and simultaneously in Back with same rider count
    for step_k, (f_idx, pts_sec) in enumerate([(0, 1.00), (1, 1.05), (2, 1.10)]):
        m1 = _make_det("front", f_idx, pts_sec, (150.0 + step_k * 5, 400.0, 300.0 + step_k * 5, 600.0), "motorcycle")
        h1 = _make_det("front", f_idx, pts_sec, (190.0 + step_k * 5, 330.0, 230.0 + step_k * 5, 370.0), "helmet")
        m2 = _make_det("front", f_idx, pts_sec, (750.0 + step_k * 5, 400.0, 900.0 + step_k * 5, 600.0), "motorcycle")
        h2 = _make_det("front", f_idx, pts_sec, (790.0 + step_k * 5, 330.0, 830.0 + step_k * 5, 370.0), "helmet")
        front_pipe.step(
            _make_bundle("front", f_idx, pts_sec, [m1, m2], [], [h1, h2], []),
            should_update_state=True,
        )

    for step_k, (f_idx, pts_sec) in enumerate([(0, 1.75), (1, 1.80), (2, 1.85)]):
        m1 = _make_det("back", f_idx, pts_sec, (200.0 + step_k * 5, 400.0, 350.0 + step_k * 5, 600.0), "motorcycle")
        h1 = _make_det("back", f_idx, pts_sec, (240.0 + step_k * 5, 330.0, 280.0 + step_k * 5, 370.0), "helmet")
        m2 = _make_det("back", f_idx, pts_sec, (800.0 + step_k * 5, 400.0, 950.0 + step_k * 5, 600.0), "motorcycle")
        h2 = _make_det("back", f_idx, pts_sec, (840.0 + step_k * 5, 330.0, 880.0 + step_k * 5, 370.0), "helmet")
        back_pipe.step(
            _make_bundle("back", f_idx, pts_sec, [m1, m2], [], [h1, h2], []),
            should_update_state=True,
        )

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(pair_id="syn_pair_05", epsilon_sync_ms=3.0)
    res = engine.associate_pipelines("syn_pair_05", front_pipe, back_pipe, tq)

    assert res.outcome_counts[AssociationOutcome.LINKED.value] == 0
    assert res.outcome_counts[AssociationOutcome.AMBIGUOUS_ABSTAIN.value] == 2


def test_06_motorcycle_in_front_only_unmatched() -> None:
    """Scenario 6: Motorcycle appears in Front only -> UNMATCHED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_06", front_pipe, back_pipe, TimestampQualityRecord())
    assert len(res.motorcycle_results) == 1
    assert res.motorcycle_results[0].outcome == AssociationOutcome.UNMATCHED
    assert res.motorcycle_results[0].front_moto_track_id == "front_moto_0001"
    assert res.motorcycle_results[0].back_moto_track_id is None


def test_07_motorcycle_in_back_only_unmatched() -> None:
    """Scenario 7: Motorcycle appears in Back only -> UNMATCHED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(back_pipe, [(0, 2.00), (1, 2.05), (2, 2.10)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_07", front_pipe, back_pipe, TimestampQualityRecord())
    assert len(res.motorcycle_results) == 1
    assert res.motorcycle_results[0].outcome == AssociationOutcome.UNMATCHED
    assert res.motorcycle_results[0].front_moto_track_id is None
    assert res.motorcycle_results[0].back_moto_track_id == "back_moto_0001"


def test_08_impossible_negative_transit_time_rejected() -> None:
    """Scenario 8: Impossible negative transit delay in front_to_back corridor -> rejected (UNMATCHED)."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    # Front at t=5.0s, Back at t=1.0s (delta_transit = -4.0s)
    _feed_moto_sequence(front_pipe, [(0, 5.00), (1, 5.05), (2, 5.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)

    cfg = CrossCameraAssociationConfig(
        transit_config=TransitTimeConfig(
            corridor_mode="front_to_back",
            min_transit_delay_sec=0.10,
            max_transit_delay_sec=5.00,
        )
    )
    engine = CrossCameraAssociationEngine(config=cfg)
    res = engine.associate_pipelines("syn_pair_08", front_pipe, back_pipe, TimestampQualityRecord())

    assert res.outcome_counts[AssociationOutcome.LINKED.value] == 0
    assert res.outcome_counts[AssociationOutcome.UNMATCHED.value] == 2
    assert not res.candidate_pair_scores[0].passed_candidate_gate
    assert (
        "IMPOSSIBLE_NEGATIVE_TRANSIT_DELAY_FRONT_TO_BACK"
        in res.candidate_pair_scores[0].gate_rejection_reasons
    )


def test_09_excessively_long_transit_delay_rejected() -> None:
    """Scenario 9: Excessively long transit delay (> max_temporal_search_window_sec) -> rejected (UNMATCHED)."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 18.00), (1, 18.05), (2, 18.10)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_09", front_pipe, back_pipe, TimestampQualityRecord())
    assert res.outcome_counts[AssociationOutcome.LINKED.value] == 0
    assert res.outcome_counts[AssociationOutcome.UNMATCHED.value] == 2
    assert (
        "EXCEEDS_MAX_TEMPORAL_SEARCH_WINDOW"
        in res.candidate_pair_scores[0].gate_rejection_reasons
    )


def test_10_pillion_occluded_in_one_camera_partial_rider_match() -> None:
    """Scenario 10: Driver + pillion in Front, pillion occluded in Back -> Moto LINKED, Driver LINKED, Pillion UNMATCHED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=2)
    _feed_moto_sequence(back_pipe, [(0, 1.70), (1, 1.75), (2, 1.80)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_10", front_pipe, back_pipe, TimestampQualityRecord())

    assert len(res.motorcycle_results) == 1
    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    outcomes_by_front_role = {r.front_slot_role: r.outcome for r in m_res.rider_associations}
    assert outcomes_by_front_role["driver"] == AssociationOutcome.LINKED
    assert outcomes_by_front_role["pillion_1"] == AssociationOutcome.UNMATCHED


def test_11_rider_identity_ambiguous_within_linked_motorcycle_abstains_at_rider_level() -> None:
    """Scenario 11: Ambiguous 2-rider ordering within a LINKED motorcycle -> Moto LINKED, Riders AMBIGUOUS_ABSTAIN."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(
        front_pipe,
        [(0, 1.00), (1, 1.05), (2, 1.10)],
        rider_count=2,
        ambiguous_rider_geometry=True,
    )
    _feed_moto_sequence(
        back_pipe,
        [(0, 1.70), (1, 1.75), (2, 1.80)],
        rider_count=2,
        ambiguous_rider_geometry=True,
    )

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_11", front_pipe, back_pipe, TimestampQualityRecord())

    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    assert len(m_res.rider_associations) == 2
    for r_res in m_res.rider_associations:
        assert r_res.outcome == AssociationOutcome.AMBIGUOUS_ABSTAIN


def test_12_helmet_class_difference_never_used_as_identity_feature() -> None:
    """Scenario 12: Helmet vs no_helmet class difference produces zero change in identity association score."""
    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(pair_id="syn_pair_12", epsilon_sync_ms=3.0)

    # Case A: both cameras detect 'helmet'
    fp_a = CameraTrackingPipeline(camera_id="front")
    bp_a = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(fp_a, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1, helmet_classes=["helmet"])
    _feed_moto_sequence(bp_a, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1, helmet_classes=["helmet"])
    res_a = engine.associate_pipelines("syn_pair_12a", fp_a, bp_a, tq)

    # Case B: Front detects 'helmet', Back detects 'no_helmet'
    fp_b = CameraTrackingPipeline(camera_id="front")
    bp_b = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(fp_b, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1, helmet_classes=["helmet"])
    _feed_moto_sequence(bp_b, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1, helmet_classes=["no_helmet"])
    res_b = engine.associate_pipelines("syn_pair_12b", fp_b, bp_b, tq)

    assert res_a.motorcycle_results[0].outcome == AssociationOutcome.LINKED
    assert res_b.motorcycle_results[0].outcome == AssociationOutcome.LINKED
    assert res_a.motorcycle_results[0].association_score == pytest.approx(
        res_b.motorcycle_results[0].association_score, abs=1e-6
    )
    assert res_a.motorcycle_results[0].rider_associations[0].association_score == pytest.approx(
        res_b.motorcycle_results[0].rider_associations[0].association_score, abs=1e-6
    )


def test_13_missing_optional_cue_triggers_weight_renormalization_not_zero_penalty() -> None:
    """Scenario 13: Unavailable optional cues (Cue C, D, E) renormalize active weights to sum to 1.0."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    # Transit delay = 0.75s == expected_transit_delay_sec, exact rider count match
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_13", front_pipe, back_pipe, TimestampQualityRecord())

    cps = res.candidate_pair_scores[0]
    assert set(cps.available_cue_names) == {"temporal_transit", "rider_cardinality"}
    assert set(cps.unavailable_cue_names) == {"appearance_hsv", "lane_spatial_context", "convoy_order"}
    eff_sum = sum(cps.cue_details[k].effective_weight for k in cps.available_cue_names)
    assert eff_sum == pytest.approx(1.0, abs=1e-6)
    assert cps.composite_score == pytest.approx(1.0, abs=1e-3)


def test_14_low_quality_or_missing_appearance_crop_cleanly_disables_cue_c() -> None:
    """Scenario 14: Flat/blurry frame fails Laplacian sharpness gate and disables Cue C cleanly."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1)

    # Uniform zero-sharpness frames (Laplacian variance = 0.0 < 12.0)
    flat_frames = {i: np.full((720, 1280, 3), 120, dtype=np.uint8) for i in range(4)}

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines(
        "syn_pair_14",
        front_pipe,
        back_pipe,
        TimestampQualityRecord(),
        front_frames_by_index=flat_frames,
        back_frames_by_index=flat_frames,
    )
    cps = res.candidate_pair_scores[0]
    assert not cps.cue_details["appearance_hsv"].available
    assert "SHARPNESS" in (cps.cue_details["appearance_hsv"].unavailable_reason or "")

    # Now provide sharp textured frames with matching HSV color -> Cue C becomes available!
    sharp_frames = {i: _make_textured_frame(hue_val=30, sat_val=210, val_val=190) for i in range(4)}
    res_sharp = engine.associate_pipelines(
        "syn_pair_14_sharp",
        front_pipe,
        back_pipe,
        TimestampQualityRecord(),
        front_frames_by_index=sharp_frames,
        back_frames_by_index=sharp_frames,
    )
    cps_sharp = res_sharp.candidate_pair_scores[0]
    assert cps_sharp.cue_details["appearance_hsv"].available
    assert (cps_sharp.cue_details["appearance_hsv"].normalized_score or 0.0) > 0.95


def test_15_predicted_only_kalman_frames_excluded_from_observed_appearance_and_geometry() -> None:
    """Scenario 15: Unmatched Kalman-predicted frames are excluded from ObservedFrameRecords."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    # Observed at frames 0, 1; missed at frames 2, 3, 4; observed at frame 5
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05)], rider_count=1)
    for f_miss, pts_miss in [(2, 1.10), (3, 1.15), (4, 1.20)]:
        front_pipe.step(
            _make_bundle("front", f_miss, pts_miss, [], [], [], []),
            should_update_state=True,
        )
    _feed_moto_sequence(
        front_pipe,
        [(5, 1.25)],
        moto_bbox_start=(440.0, 400.0, 600.0, 620.0),
        rider_count=1,
    )

    builder = TrackletBuilder()
    motos, _ = builder.extract_from_pipeline(front_pipe, TimestampQualityRecord())
    assert len(motos) == 1
    mt = motos[0]
    assert mt.observed_frame_count == 3
    assert mt.predicted_only_frame_count == 3
    assert mt.total_span_frames == 6
    assert mt.occlusion_intervals == [(2, 4)]
    assert [r.frame_index for r in mt.observed_frames] == [0, 1, 5]


def test_16_fragmented_or_one_frame_tracklet_classified_as_insufficient_evidence() -> None:
    """Scenario 16: A 1-frame tentative tracklet is classified as INSUFFICIENT_EVIDENCE, never LINKED."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    # Only 1 frame in Front (remains TENTATIVE and observed_frame_count == 1 < 2)
    _feed_moto_sequence(front_pipe, [(0, 1.00)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_16", front_pipe, back_pipe, TimestampQualityRecord())

    outcomes = {r.front_moto_track_id or r.back_moto_track_id: r.outcome for r in res.motorcycle_results}
    assert outcomes["front_moto_0001"] == AssociationOutcome.INSUFFICIENT_EVIDENCE
    assert outcomes["back_moto_0001"] == AssociationOutcome.UNMATCHED


def test_17_orphan_rider_tracklet_matched_with_orphan_level_label() -> None:
    """Scenario 17: Uncoupled OrphanRiderTracklets with strong temporal compatibility match as ORPHAN_LEVEL_LINK."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    for k, (f_idx, pts_sec) in enumerate([(0, 1.00), (1, 1.05), (2, 1.10), (3, 1.15)]):
        h = _make_det("front", f_idx, pts_sec, (300.0 + k * 4, 300.0, 340.0 + k * 4, 340.0), "no_helmet", 0.90)
        front_pipe.step(
            _make_bundle("front", f_idx, pts_sec, [], [], [], [h]),
            should_update_state=True,
        )

    for k, (f_idx, pts_sec) in enumerate([(0, 1.75), (1, 1.80), (2, 1.85), (3, 1.90)]):
        h = _make_det("back", f_idx, pts_sec, (320.0 + k * 4, 300.0, 360.0 + k * 4, 340.0), "no_helmet", 0.90)
        back_pipe.step(
            _make_bundle("back", f_idx, pts_sec, [], [], [], [h]),
            should_update_state=True,
        )

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_17", front_pipe, back_pipe, TimestampQualityRecord())

    assert len(res.orphan_results) == 1
    o_res = res.orphan_results[0]
    assert o_res.outcome == AssociationOutcome.LINKED
    assert o_res.association_level == "ORPHAN_LEVEL_LINK"
    assert o_res.implies_confirmed_motorcycle_link is False
    assert o_res.global_orphan_link_id == "global_orphan_link_0001"


def test_18_orphan_rider_tracklet_ambiguous_competitors_abstain() -> None:
    """Scenario 18: One Front orphan rider vs two simultaneous Back orphan riders -> AMBIGUOUS_ABSTAIN."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    for k, (f_idx, pts_sec) in enumerate([(0, 1.00), (1, 1.05), (2, 1.10), (3, 1.15)]):
        h = _make_det("front", f_idx, pts_sec, (300.0 + k * 4, 300.0, 340.0 + k * 4, 340.0), "helmet", 0.90)
        front_pipe.step(
            _make_bundle("front", f_idx, pts_sec, [], [], [h], []),
            should_update_state=True,
        )

    for k, (f_idx, pts_sec) in enumerate([(0, 1.75), (1, 1.80), (2, 1.85), (3, 1.90)]):
        h1 = _make_det("back", f_idx, pts_sec, (250.0 + k * 4, 300.0, 290.0 + k * 4, 340.0), "helmet", 0.90)
        h2 = _make_det("back", f_idx, pts_sec, (650.0 + k * 4, 300.0, 690.0 + k * 4, 340.0), "helmet", 0.90)
        back_pipe.step(
            _make_bundle("back", f_idx, pts_sec, [], [], [h1, h2], []),
            should_update_state=True,
        )

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_18", front_pipe, back_pipe, TimestampQualityRecord())

    assert res.orphan_outcome_counts[AssociationOutcome.LINKED.value] == 0
    assert res.orphan_outcome_counts[AssociationOutcome.AMBIGUOUS_ABSTAIN.value] == 1


def test_19_repeated_30fps_tick_does_not_create_duplicate_tracklet_observations() -> None:
    """Scenario 19: Repeated 30fps ticks (should_update_state=False) never inflate observed_frame_count."""
    back_pipe = CameraTrackingPipeline(camera_id="back")
    for f_idx, pts_sec in [(0, 1.00), (1, 1.033), (2, 1.066)]:
        m = _make_det("back", f_idx, pts_sec, (400.0, 400.0, 560.0, 620.0), "motorcycle", 0.90)
        h = _make_det("back", f_idx, pts_sec, (450.0, 330.0, 490.0, 370.0), "helmet", 0.90)
        bundle = _make_bundle("back", f_idx, pts_sec, [m], [], [h], [])
        back_pipe.step(bundle, should_update_state=True)
        # Simulate repeated 30fps frame on next 60 Hz tick
        back_pipe.step(bundle, should_update_state=False)

    builder = TrackletBuilder()
    motos, _ = builder.extract_from_pipeline(back_pipe, TimestampQualityRecord())
    assert len(motos) == 1
    assert motos[0].observed_frame_count == 3
    assert motos[0].rider_tracklets[0].observed_frame_count == 3


def test_20_phase10_temporal_states_remain_strictly_independent_after_linking() -> None:
    """Scenario 20: Linking a Front rider (CONFIRMED_VIOLATION) and Back rider (HELMET_NORMAL) never merges Phase 10 states."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    # 6 frames of no_helmet in Front -> CONFIRMED_VIOLATION
    _feed_moto_sequence(
        front_pipe,
        [(i, 1.00 + i * 0.05) for i in range(6)],
        rider_count=1,
        helmet_classes=["no_helmet"],
    )
    # 6 frames of helmet in Back -> HELMET_NORMAL
    _feed_moto_sequence(
        back_pipe,
        [(i, 1.75 + i * 0.05) for i in range(6)],
        rider_count=1,
        helmet_classes=["helmet"],
    )

    f_sub = front_pipe.active_moto_tracks["front_moto_0001"].rider_subtracks[0]
    b_sub = back_pipe.active_moto_tracks["back_moto_0001"].rider_subtracks[0]
    f_state_before = f_sub.temporal_state.value
    b_state_before = b_sub.temporal_state.value
    f_conf_before = dict(f_sub.aggregated_confidence)
    b_conf_before = dict(b_sub.aggregated_confidence)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_20", front_pipe, back_pipe, TimestampQualityRecord())

    m_res = res.motorcycle_results[0]
    assert m_res.outcome == AssociationOutcome.LINKED
    r_res = m_res.rider_associations[0]
    assert r_res.outcome == AssociationOutcome.LINKED

    # Verify per-camera Phase 10 states were never mutated or merged
    assert f_sub.temporal_state.value == f_state_before == "NO_HELMET_CONFIRMED"
    assert b_sub.temporal_state.value == b_state_before == "HELMET_CONFIRMED"
    assert f_sub.aggregated_confidence == f_conf_before
    assert b_sub.aggregated_confidence == b_conf_before
    assert r_res.front_phase10_readonly is not None
    assert r_res.back_phase10_readonly is not None
    assert r_res.front_phase10_readonly["temporal_state"] == "NO_HELMET_CONFIRMED"
    assert r_res.back_phase10_readonly["temporal_state"] == "HELMET_CONFIRMED"


def test_21_one_to_one_bipartite_constraint_prevents_many_to_one_linking() -> None:
    """Scenario 21: Three Front motorcycles and two Back motorcycles -> at most 2 LINKED, strictly 1-to-1."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")

    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], moto_bbox_start=(100.0, 400.0, 250.0, 600.0), rider_count=1)
    _feed_moto_sequence(front_pipe, [(20, 4.50), (21, 4.55), (22, 4.60)], moto_bbox_start=(500.0, 400.0, 650.0, 600.0), rider_count=2)
    _feed_moto_sequence(front_pipe, [(40, 8.50), (41, 8.55), (42, 8.60)], moto_bbox_start=(900.0, 400.0, 1050.0, 600.0), rider_count=3)

    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], moto_bbox_start=(100.0, 400.0, 250.0, 600.0), rider_count=1)
    _feed_moto_sequence(back_pipe, [(20, 5.25), (21, 5.30), (22, 5.35)], moto_bbox_start=(500.0, 400.0, 650.0, 600.0), rider_count=2)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_21", front_pipe, back_pipe, TimestampQualityRecord())

    linked = [r for r in res.motorcycle_results if r.outcome == AssociationOutcome.LINKED]
    assert len(linked) == 2
    front_linked_ids = [r.front_moto_track_id for r in linked]
    back_linked_ids = [r.back_moto_track_id for r in linked]
    assert len(set(front_linked_ids)) == 2
    assert len(set(back_linked_ids)) == 2
    assert res.outcome_counts[AssociationOutcome.UNMATCHED.value] == 1


def test_22_uncertainty_fields_remain_strictly_separate_in_all_outputs() -> None:
    """Scenario 22: epsilon_sync_ms, delta_pts_front_ms, delta_pts_back_ms, delta_transit_sec, and transit_uncertainty_sec remain separate."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.80), (1, 1.85), (2, 1.90)], rider_count=1)

    engine = CrossCameraAssociationEngine.from_yaml()
    tq = TimestampQualityRecord(
        pair_id="pair_015",
        global_offset_sec=-1.148,
        epsilon_sync_ms=37.0,
        mean_delta_pts_front_ms=2.45,
        mean_delta_pts_back_ms=8.33,
    )
    res = engine.associate_pipelines("pair_015", front_pipe, back_pipe, tq)
    cps = res.candidate_pair_scores[0]
    unc = cps.to_dict()["uncertainty_separation"]

    assert unc["epsilon_sync_ms"] == pytest.approx(37.0, abs=1e-4)
    assert unc["delta_pts_front_ms"] == pytest.approx(2.45, abs=1e-4)
    assert unc["delta_pts_back_ms"] == pytest.approx(8.33, abs=1e-4)
    assert unc["transit_uncertainty_sec"] == pytest.approx(0.35, abs=1e-4)
    # Canonical front centroid = 1.05 + (-1.148) = -0.098; back centroid = 1.85 -> delta_transit = 1.948s
    assert unc["delta_transit_sec"] == pytest.approx(1.85 - (1.05 - 1.148), abs=1e-4)


def test_23_empty_front_or_back_tracklet_list_handled_cleanly() -> None:
    """Scenario 23: Empty Front and Back pipelines produce valid empty batch result without error."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_23", front_pipe, back_pipe, TimestampQualityRecord())

    assert res.motorcycle_results == []
    assert res.orphan_results == []
    assert res.outcome_counts[AssociationOutcome.LINKED.value] == 0


def test_24_audit_record_completeness_and_json_csv_serialization(tmp_path: Path) -> None:
    """Scenario 24: Every decision is serialized to JSON and CSV with complete audit columns."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=2)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=2)

    engine = CrossCameraAssociationEngine.from_yaml()
    res = engine.associate_pipelines("syn_pair_24", front_pipe, back_pipe, TimestampQualityRecord(pair_id="syn_pair_24", epsilon_sync_ms=3.0))

    audit_rows = AssociationAuditor.build_flat_audit_rows(res)
    cue_rows = AssociationAuditor.build_cue_distribution_rows(res)
    assert len(audit_rows) == 3  # 1 motorcycle row + 2 rider subtrack rows
    assert len(cue_rows) == 5    # 5 cues evaluated for the candidate pair

    csv_path = tmp_path / "audit.csv"
    json_path = tmp_path / "summary.json"
    AssociationAuditor.write_csv(audit_rows, csv_path)
    AssociationAuditor.write_json(res.to_dict(), json_path)

    assert csv_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0.0"
    assert loaded["outcome_counts"]["LINKED"] == 1


def test_25_recovered_motorcycle_track_builds_single_tracklet_without_duplication() -> None:
    """Scenario 25: A MotoTrack that transitions CONFIRMED -> LOST -> RECOVERED produces 1 MotorcycleTracklet."""
    front_pipe = CameraTrackingPipeline(camera_id="front")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05)], rider_count=1)
    # Miss 1 frame -> LOST
    front_pipe.step(
        _make_bundle("front", 2, 1.10, [], [], [], []),
        should_update_state=True,
    )
    # Match again -> RECOVERED
    _feed_moto_sequence(
        front_pipe,
        [(3, 1.15)],
        moto_bbox_start=(416.0, 400.0, 576.0, 620.0),
        rider_count=1,
    )

    builder = TrackletBuilder()
    motos, _ = builder.extract_from_pipeline(front_pipe, TimestampQualityRecord())
    assert len(motos) == 1
    assert motos[0].moto_track_id == "front_moto_0001"
    assert motos[0].recovery_count == 1
    assert motos[0].occlusion_intervals == [(2, 2)]


def test_26_yaml_config_override_propagation_to_runtime_engine(tmp_path: Path) -> None:
    """Scenario 26: Overriding thresholds and weights in YAML propagates to CrossCameraAssociationEngine."""
    custom_yaml = tmp_path / "custom_assoc.yaml"
    custom_yaml.write_text(
        """
schema_version: "1.0.0"
tracklet_builder:
  min_observed_frames_moto:
    value: 4
    engineering_hypothesis: true
transit_time_model:
  corridor_mode:
    value: "front_to_back"
    engineering_hypothesis: true
  min_transit_delay_sec:
    value: 0.50
    engineering_hypothesis: true
  max_transit_delay_sec:
    value: 2.50
    engineering_hypothesis: true
association_scoring:
  cue_weights:
    temporal_transit:
      value: 0.70
      engineering_hypothesis: true
    rider_cardinality:
      value: 0.30
      engineering_hypothesis: true
global_assignment:
  min_association_score:
    value: 0.92
    engineering_hypothesis: true
  ambiguous_margin_threshold:
    value: 0.22
    engineering_hypothesis: true
rider_association:
  min_rider_association_score:
    value: 0.85
    engineering_hypothesis: true
""",
        encoding="utf-8",
    )

    engine = CrossCameraAssociationEngine.from_yaml(custom_yaml)
    assert engine.config.builder_config.min_observed_frames_moto == 4
    assert engine.config.transit_config.corridor_mode == "front_to_back"
    assert engine.config.transit_config.min_transit_delay_sec == pytest.approx(0.50)
    assert engine.config.transit_config.max_transit_delay_sec == pytest.approx(2.50)
    assert engine.config.scorer_config.weight_temporal_transit == pytest.approx(0.70)
    assert engine.config.scorer_config.weight_rider_cardinality == pytest.approx(0.30)
    assert engine.config.global_assignment_config.min_association_score == pytest.approx(0.92)
    assert engine.config.global_assignment_config.ambiguous_margin_threshold == pytest.approx(0.22)
    assert engine.config.rider_association_config.min_rider_association_score == pytest.approx(0.85)

    # Verify runtime behavioral change: a 3-frame tracklet is now < min_observed_frames_moto (4)
    front_pipe = CameraTrackingPipeline(camera_id="front")
    back_pipe = CameraTrackingPipeline(camera_id="back")
    _feed_moto_sequence(front_pipe, [(0, 1.00), (1, 1.05), (2, 1.10)], rider_count=1)
    _feed_moto_sequence(back_pipe, [(0, 1.75), (1, 1.80), (2, 1.85)], rider_count=1)

    res = engine.associate_pipelines("syn_pair_26", front_pipe, back_pipe, TimestampQualityRecord())
    assert res.outcome_counts[AssociationOutcome.LINKED.value] == 0


# ==============================================================================
# Phase 12E Review Correction 1 Regression Tests
# Verification of candidate gating in global Hungarian assignment,
# partial bipartite matching, and margin isolation.
# ==============================================================================


def _make_dummy_tracklet(
    camera_id: str,
    track_id: str,
    observed_frame_count: int = 5,
    completeness: TrackletCompleteness = TrackletCompleteness.COMPLETE,
) -> MotorcycleTracklet:
    return MotorcycleTracklet(
        schema_version="1.0.0",
        camera_id=camera_id,
        moto_track_id=track_id,
        lifecycle_state="CONFIRMED",
        ever_confirmed=True,
        t_start_sec=1.0,
        t_end_sec=2.0,
        t_peak_sec=1.5,
        t_centroid_sec=1.5,
        camera_pts_start_sec=1.0,
        camera_pts_end_sec=2.0,
        first_frame_index=0,
        last_frame_index=10,
        observed_frame_count=observed_frame_count,
        predicted_only_frame_count=0,
        total_span_frames=10,
        observation_density=0.5,
        stage2_low_conf_matches=0,
        recovery_count=0,
        occlusion_intervals=[],
        entry_bbox=(100.0, 100.0, 200.0, 200.0),
        exit_bbox=(200.0, 200.0, 300.0, 300.0),
        peak_area_bbox=(150.0, 150.0, 250.0, 250.0),
        entry_velocity_xy=(5.0, 5.0),
        exit_velocity_xy=(5.0, 5.0),
        mean_velocity_xy=(5.0, 5.0),
        motion_direction_unit_xy=(0.707, 0.707),
        mean_detection_confidence=0.9,
        entry_near_border=False,
        exit_near_border=False,
        rider_count_distribution={1: observed_frame_count},
        modal_rider_count=1,
        max_rider_count=1,
        rider_subtrack_count=1,
        slot_spatial_conflicts_detected=0,
        rider_abstained_updates=0,
        spatial_identity_overrides=0,
        completeness=completeness,
        completeness_reasons=[],
        timestamp_quality=TimestampQualityRecord(is_in_overlap=True),
        rider_tracklets={},
        observed_frames=[],
        appearance_descriptor=AppearanceDescriptorRecord(available=False),
        source_metadata={},
    )


def _make_candidate_score(
    front_id: str,
    back_id: str,
    score: float,
    passed: bool = True,
    reasons: list[str] | None = None,
    active_cues: int = 2,
) -> CandidatePairScore:
    return CandidatePairScore(
        front_moto_track_id=front_id,
        back_moto_track_id=back_id,
        passed_candidate_gate=passed,
        gate_rejection_reasons=reasons or ([] if passed else ["GATE_FAILED"]),
        composite_score=float(score),
        active_cues_count=active_cues,
        available_cue_names=["temporal_transit", "rider_cardinality"] if active_cues >= 2 else ["temporal_transit"],
        unavailable_cue_names=[],
        cue_details={},
        epsilon_sync_ms=3.0,
        delta_pts_front_ms=1.0,
        delta_pts_back_ms=1.0,
        delta_transit_sec=0.8,
        transit_uncertainty_sec=0.1,
    )


def test_review_correction_test_a_failed_gate_candidate_cannot_block_valid_associations() -> None:
    """Test A: A failed-gate candidate with an artificially high score cannot block valid associations or be linked."""
    f0 = _make_dummy_tracklet("front", "f0")
    f1 = _make_dummy_tracklet("front", "f1")
    b0 = _make_dummy_tracklet("back", "b0")
    b1 = _make_dummy_tracklet("back", "b1")

    # (f1, b0) has high score (0.96) but failed transit gate.
    # If not masked, Hungarian would pair (f1, b0) and (f0, b1), blocking (f0, b0) and (f1, b1).
    scores = [
        _make_candidate_score("f0", "b0", 0.78, passed=True),
        _make_candidate_score("f0", "b1", 0.65, passed=True),
        _make_candidate_score("f1", "b0", 0.96, passed=False, reasons=["TRANSIT_TIME_INFEASIBLE"]),
        _make_candidate_score("f1", "b1", 0.76, passed=True),
    ]

    solver = GlobalAssignmentSolver()
    res = solver.solve([f0, f1], [b0, b1], scores)

    linked = [r for r in res if r.outcome == AssociationOutcome.LINKED]
    assert len(linked) == 2

    pairs = {(r.front_moto_track_id, r.back_moto_track_id) for r in linked}
    assert ("f0", "b0") in pairs
    assert ("f1", "b1") in pairs
    assert ("f1", "b0") not in pairs

    # Verify strictly 1-to-1 global IDs
    global_ids = [r.global_motorcycle_id for r in linked]
    assert len(global_ids) == 2
    assert len(set(global_ids)) == 2
    assert all(gid is not None and gid.startswith("global_moto_") for gid in global_ids)


def test_review_correction_test_b_failed_gate_candidate_cannot_create_false_ambiguity() -> None:
    """Test B: A failed-gate candidate with a score close to a valid candidate is excluded from margin calculation."""
    f0 = _make_dummy_tracklet("front", "f0")
    b0 = _make_dummy_tracklet("back", "b0")
    b1 = _make_dummy_tracklet("back", "b1")

    # (f0, b0) is valid with score 0.82.
    # (f0, b1) failed candidate gate with score 0.80.
    # If (f0, b1) were included in margins, rc_margin = 0.82 - 0.80 = 0.02 < 0.10 -> false AMBIGUOUS_ABSTAIN.
    scores = [
        _make_candidate_score("f0", "b0", 0.82, passed=True),
        _make_candidate_score("f0", "b1", 0.80, passed=False, reasons=["OUTSIDE_OVERLAP_WINDOW"]),
    ]

    solver = GlobalAssignmentSolver()
    res = solver.solve([f0], [b0, b1], scores)

    r_f0 = next(r for r in res if r.front_moto_track_id == "f0")
    assert r_f0.outcome == AssociationOutcome.LINKED
    assert r_f0.back_moto_track_id == "b0"
    assert r_f0.second_best_score == 0.0
    assert r_f0.ambiguity_margin == pytest.approx(0.82, abs=1e-6)

    # Failed candidate b1 remains documented as UNMATCHED with rejection reason preserved
    r_b1 = next(r for r in res if r.back_moto_track_id == "b1")
    assert r_b1.outcome == AssociationOutcome.UNMATCHED
    assert "OUTSIDE_OVERLAP_WINDOW" in r_b1.decision_reasons
    assert r_b1.pair_score_detail is not None
    assert r_b1.pair_score_detail.composite_score == pytest.approx(0.80, abs=1e-6)


def test_review_correction_test_c_no_valid_candidate_produces_all_unmatched() -> None:
    """Test C: When all candidate pairs fail hard gates, no links are made and all tracklets are UNMATCHED."""
    f0 = _make_dummy_tracklet("front", "f0")
    f1 = _make_dummy_tracklet("front", "f1")
    b0 = _make_dummy_tracklet("back", "b0")
    b1 = _make_dummy_tracklet("back", "b1")

    scores = [
        _make_candidate_score("f0", "b0", 0.88, passed=False, reasons=["TRANSIT_TIME_INFEASIBLE"]),
        _make_candidate_score("f0", "b1", 0.92, passed=False, reasons=["EXCEEDS_MAX_DELAY"]),
        _make_candidate_score("f1", "b0", 0.75, passed=False, reasons=["OUTSIDE_OVERLAP_WINDOW"]),
        _make_candidate_score("f1", "b1", 0.80, passed=False, reasons=["TRANSIT_TIME_INFEASIBLE"]),
    ]

    solver = GlobalAssignmentSolver()
    res = solver.solve([f0, f1], [b0, b1], scores)

    assert len(res) == 4
    for r in res:
        assert r.outcome == AssociationOutcome.UNMATCHED
        assert r.global_motorcycle_id is None
        assert r.association_score == 0.0
        assert len(r.decision_reasons) > 0


def test_review_correction_test_d_partial_matching_different_camera_counts() -> None:
    """Test D: Partial matching handles different Front/Back camera counts correctly."""
    # Case 1: 3 Front tracklets, 2 Back tracklets
    f0 = _make_dummy_tracklet("front", "f0")
    f1 = _make_dummy_tracklet("front", "f1")
    f2 = _make_dummy_tracklet("front", "f2")
    b0 = _make_dummy_tracklet("back", "b0")
    b1 = _make_dummy_tracklet("back", "b1")

    scores = [
        _make_candidate_score("f0", "b0", 0.84, passed=True),
        _make_candidate_score("f1", "b1", 0.81, passed=True),
        _make_candidate_score("f2", "b0", 0.50, passed=False, reasons=["TRANSIT_TIME_INFEASIBLE"]),
        _make_candidate_score("f2", "b1", 0.40, passed=False, reasons=["TRANSIT_TIME_INFEASIBLE"]),
    ]

    solver = GlobalAssignmentSolver()
    res = solver.solve([f0, f1, f2], [b0, b1], scores)

    linked = [r for r in res if r.outcome == AssociationOutcome.LINKED]
    unmatched = [r for r in res if r.outcome == AssociationOutcome.UNMATCHED]

    assert len(linked) == 2
    assert {(r.front_moto_track_id, r.back_moto_track_id) for r in linked} == {("f0", "b0"), ("f1", "b1")}
    assert len(unmatched) == 1
    assert unmatched[0].front_moto_track_id == "f2"
    assert unmatched[0].global_motorcycle_id is None

    # Case 2: 2 Front tracklets competing for 1 Back tracklet closely -> AMBIGUOUS_ABSTAIN
    f_comp0 = _make_dummy_tracklet("front", "fc0")
    f_comp1 = _make_dummy_tracklet("front", "fc1")
    b_single = _make_dummy_tracklet("back", "bs0")

    comp_scores = [
        _make_candidate_score("fc0", "bs0", 0.70, passed=True),
        _make_candidate_score("fc1", "bs0", 0.68, passed=True),
    ]

    res_comp = solver.solve([f_comp0, f_comp1], [b_single], comp_scores)
    assert len(res_comp) == 2
    assert all(r.outcome == AssociationOutcome.AMBIGUOUS_ABSTAIN for r in res_comp)


def test_review_correction_test_e_margin_calculations_use_only_valid_candidates() -> None:
    """Test E: Margin calculations strictly use valid candidates for row, column, and 2x2 swap margins."""
    f0 = _make_dummy_tracklet("front", "f0")
    f1 = _make_dummy_tracklet("front", "f1")
    b0 = _make_dummy_tracklet("back", "b0")
    b1 = _make_dummy_tracklet("back", "b1")

    # Hand-constructed matrix:
    # (f0, b0) = 0.85 (valid)
    # (f0, b1) = 0.65 (valid)
    # (f1, b0) = 0.90 (INVALID - failed gate)
    # (f1, b1) = 0.75 (valid)
    scores = [
        _make_candidate_score("f0", "b0", 0.85, passed=True),
        _make_candidate_score("f0", "b1", 0.65, passed=True),
        _make_candidate_score("f1", "b0", 0.90, passed=False, reasons=["NEGATIVE_TRANSIT_DELAY"]),
        _make_candidate_score("f1", "b1", 0.75, passed=True),
    ]

    solver = GlobalAssignmentSolver()
    res = solver.solve([f0, f1], [b0, b1], scores)

    r_f0 = next(r for r in res if r.front_moto_track_id == "f0")
    r_f1 = next(r for r in res if r.front_moto_track_id == "f1")

    # For (f0, b0):
    # Valid row competitors: (f0, b1) with score 0.65
    # Valid col competitors: (f1, b0) is INVALID, so ignored
    # second_best = max([0.65]) = 0.65
    # rc_margin = 0.85 - 0.65 = 0.20
    # 2x2 swap cross term (f1, b0) is invalid, so swap_margin_2x2 is None
    assert r_f0.second_best_score == pytest.approx(0.65, abs=1e-6)
    assert r_f0.ambiguity_margin == pytest.approx(0.20, abs=1e-6)
    assert r_f0.swap_margin_2x2 is None
    assert r_f0.outcome == AssociationOutcome.LINKED

    # For (f1, b1):
    # Valid row competitors: (f1, b0) is INVALID, so ignored
    # Valid col competitors: (f0, b1) with score 0.65
    # second_best = max([0.65]) = 0.65
    # rc_margin = 0.75 - 0.65 = 0.10
    # swap_margin_2x2 is None
    assert r_f1.second_best_score == pytest.approx(0.65, abs=1e-6)
    assert r_f1.ambiguity_margin == pytest.approx(0.10, abs=1e-6)
    assert r_f1.swap_margin_2x2 is None
    assert r_f1.outcome == AssociationOutcome.LINKED

    # Subcase with all 4 valid candidates: verify exact 2x2 swap margin calculation
    scores_valid_4 = [
        _make_candidate_score("f0", "b0", 0.85, passed=True),
        _make_candidate_score("f0", "b1", 0.65, passed=True),
        _make_candidate_score("f1", "b0", 0.60, passed=True),
        _make_candidate_score("f1", "b1", 0.75, passed=True),
    ]
    res_valid_4 = solver.solve([f0, f1], [b0, b1], scores_valid_4)
    r4_f0 = next(r for r in res_valid_4 if r.front_moto_track_id == "f0")
    r4_f1 = next(r for r in res_valid_4 if r.front_moto_track_id == "f1")

    # 2x2 swap difference = (0.85 + 0.75) - (0.65 + 0.60) = 1.60 - 1.25 = 0.35
    assert r4_f0.swap_margin_2x2 == pytest.approx(0.35, abs=1e-6)
    assert r4_f1.swap_margin_2x2 == pytest.approx(0.35, abs=1e-6)
