"""TASK-06 (GAP-B) unit and integration tests.

Validates GAP-B1..B4 requirements per docs/phase12/ARGON_TASK05B_NEXT_RESEARCH_GAP_ASSESSMENT.md:
  - GAP-B1: Coordinate-space contract (4K->720p cropping, unknown-dims rejection,
            proportional min_dim_scale with floor 8px, track-aware keyframe retention).
  - GAP-B2: Dynamic frame dimensions (4K/1080p/mixed/None/mismatch) & border status.
  - GAP-B3: TASK-04 genuine helmet evidence propagation into RiderTracklet,
            OrphanRiderTracklet, and RiderAssociationResult + INSUFFICIENT_ACTIVE_CUES gate.
  - GAP-B4: Stage K item-level rider association extraction/scoring & script portability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.association.appearance_features import (
    AppearanceConfig,
    bhattacharyya_histogram_similarity,
    compute_crop_quality,
    extract_tracklet_appearance_descriptor,
)
from src.association.association_scorer import AssociationScorer, AssociationScorerConfig
from src.association.candidate_generator import CandidateGenerator
from src.association.coordinate_space import (
    REJ_INVALID_BOX,
    REJ_OUT_OF_FRAME,
    REJ_UNKNOWN_DIMS_MISMATCH,
    REJ_ZERO_AREA,
    crop_native_box,
)
from src.association.cross_camera_association import CrossCameraAssociationEngine
from src.association.global_assignment import GlobalAssignmentSolver
from src.association.transit_time_model import TransitTimeModel
from src.association.tracklet_builder import (
    BORDER_MISMATCH_REASON,
    BORDER_UNKNOWN_REASON,
    TrackletBuilder,
    TrackletBuilderConfig,
)
from src.association.tracklet_models import (
    GENUINE_EVIDENCE_PROVENANCE,
    ObservedFrameRecord,
    TimestampQualityRecord,
    genuine_evidence_to_dict,
)
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.pipeline.dual_camera_pipeline import should_retain_keyframe
from src.tracking.camera_tracker import CameraTrackingPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _det(
    cls_name: str,
    bbox: tuple[float, float, float, float],
    cam: str = "front",
    f: int = 0,
    conf: float = 0.85,
) -> StructuredDetection:
    cat = "motorcycle" if cls_name == "motorcycle" else ("person" if cls_name == "person" else "head")
    cid = 3 if cls_name == "motorcycle" else (0 if cls_name in ("person", "no_helmet") else 1)
    src_m = "yolov8n_coco" if cat in ("motorcycle", "person") else "phase9_best_helmet"
    return StructuredDetection(
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        confidence=conf,
        class_id=cid,
        class_name=cls_name,
        category=cat,
        source_model=src_m,
        camera_id=cam,
        frame_index=f,
        pts_timestamp_sec=f / 60.0,
    )


def _bundle(
    cam: str,
    f: int,
    motos: list[tuple[float, float, float, float]] | None = None,
    persons: list[tuple[float, float, float, float]] | None = None,
    heads: list[tuple[str, tuple[float, float, float, float], float]] | None = None,
) -> FrameDetectionBundle:
    m_list = [_det("motorcycle", b, cam, f, 0.85) for b in (motos or [])]
    p_list = [_det("person", b, cam, f, 0.85) for b in (persons or [])]
    h_list = [_det(cls, b, cam, f, conf) for cls, b, conf in (heads or [])]
    return FrameDetectionBundle(
        camera_id=cam,
        frame_index=f,
        pts_timestamp_sec=f / 60.0,
        motorcycles=m_list,
        persons=p_list,
        helmet_boxes=[h for h in h_list if h.class_name == "helmet"],
        no_helmet_boxes=[h for h in h_list if h.class_name == "no_helmet"],
        all_heads=h_list,
        vehicle_latency_ms=1.0,
        helmet_latency_ms=1.0,
        total_latency_ms=2.0,
        device="cpu",
    )


def _obs(f: int, bbox: tuple[float, float, float, float]) -> ObservedFrameRecord:
    return ObservedFrameRecord(
        frame_index=f,
        camera_pts_sec=f / 60.0,
        canonical_pts_sec=f / 60.0,
        bbox_xyxy=bbox,
        kalman_bbox_xyxy=bbox,
        confidence=0.9,
    )


def _paint_textured_rect(
    img: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    base_bgr: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    yy, xx = np.mgrid[0:bh, 0:bw]
    checker = (((xx // 2) + (yy // 2)) % 2) * 50
    for c in range(3):
        img[y1:y2, x1:x2, c] = np.clip(base_bgr[c] - 25 + checker, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GAP-B1: Coordinate-space mapping & keyframe retention
# ---------------------------------------------------------------------------


def test_gap_b1_4k_to_720p_crop_all_quadrants_and_no_top_left_distractor() -> None:
    img = np.full((720, 1280, 3), 25, dtype=np.uint8)
    # Bright red distractor in top-left corner of the 720p image
    _paint_textured_rect(img, (0, 0, 200, 200), (20, 20, 230))
    # Blue target in bottom-right region corresponding to 4K [2400, 1200, 2700, 1650] -> [800, 400, 900, 550]
    _paint_textured_rect(img, (800, 400, 900, 550), (220, 140, 20))

    res = crop_native_box(img, (2400.0, 1200.0, 2700.0, 1650.0), native_wh=(3840, 2160))
    assert res.reason is None
    assert res.crop is not None
    assert res.crop.shape == (150, 100, 3)
    # Must be blue-dominant target, NOT red distractor or empty
    assert res.crop[:, :, 0].mean() > 180
    assert res.crop[:, :, 2].mean() < 60

    # Top-left quadrant box [60, 60, 240, 240] in 4K -> [20, 20, 80, 80] in 720p (red distractor)
    tl_res = crop_native_box(img, (60.0, 60.0, 240.0, 240.0), native_wh=(3840, 2160))
    assert tl_res.reason is None and tl_res.crop is not None
    assert tl_res.crop.shape == (60, 60, 3)
    assert tl_res.crop[:, :, 2].mean() > 180


def test_gap_b1_anisotropic_scaling_and_rejection_reasons() -> None:
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # 4000x2000 -> 1000x1000 (sx=0.25, sy=0.50)
    res = crop_native_box(img, (400.0, 400.0, 800.0, 800.0), native_wh=(4000, 2000))
    assert res.reason is None
    assert res.image_box_xyxy == (100, 200, 200, 400)
    assert res.scale_xy == pytest.approx((0.25, 0.5))

    # Slight tolerance overshoot clamps cleanly
    clamp_res = crop_native_box(img, (-1.5, -1.0, 200.0, 200.0), native_wh=(1000, 1000))
    assert clamp_res.reason is None and clamp_res.image_box_xyxy == (0, 0, 200, 200)

    # Wholly out of frame
    oof = crop_native_box(img, (1500.0, 500.0, 1800.0, 700.0), native_wh=(1000, 1000))
    assert oof.crop is None and oof.reason == REJ_OUT_OF_FRAME

    # Sub-pixel zero-area after scaling vs degenerate/malformed boxes
    subpixel_zero = crop_native_box(img, (-1.0, 400.0, 1.5, 800.0), native_wh=(40000, 20000))
    assert subpixel_zero.crop is None and subpixel_zero.reason == REJ_ZERO_AREA
    degen = crop_native_box(img, (100.0, 100.0, 100.0, 200.0), native_wh=(1000, 1000))
    assert degen.crop is None and degen.reason == REJ_INVALID_BOX
    invalid = crop_native_box(img, (300.0, 100.0, 100.0, 200.0), native_wh=(1000, 1000))
    assert invalid.crop is None and invalid.reason == REJ_INVALID_BOX


def test_gap_b1_unknown_native_dims_single_and_multiframe_rejection() -> None:
    img = np.full((720, 1280, 3), 120, dtype=np.uint8)
    cfg = AppearanceConfig()

    # Single-frame out of 1280x720 bounds with native_frame_dimensions=None -> explicit rejection
    desc_single = extract_tracklet_appearance_descriptor(
        observed_frames=[_obs(0, (2200.0, 800.0, 2400.0, 1000.0))],
        frame_images_by_index={0: img},
        config=cfg,
        is_motorcycle_tracklet=True,
        native_frame_dimensions=None,
    )
    assert desc_single.available is False
    assert desc_single.unavailable_reason == REJ_UNKNOWN_DIMS_MISMATCH

    # Multi-frame transition: frame 0 looks <=1280x720, frame 1 exceeds >1280 with native_frame_dimensions=None
    img0 = np.full((720, 1280, 3), 80, dtype=np.uint8)
    _paint_textured_rect(img0, (100, 100, 300, 350), (200, 180, 50))
    desc_multi = extract_tracklet_appearance_descriptor(
        observed_frames=[
            _obs(0, (100.0, 100.0, 300.0, 350.0)),
            _obs(1, (2000.0, 800.0, 2200.0, 1050.0)),
        ],
        frame_images_by_index={0: img0, 1: img},
        config=cfg,
        is_motorcycle_tracklet=True,
        native_frame_dimensions=None,
    )
    assert desc_multi.available is False
    assert desc_multi.unavailable_reason == REJ_UNKNOWN_DIMS_MISMATCH


def test_gap_b1_proportional_min_dim_scale_and_similarity_consistency() -> None:
    cfg = AppearanceConfig(min_crop_width_px=20, min_crop_height_px=20, min_laplacian_sharpness=5.0)
    # 15x15 textured crop (scaled from 45x45 at 4K -> 720p, scale=1/3 -> eff_min = max(8, 7) = 8)
    crop15 = np.zeros((15, 15, 3), dtype=np.uint8)
    _paint_textured_rect(crop15, (0, 0, 15, 15), (180, 120, 60))
    ok_scaled, q_scaled, rej_scaled = compute_crop_quality(crop15, cfg, min_dim_scale=1.0 / 3.0)
    assert ok_scaled is True and q_scaled > 0.0 and rej_scaled is None

    # Tiny 6x6 crop falls below 8px floor even at 1/3 scale
    crop6 = crop15[:6, :6]
    ok_tiny, _, rej_tiny = compute_crop_quality(crop6, cfg, min_dim_scale=1.0 / 3.0)
    assert ok_tiny is False and rej_tiny == "CROP_BELOW_MIN_RESOLUTION"

    # Appearance similarity high for same visual target rendered at 720p vs 4K
    img_720 = np.full((720, 1280, 3), 25, dtype=np.uint8)
    _paint_textured_rect(img_720, (800, 400, 900, 560), (210, 90, 40))
    img_4k = np.full((2160, 3840, 3), 25, dtype=np.uint8)
    _paint_textured_rect(img_4k, (2400, 1200, 2700, 1680), (210, 90, 40))

    d720 = extract_tracklet_appearance_descriptor(
        [_obs(0, (2400.0, 1200.0, 2700.0, 1680.0))],
        {0: img_720},
        cfg,
        is_motorcycle_tracklet=True,
        native_frame_dimensions=(3840, 2160),
    )
    d4k = extract_tracklet_appearance_descriptor(
        [_obs(0, (2400.0, 1200.0, 2700.0, 1680.0))],
        {0: img_4k},
        cfg,
        is_motorcycle_tracklet=True,
        native_frame_dimensions=(3840, 2160),
    )
    assert d720.available is True and d4k.available is True
    sim = bhattacharyya_histogram_similarity(d720.moto_chassis_histogram, d4k.moto_chassis_histogram)
    assert sim is not None and sim >= 0.85


def test_gap_b1_should_retain_keyframe_covers_late_and_short_tracks_within_budget() -> None:
    stored: list[int] = []
    track_to_kf: dict[str, set[int]] = {}
    stride = 3
    max_kf = 24

    # Fill budget using Track A across frames 0..80
    for fidx in range(0, 81):
        retain, evicted = should_retain_keyframe(
            frame_index=fidx,
            has_detections=True,
            observed_track_ids=["track_A"],
            stored_indices=stored,
            track_to_keyframes=track_to_kf,
            keyframe_stride=stride,
            keyframe_max_per_camera=max_kf,
        )
        if retain:
            if evicted is not None:
                stored.remove(evicted)
                for s in track_to_kf.values():
                    s.discard(evicted)
            stored.append(fidx)
            track_to_kf.setdefault("track_A", set()).add(fidx)
    assert len(stored) == max_kf

    # Late short Track B appears only on frames 82 and 83 (neither divisible by stride=3!)
    for fidx in (82, 83):
        retain, evicted = should_retain_keyframe(
            frame_index=fidx,
            has_detections=True,
            observed_track_ids=["track_A", "track_B"],
            stored_indices=stored,
            track_to_keyframes=track_to_kf,
            keyframe_stride=stride,
            keyframe_max_per_camera=max_kf,
        )
        if retain:
            if evicted is not None:
                stored.remove(evicted)
                for s in track_to_kf.values():
                    s.discard(evicted)
            stored.append(fidx)
            for tid in ("track_A", "track_B"):
                track_to_kf.setdefault(tid, set()).add(fidx)

    assert len(stored) <= max_kf
    assert len(track_to_kf["track_B"]) >= 1
    assert len(track_to_kf["track_A"]) >= 1


# ---------------------------------------------------------------------------
# GAP-B2: Dynamic frame dimensions & border status
# ---------------------------------------------------------------------------


def test_gap_b2_border_classification_4k_vs_1080p_vs_none_vs_mixed() -> None:
    # Interior 4K track: x2=2200, y2=1400 (falsely flagged near 1080p border if hardcoded!)
    f_pipe = CameraTrackingPipeline("front")
    b_pipe = CameraTrackingPipeline("back")
    for f in range(4):
        f_pipe.step(
            _bundle(
                "front",
                f,
                motos=[(1900.0, 1050.0, 2200.0, 1400.0)],
                persons=[(1930.0, 820.0, 2170.0, 1340.0)],
                heads=[("helmet", (2010.0, 830.0, 2090.0, 910.0), 0.9)],
            )
        )
        # Back camera at 1920x1080 true right-border entry on frame 0
        b_box = (1880.0, 500.0, 1918.0, 800.0) if f == 0 else (1500.0, 500.0, 1700.0, 800.0)
        b_pipe.step(
            _bundle(
                "back",
                f,
                motos=[b_box],
                persons=[(b_box[0] + 10, b_box[1] - 120, b_box[2] - 10, b_box[3] - 30)],
                heads=[("helmet", (b_box[0] + 20, b_box[1] - 110, b_box[0] + 60, b_box[1] - 60), 0.9)],
            )
        )

    tq = TimestampQualityRecord(pair_id="pair_test")
    builder_default = TrackletBuilder(TrackletBuilderConfig())

    # 1. With authoritative 4K dims: interior box is NOT near border
    f_motos_4k, _ = builder_default.extract_from_pipeline(f_pipe, tq, frame_dimensions=(3840, 2160))
    assert len(f_motos_4k) == 1
    assert f_motos_4k[0].entry_near_border is False
    assert f_motos_4k[0].exit_near_border is False
    assert BORDER_UNKNOWN_REASON not in f_motos_4k[0].completeness_reasons
    assert BORDER_MISMATCH_REASON not in f_motos_4k[0].completeness_reasons

    # 2. With 1080p dims on 4K coordinates: explicit mismatch reason, NOT false positive border
    f_motos_1080, _ = builder_default.extract_from_pipeline(f_pipe, tq, frame_dimensions=(1920, 1080))
    assert f_motos_1080[0].entry_near_border is False
    assert f_motos_1080[0].exit_near_border is False
    assert BORDER_MISMATCH_REASON in f_motos_1080[0].completeness_reasons

    # 3. With None dims: explicit unknown-dims reason
    f_motos_none, _ = builder_default.extract_from_pipeline(f_pipe, tq, frame_dimensions=None)
    assert f_motos_none[0].entry_near_border is False
    assert f_motos_none[0].exit_near_border is False
    assert BORDER_UNKNOWN_REASON in f_motos_none[0].completeness_reasons

    # 4. Mixed resolution association: front=3840x2160, back=1920x1080
    engine = CrossCameraAssociationEngine.from_yaml(REPO_ROOT / "configs/cross_camera_association_config.yaml")
    res = engine.associate_pipelines(
        pair_id="pair_mixed",
        front_pipeline=f_pipe,
        back_pipeline=b_pipe,
        timestamp_quality=tq,
        front_frame_dimensions=(3840, 2160),
        back_frame_dimensions=(1920, 1080),
    )
    assert res.front_motorcycle_tracklets[0].entry_near_border is False
    assert res.back_motorcycle_tracklets[0].entry_near_border is True


def test_gap_b2_true_4k_border_and_combined_descriptor_audit_fields() -> None:
    pipe = CameraTrackingPipeline("front")
    img = np.full((720, 1280, 3), 25, dtype=np.uint8)
    _paint_textured_rect(img, (1240, 560, 1279, 718), (220, 170, 50))
    frames_by_idx = {}
    for f in range(3):
        frames_by_idx[f] = img
        pipe.step(
            _bundle(
                "front",
                f,
                motos=[(3790.0, 2000.0, 3838.0, 2158.0)],
                persons=[(3750.0, 1680.0, 3835.0, 2100.0)],
                heads=[("helmet", (3770.0, 1685.0, 3825.0, 1750.0), 0.9)],
            )
        )
    builder = TrackletBuilder(TrackletBuilderConfig())
    motos, _ = builder.extract_from_pipeline(
        pipe,
        TimestampQualityRecord(pair_id="pair_border_4k"),
        frame_images_by_index=frames_by_idx,
        frame_dimensions=(3840, 2160),
    )
    assert motos[0].entry_near_border is True
    assert motos[0].appearance_descriptor.observed_frames_total == 3
    assert motos[0].appearance_descriptor.observed_frames_with_keyframe == 3
    assert isinstance(motos[0].appearance_descriptor.crop_rejection_counts, dict)


# ---------------------------------------------------------------------------
# GAP-B3: TASK-04 genuine evidence propagation & INSUFFICIENT_ACTIVE_CUES gate
# ---------------------------------------------------------------------------


def test_gap_b3_genuine_evidence_propagated_even_after_phase10_occlusion_decay() -> None:
    f_pipe = CameraTrackingPipeline("front")
    b_pipe = CameraTrackingPipeline("back")

    for cam, pipe in [("front", f_pipe), ("back", b_pipe)]:
        # 4 genuine NO_HELMET observations, then 15 frames with NO head detections (unmatched decay)
        for f in range(4):
            pipe.step(
                _bundle(
                    cam,
                    f,
                    motos=[(1500.0, 900.0, 1800.0, 1400.0)],
                    persons=[(1520.0, 680.0, 1760.0, 1300.0)],
                    heads=[("no_helmet", (1600.0, 690.0, 1680.0, 770.0), 0.90)],
                )
            )
        for f in range(4, 19):
            pipe.step(
                _bundle(
                    cam,
                    f,
                    motos=[(1500.0, 900.0, 1800.0, 1400.0)],
                    persons=[(1520.0, 680.0, 1760.0, 1300.0)],
                    heads=[],
                )
            )
        # Also create an orphan track with 3 HELMET obs + 12 unmatched-style frames where another orphan survives
        for f in range(3):
            pipe.step(
                _bundle(
                    cam,
                    19 + f,
                    motos=[],
                    persons=[],
                    heads=[("helmet", (400.0, 400.0, 460.0, 460.0), 0.88)],
                )
            )

    engine = CrossCameraAssociationEngine.from_yaml(REPO_ROOT / "configs/cross_camera_association_config.yaml")
    res = engine.associate_pipelines(
        pair_id="pair_gapb3",
        front_pipeline=f_pipe,
        back_pipeline=b_pipe,
        timestamp_quality=TimestampQualityRecord(pair_id="pair_gapb3"),
        front_frame_dimensions=(3840, 2160),
        back_frame_dimensions=(3840, 2160),
    )

    rider = next(iter(res.front_motorcycle_tracklets[0].rider_tracklets.values()))
    # Legacy snapshot decayed/abstained, whereas genuine evidence preserved NO_HELMET
    assert rider.genuine_helmet_evidence is not None
    assert rider.track_helmet_state == "NO_HELMET"
    assert rider.coupled_observations_count == 4
    assert rider.coupled_no_helmet_count == 4
    assert rider.coupled_helmet_count == 0
    r_dict = rider.to_dict()
    assert r_dict["genuine_observation_helmet_evidence"]["available"] is True
    assert r_dict["genuine_observation_helmet_evidence"]["provenance"] == GENUINE_EVIDENCE_PROVENANCE
    # Legacy phase10_snapshot_readonly key remains untouched
    assert "phase10_snapshot_readonly" in r_dict

    # Orphan evidence propagation
    assert len(res.front_orphan_tracklets) >= 1
    orph = res.front_orphan_tracklets[0]
    assert orph.track_helmet_state == "HELMET"
    assert orph.coupled_observations_count == 3
    assert orph.to_dict()["genuine_observation_helmet_evidence"]["available"] is True

    # RiderAssociationResult readonly propagation
    assert len(res.motorcycle_results) == 1
    assert len(res.motorcycle_results[0].rider_associations) >= 1
    ra = res.motorcycle_results[0].rider_associations[0]
    assert ra.front_phase10_readonly is not None
    assert ra.front_phase10_readonly["genuine_observation_helmet_evidence"]["track_helmet_state"] == "NO_HELMET"
    if ra.back_phase10_readonly is not None:
        assert ra.back_phase10_readonly["genuine_observation_helmet_evidence"]["track_helmet_state"] == "NO_HELMET"

    # Empty / absent accumulator export safety -> UNCERTAIN with zero fabricated observations
    empty_export = genuine_evidence_to_dict(None)
    assert empty_export["available"] is False
    assert empty_export["track_helmet_state"] == "UNCERTAIN"
    assert empty_export["coupled_observations_count"] == 0


def test_gap_b3_insufficient_active_cues_gate_blocks_pair_and_solver() -> None:
    pipe_f_nohead = CameraTrackingPipeline("front")
    pipe_b_nohead = CameraTrackingPipeline("back")
    pipe_f = CameraTrackingPipeline("front")
    pipe_b = CameraTrackingPipeline("back")
    for f in range(3):
        pipe_f_nohead.step(_bundle("front", f, motos=[(1000.0, 800.0, 1300.0, 1200.0)], persons=[(1050.0, 600.0, 1250.0, 1150.0)]))
        pipe_b_nohead.step(_bundle("back", f, motos=[(1000.0, 800.0, 1300.0, 1200.0)], persons=[(1050.0, 600.0, 1250.0, 1150.0)]))
        pipe_f.step(
            _bundle(
                "front",
                f,
                motos=[(1000.0, 800.0, 1300.0, 1200.0)],
                persons=[(1050.0, 600.0, 1250.0, 1150.0)],
                heads=[("helmet", (1100.0, 610.0, 1180.0, 680.0), 0.90)],
            )
        )
        pipe_b.step(
            _bundle(
                "back",
                f,
                motos=[(1000.0, 800.0, 1300.0, 1200.0)],
                persons=[(1050.0, 600.0, 1250.0, 1150.0)],
                heads=[("helmet", (1100.0, 610.0, 1180.0, 680.0), 0.90)],
            )
        )

    builder = TrackletBuilder(TrackletBuilderConfig())
    tq = TimestampQualityRecord(pair_id="pair_gate")
    f_nohead, _ = builder.extract_from_pipeline(pipe_f_nohead, tq, frame_dimensions=(3840, 2160))
    b_nohead, _ = builder.extract_from_pipeline(pipe_b_nohead, tq, frame_dimensions=(3840, 2160))
    f_motos, _ = builder.extract_from_pipeline(pipe_f, tq, frame_dimensions=(3840, 2160))
    b_motos, _ = builder.extract_from_pipeline(pipe_b, tq, frame_dimensions=(3840, 2160))

    transit_model = TransitTimeModel()
    cand_1cue = CandidateGenerator(transit_model).generate_candidates(f_nohead, b_nohead)[0]
    cand_2cue = CandidateGenerator(transit_model).generate_candidates(f_motos, b_motos)[0]
    assert cand_1cue.passed_gate is True and cand_2cue.passed_gate is True

    # 1 active cue < default min=2 -> rejected and solver cannot LINK
    scorer_2 = AssociationScorer(transit_model=transit_model, config=AssociationScorerConfig(min_active_cues_required=2))
    score_1 = scorer_2.score_candidate_pair(cand_1cue, f_nohead, b_nohead)
    assert score_1.active_cues_count == 1
    assert "INSUFFICIENT_ACTIVE_CUES" in score_1.gate_rejection_reasons
    assert score_1.passed_candidate_gate is False
    solver = GlobalAssignmentSolver()
    assert all(r.outcome.value != "LINKED" for r in solver.solve(f_nohead, b_nohead, [score_1]))

    # 2 active cues < min=3 -> rejected and solver cannot LINK
    scorer_3 = AssociationScorer(transit_model=transit_model, config=AssociationScorerConfig(min_active_cues_required=3))
    score_short = scorer_3.score_candidate_pair(cand_2cue, f_motos, b_motos)
    assert score_short.active_cues_count == 2
    assert "INSUFFICIENT_ACTIVE_CUES" in score_short.gate_rejection_reasons
    assert score_short.passed_candidate_gate is False
    assert all(r.outcome.value != "LINKED" for r in solver.solve(f_motos, b_motos, [score_short]))

    # 2 active cues == min=2 -> gate passes
    score_ok = scorer_2.score_candidate_pair(cand_2cue, f_motos, b_motos)
    assert score_ok.active_cues_count == 2
    assert "INSUFFICIENT_ACTIVE_CUES" not in score_ok.gate_rejection_reasons
    assert score_ok.passed_candidate_gate is True


# ---------------------------------------------------------------------------
# GAP-B4: Stage K item-level rider association & script portability
# ---------------------------------------------------------------------------


def test_gap_b4_stagek_extract_and_evaluate_rider_item_level_and_no_vm_paths() -> None:
    repro_dir = REPO_ROOT / "scripts" / "reproducibility"
    forbidden = ["/home/hatch", "~/workspace", "/work/stage", "/work/postcompletion", "/work/pass1"]
    for py_path in sorted(repro_dir.glob("*.py")):
        txt = py_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in txt, f"Hardcoded path '{token}' found in {py_path.name}"

    import scripts.reproducibility.evaluate as stagek_eval

    res, _pp = stagek_eval.run_evaluation(write_files=False)
    # Existing headline metrics preserved + new additive item-level rider association metric
    assert res["association_motorcycle"]["tp"] == 10
    assert res["association_rider"]["gt_outcome_counts"]["LINKED"] == 4
    assert res["association_rider"]["pred_outcome_counts_aggregate"]["LINKED"] == 4
    item_metric = res["association_rider_item_level"]
    assert item_metric["retained_motorcycles"]["tp"] == 4
    assert item_metric["retained_motorcycles"]["fp"] == 0
    assert item_metric["retained_motorcycles"]["fn"] == 0
    assert item_metric["retained_motorcycles"]["f1"] == pytest.approx(1.0)
    assert item_metric["retained_motorcycles"]["exact_outcome_agreement"] == 17
    assert item_metric["retained_motorcycles"]["exact_outcome_agreement_rate"] == pytest.approx(1.0)
    assert item_metric["all_predicted_motorcycles"]["pred_units"] == 20
    assert item_metric["all_predicted_motorcycles"]["excluded_parent_moto_records"] == 3


def test_gap_b4_synthetic_rider_association_schema_fixture(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "reproducibility"))
    import extract as stagek_extract

    sj_pair = tmp_path / "pair_999"
    sj_pair.mkdir(parents=True)
    for cam in ("front", "back"):
        (sj_pair / f"pair_999_{cam}_detections.jsonl").write_text("", encoding="utf-8")
        (sj_pair / f"pair_999_{cam}_tracks_final.json").write_text(
            json.dumps({"moto_tracks": [], "orphan_tracks": []}), encoding="utf-8"
        )
        (sj_pair / f"pair_999_{cam}_coverage.csv").write_text("frame_index\n0\n", encoding="utf-8")
    (sj_pair / "pair_999_events.json").write_text(json.dumps([]), encoding="utf-8")
    (sj_pair / "pair_999_sync_diagnostics.json").write_text(
        json.dumps({"offset_sec": 0.0, "epsilon_sync_ms": 0.0, "drift_ms": 0.0}),
        encoding="utf-8",
    )
    (sj_pair / "pair_999_association.json").write_text(
        json.dumps(
            {
                "front_motorcycle_tracklets": [],
                "back_motorcycle_tracklets": [],
                "motorcycle_results": [
                    {
                        "front_moto_track_id": "front_moto_0001",
                        "back_moto_track_id": "back_moto_0001",
                        "outcome": "LINKED",
                        "rider_associations": [
                            {
                                "front_moto_track_id": "front_moto_0001",
                                "back_moto_track_id": "back_moto_0001",
                                "front_subtrack_id": "front_moto_0001_rider_s0",
                                "back_subtrack_id": "back_moto_0001_rider_s0",
                                "outcome": "LINKED",
                                "association_score": 0.91,
                            },
                            {
                                "front_moto_track_id": "front_moto_0001",
                                "back_moto_track_id": "back_moto_0001",
                                "front_subtrack_id": "front_moto_0001_rider_s1",
                                "back_subtrack_id": None,
                                "outcome": "UNMATCHED",
                                "association_score": 0.0,
                            },
                        ],
                    }
                ],
                "orphan_results": [],
                "outcome_counts": {"LINKED": 1, "UNMATCHED": 0, "AMBIGUOUS_ABSTAIN": 0, "INSUFFICIENT_EVIDENCE": 0},
                "rider_outcome_counts": {"LINKED": 1, "UNMATCHED": 1, "AMBIGUOUS_ABSTAIN": 0, "INSUFFICIENT_EVIDENCE": 0},
                "orphan_outcome_counts": {"LINKED": 0, "UNMATCHED": 0, "AMBIGUOUS_ABSTAIN": 0, "INSUFFICIENT_EVIDENCE": 0},
            }
        ),
        encoding="utf-8",
    )

    pred = stagek_extract.load_pred(sj_dir=tmp_path, pairs=["pair_999"])
    assert len(pred["assoc_rider"]) == 2
    assert pred["assoc_rider"][0]["front"] == "front_moto_0001_rider_s0"
    assert pred["assoc_rider"][0]["back"] == "back_moto_0001_rider_s0"
    assert pred["assoc_rider"][0]["outcome"] == "LINKED"
    assert pred["assoc_rider"][1]["outcome"] == "UNMATCHED"
