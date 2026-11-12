"""GAP-C Unit & Regression Test Suite:
Hierarchical Motorcycle-Anchored Rider Kinematics, Finite Coasting,
Conflict Unlocking, Longitudinal Role Consensus & Orphan Lifecycle Gating.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.association.cross_camera_association import CrossCameraAssociationEngine
from src.association.tracklet_models import TimestampQualityRecord
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.tracking.camera_tracker import CameraTrackingPipeline
from src.tracking.kalman_bytetrack import (
    ByteTrackConfig,
    KalmanBoxFilter,
    TrackLifecycleState,
    bbox_iou_xyxy,
)
from src.tracking.rider_kinematics import (
    RelativeKinematicsConfig,
    RiderRelativeKalmanState,
    bbox_to_moto_relative,
    clamp_bbox_to_frame,
    moto_relative_to_bbox,
    relative_distance,
    score_longitudinal_ranks,
)

REPO = Path(__file__).resolve().parents[1]
STAGEJ_DIR = REPO / "results" / "phase12_stagej"
CONFIG_PATH = REPO / "configs" / "tracking_config.yaml"


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


def _gapc_bt_config(**overrides) -> ByteTrackConfig:
    defaults = dict(
        use_motorcycle_relative_kinematics=True,
        rel_damping_factor=0.20,
        rel_pos_alpha=0.70,
        rel_size_alpha=0.50,
        rel_vel_beta=0.25,
        rel_max_vel=0.35,
        rider_rel_match_max_dist=0.58,
        rider_rel_ambiguity_margin=0.08,
        finite_rider_coast_enabled=True,
        max_rider_coast_frames=15,
        tentative_rider_coast_frames=3,
        unlock_slot_conflicts=True,
        duplicate_head_iou_threshold=0.55,
        sibling_overlap_max_iou=0.35,
        conflict_distinct_min_rel_dist=0.58,
        max_active_riders_per_moto=3,
        max_total_subtracks_per_moto=6,
        role_consensus_enabled=True,
        min_role_consensus_frames=3,
        orphan_lifecycle_gating_enabled=True,
        orphan_min_confirm_hits=2,
        orphan_tentative_max_lost_frames=3,
        orphan_tentative_min_conf=0.18,
        export_confirmed_orphans_only=True,
    )
    defaults.update(overrides)
    return ByteTrackConfig(**defaults)


def _load_stagej_stream(pair: str, cam: str):
    pdir = STAGEJ_DIR / pair
    cov_path = pdir / f"{pair}_{cam}_coverage.csv"
    det_path = pdir / f"{pair}_{cam}_detections.jsonl"
    inv_path = REPO / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.0_ai_assisted_audited" / "video_inventory.csv"
    native_wh = (3840, 2160)
    if inv_path.exists():
        with open(inv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["pair_id"] == pair and row["camera"] == cam:
                    w_s, h_s = str(row["resolution"]).lower().split("x")
                    native_wh = (int(w_s), int(h_s))
                    break

    update_flags: dict[int, bool] = {}
    pts_map: dict[int, float] = {}
    with open(cov_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_index"])
            update_flags[fi] = bool(int(row["detector_ran"]))
            pts_map[fi] = float(row["canonical_pts_sec"])

    assoc = json.loads((pdir / f"{pair}_association.json").read_text(encoding="utf-8"))
    bundles: list[tuple[FrameDetectionBundle, bool]] = []
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            fi = int(row["frame_index"])
            pts = float(row["pts_timestamp_sec"])
            motos = [
                _make_det(*r["bbox"], "motorcycle", r["confidence"], cam, fi, pts)
                for r in row.get("motorcycles", [])
            ]
            persons = [
                _make_det(*r["bbox"], "person", r["confidence"], cam, fi, pts)
                for r in row.get("persons", [])
            ]
            heads = [
                _make_det(*r["bbox"], r["class_name"], r["confidence"], cam, fi, pts)
                for r in row.get("heads", [])
            ]
            bundles.append((_make_bundle(cam, fi, pts, motos, persons, heads), update_flags.get(fi, True)))
    return bundles, native_wh, pts_map, TimestampQualityRecord(**assoc["timestamp_quality"])


# ============================================================================
# GAP-C1: Motorcycle-Anchored Relative Kinematics & Boundary Clamping
# ============================================================================


def test_bbox_to_moto_relative_and_back_roundtrip():
    moto = (1000.0, 900.0, 1400.0, 1500.0)
    head = (1140.0, 660.0, 1260.0, 810.0)
    cfg = RelativeKinematicsConfig()
    rel, valid = bbox_to_moto_relative(head, moto, cfg)
    assert valid is True
    recon, off_frame = moto_relative_to_bbox(rel, moto, frame_dimensions=(3840, 2160), config=cfg)
    assert off_frame is False
    for a, b in zip(head, recon):
        assert abs(a - b) < 1e-4


def test_perspective_scaling_looming_motorcycle_preserves_match():
    """Approaching motorcycle expands 2.2x; stationary relative rider scales proportionally."""
    moto_t0 = (1000.0, 1000.0, 1200.0, 1300.0)
    head_t0 = (1070.0, 880.0, 1130.0, 955.0)
    cfg = RelativeKinematicsConfig(vel_damping=0.20, pos_alpha=0.70, size_alpha=0.50, vel_beta=0.25)
    rel_state = RiderRelativeKalmanState(
        initial_head_bbox=head_t0,
        initial_moto_bbox=moto_t0,
        config=cfg,
        frame_dimensions=(3840, 2160),
    )

    moto_t1 = (900.0, 800.0, 1340.0, 1460.0)  # 2.2x expansion, Looming forward
    true_head_t1, _ = moto_relative_to_bbox(rel_state.rel_tuple, moto_t1, frame_dimensions=(3840, 2160), config=cfg)
    pred_head_t1 = rel_state.predict(moto_bbox=moto_t1, dt_frames=1.0, frame_dimensions=(3840, 2160))

    assert bbox_iou_xyxy(pred_head_t1, true_head_t1) > 0.70
    rel_obs, valid = bbox_to_moto_relative(true_head_t1, moto_t1, cfg)
    assert valid is True
    assert relative_distance(rel_state.rel_tuple, rel_obs) < 0.05


def test_relative_velocity_damping_decays_to_zero():
    moto = (500.0, 400.0, 700.0, 700.0)
    head = (570.0, 320.0, 630.0, 390.0)
    cfg = RelativeKinematicsConfig(vel_damping=0.20, vel_beta=0.50)
    st = RiderRelativeKalmanState(head, moto, config=cfg, frame_dimensions=(1920, 1080))
    st.update((600.0, 320.0, 660.0, 390.0), moto_bbox=moto, frame_dimensions=(1920, 1080))
    v0 = abs(st.du)
    assert v0 > 0.0
    for _ in range(4):
        st.predict(moto_bbox=moto, frame_dimensions=(1920, 1080), coasting=True)
    assert abs(st.du) < v0 * 0.05


def test_sliver_border_box_synced_bbox_prevents_kalman_oob():
    """Regression for OOB=9: sliver border boxes (w/h < 0.05) stay inside [0, W]x[0, H]."""
    kf = KalmanBoxFilter((3800.0, 240.0, 3840.0, 680.0))
    sliver = (3822.1, 247.0, 3840.0, 684.0)  # aspect = 17.9 / 437 = 0.041 < 0.05
    kf.sync_bbox_state(sliver)
    assert kf.current_bbox_xyxy[2] <= 3840.0
    assert kf.current_bbox_xyxy == pytest.approx(sliver)

    clamped, off = clamp_bbox_to_frame((3822.1, 247.0, 3860.0, 684.0), (3840, 2160))
    assert clamped[2] == 3840.0
    assert off is False

    # Anchor on a border-adjacent moto holding steady at the right frame edge stays strictly in-bounds
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(), frame_dimensions=(3840, 2160))
    for f_idx in range(3):
        m_det = _make_det(3550.0, 200.0, 3840.0, 900.0, "motorcycle", 0.90, "front", f_idx, f_idx * 0.0167)
        h_dets = [_make_det(3822.0, 247.0, 3840.0, 684.0, "no_helmet", 0.85, "front", f_idx, f_idx * 0.0167)] if f_idx == 0 else []
        pipe.step(_make_bundle("front", f_idx, f_idx * 0.0167, [m_det], [], h_dets))
    m_track = pipe.active_moto_tracks["front_moto_0001"]
    sub = m_track.rider_subtracks[0]
    assert sub.kalman.current_bbox_xyxy[2] <= 3840.0
    assert sub.off_frame_count == 0


# ============================================================================
# GAP-C2: Finite Coasting, Conflict Unlocking, Dedup & Tentative Sibling Eviction
# ============================================================================


def test_finite_rider_coast_expiration_after_horizon():
    pipe = CameraTrackingPipeline(camera_id="back", bytetrack_config=_gapc_bt_config(max_rider_coast_frames=4), frame_dimensions=(1920, 1080))
    moto = (600.0, 400.0, 820.0, 820.0)
    head = (680.0, 300.0, 740.0, 370.0)
    # 2 hits -> confirmed subtrack uses max_rider_coast_frames=4
    for f in range(2):
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "back", f)], [], [_make_det(*head, "helmet", 0.9, "back", f)]))
    for f in range(2, 8):
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "back", f)], [], []))
    sub = pipe.active_moto_tracks["back_moto_0001"].rider_subtracks[0]
    assert sub.is_coast_expired is True
    assert sub.coast_state == "COAST_EXPIRED"
    assert sub.coasted_frames_count == 4


def test_tentative_1hit_sibling_expires_early_and_pruned():
    pipe = CameraTrackingPipeline(
        camera_id="back",
        bytetrack_config=_gapc_bt_config(tentative_rider_coast_frames=3, max_rider_coast_frames=15),
        frame_dimensions=(1920, 1080),
    )
    moto = (600.0, 400.0, 850.0, 850.0)
    h0 = (680.0, 300.0, 740.0, 370.0)
    h_noise = (610.0, 410.0, 650.0, 460.0)
    for f in range(6):
        heads = [_make_det(*h0, "no_helmet", 0.9, "back", f)]
        if f == 1:
            heads.append(_make_det(*h_noise, "helmet", 0.4, "back", f))
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "back", f)], [], heads))
    m = pipe.active_moto_tracks["back_moto_0001"]
    assert list(m.rider_subtracks.keys()) == [0]
    assert len(m.discarded_tentative_subtracks) == 1
    assert m.discarded_tentative_subtracks[0].phase10_track.hits == 1


def test_dual_class_co_located_duplicate_head_suppressed():
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(), frame_dimensions=(1920, 1080))
    moto = (500.0, 400.0, 750.0, 850.0)
    for f in range(4):
        heads = [
            _make_det(580.0, 300.0, 650.0, 380.0, "no_helmet", 0.88, "front", f),
            _make_det(582.0, 302.0, 652.0, 382.0, "helmet", 0.42, "front", f),  # IoU ~ 0.90 duplicate
        ]
        pipe.step(_make_bundle("front", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "front", f)], [], heads))
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert list(m.rider_subtracks.keys()) == [0]
    assert m.rider_abstained_updates == 0
    assert m.rider_subtracks[0].phase10_track.hits == 4
    assert m.rider_subtracks[0].track_helmet_state == "NO_HELMET"


def test_single_to_multi_transition_spawns_pillion_without_abstention():
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(), frame_dimensions=(1920, 1080))
    moto = (500.0, 400.0, 760.0, 860.0)
    h_drv = (580.0, 300.0, 645.0, 375.0)
    h_pil = (660.0, 260.0, 720.0, 330.0)
    for f in range(2):
        pipe.step(_make_bundle("front", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "front", f)], [], [_make_det(*h_drv, "helmet", 0.9, "front", f)]))
    pipe.step(
        _make_bundle(
            "front",
            2,
            2 * 0.0167,
            [_make_det(*moto, "motorcycle", 0.9, "front", 2)],
            [],
            [_make_det(*h_pil, "no_helmet", 0.9, "front", 2), _make_det(*h_drv, "helmet", 0.9, "front", 2)],
        )
    )
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert m.rider_abstained_updates == 0
    assert m.rider_subtracks[0].phase10_track.hits == 3
    assert 1 in m.rider_subtracks
    assert m.rider_subtracks[1].phase10_track.hits == 1


def test_reinit_after_all_subtracks_coast_expired():
    """Verify pair_021 pattern: 1-hit early noise expires and frees s0 for real later rider."""
    pipe = CameraTrackingPipeline(
        camera_id="back",
        bytetrack_config=_gapc_bt_config(tentative_rider_coast_frames=3),
        frame_dimensions=(3840, 2160),
    )
    moto = (2800.0, 1200.0, 3500.0, 2100.0)
    noise_head = (3350.0, 1450.0, 3440.0, 1550.0)
    real_head = (3020.0, 800.0, 3220.0, 1020.0)
    pipe.step(_make_bundle("back", 0, 0.0, [_make_det(*moto, "motorcycle", 0.9, "back", 0)], [], [_make_det(*noise_head, "no_helmet", 0.4, "back", 0)]))
    for f in range(1, 5):
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "back", f)], [], []))
    for f in range(5, 10):
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*moto, "motorcycle", 0.9, "back", f)], [], [_make_det(*real_head, "helmet", 0.9, "back", f)]))
    m = pipe.active_moto_tracks["back_moto_0001"]
    assert 0 in m.rider_subtracks
    assert m.rider_subtracks[0].phase10_track.hits == 5
    assert m.rider_subtracks[0].track_helmet_state == "HELMET"
    assert len(m.discarded_tentative_subtracks) == 1


def test_max_active_and_total_subtracks_per_moto_enforced():
    pipe = CameraTrackingPipeline(
        camera_id="front",
        bytetrack_config=_gapc_bt_config(max_active_riders_per_moto=2, max_total_subtracks_per_moto=2, max_rider_coast_frames=2),
        frame_dimensions=(3840, 2160),
    )
    moto = (1000.0, 800.0, 2000.0, 1800.0)
    heads_two = [
        _make_det(1100.0, 500.0, 1200.0, 620.0, "helmet", 0.9, "front", 0),
        _make_det(1400.0, 480.0, 1500.0, 600.0, "no_helmet", 0.9, "front", 0),
    ]
    pipe.step(_make_bundle("front", 0, 0.0, [_make_det(*moto, "motorcycle", 0.9, "front", 0)], [], heads_two))
    heads_three = [
        _make_det(1100.0, 500.0, 1200.0, 620.0, "helmet", 0.9, "front", 1),
        _make_det(1400.0, 480.0, 1500.0, 600.0, "no_helmet", 0.9, "front", 1),
        _make_det(1700.0, 460.0, 1800.0, 580.0, "helmet", 0.9, "front", 1),
    ]
    pipe.step(_make_bundle("front", 1, 0.0167, [_make_det(*moto, "motorcycle", 0.9, "front", 1)], [], heads_three))
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert len(m.rider_subtracks) == 2


# ============================================================================
# GAP-C3: Longitudinal Role Consensus & Ordering Confidence Persistence
# ============================================================================


def test_score_longitudinal_ranks_motion_vector_back_vs_front():
    centroids = {0: (500.0, 400.0), 1: (620.0, 420.0)}
    # Moving right (+dx): larger x is leading ("driver") in both cameras
    ordered_f, mode_f = score_longitudinal_ranks(centroids, camera_id="front", velocity_xy=(12.0, 1.0))
    assert mode_f == "motion_vector"
    assert ordered_f == [1, 0]

    # Moving left (-dx): smaller x is leading ("driver")
    ordered_b, mode_b = score_longitudinal_ranks(centroids, camera_id="back", velocity_xy=(-12.0, -1.0))
    assert mode_b == "motion_vector"
    assert ordered_b == [0, 1]


def test_role_consensus_tie_yields_uncertain():
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(min_role_consensus_frames=2), frame_dimensions=(1920, 1080))
    moto = (500.0, 400.0, 800.0, 800.0)
    # 2 frames where A leads, then 2 frames where B leads -> exact vote tie -> uncertain
    h_left = (550.0, 280.0, 610.0, 350.0)
    h_right = (680.0, 240.0, 740.0, 310.0)
    for f in range(4):
        mx_shift = 20.0 * f if f < 2 else -20.0 * (f - 2)
        m_box = (moto[0] + mx_shift, moto[1], moto[2] + mx_shift, moto[3])
        if f < 2:
            b1 = (h_left[0] + mx_shift, h_left[1], h_left[2] + mx_shift, h_left[3])
            b2 = (h_right[0] + mx_shift, h_right[1], h_right[2] + mx_shift, h_right[3])
        else:
            b1 = (h_right[0] + mx_shift, h_right[1], h_right[2] + mx_shift, h_right[3])
            b2 = (h_left[0] + mx_shift, h_left[1], h_left[2] + mx_shift, h_left[3])
        pipe.step(_make_bundle("front", f, f * 0.0167, [_make_det(*m_box, "motorcycle", 0.9, "front", f)], [], [_make_det(*b1, "helmet", 0.9, "front", f), _make_det(*b2, "no_helmet", 0.9, "front", f)]))
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert m.rider_subtracks[0].slot_role in ("uncertain", "driver", "pillion_1")
    for sub in m.rider_subtracks.values():
        assert {e["slot_role"] for e in sub.observation_log} == {sub.slot_role}


def test_transient_1hit_sibling_never_demotes_confirmed_driver():
    """Regression for pair_018/front: 1-hit transient sibling never demotes confirmed driver."""
    pipe = CameraTrackingPipeline(
        camera_id="front",
        bytetrack_config=_gapc_bt_config(tentative_rider_coast_frames=2, max_rider_coast_frames=4, min_role_consensus_frames=3),
        frame_dimensions=(3840, 2160),
    )
    # Moto moves steadily RIGHT; s0 leads s1 over frames 0..4, then s1 vanishes & expires
    for f in range(10):
        mx = 1000.0 + f * 30.0
        m_box = (mx, 500.0, mx + 500.0, 1100.0)
        heads = [_make_det(mx + 300.0, 350.0, mx + 380.0, 460.0, "no_helmet", 0.9, "front", f)]  # s0 leading
        if f < 4:
            heads.append(_make_det(mx + 120.0, 320.0, mx + 200.0, 420.0, "no_helmet", 0.85, "front", f))  # s1 trailing
        if f == 9:
            # 1-hit spurious sibling ahead of s0
            heads.append(_make_det(mx + 440.0, 360.0, mx + 495.0, 450.0, "helmet", 0.5, "front", f))
        pipe.step(_make_bundle("front", f, f * 0.0167, [_make_det(*m_box, "motorcycle", 0.9, "front", f)], [], heads))
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert m.rider_subtracks[0].slot_role == "driver"
    assert m.rider_subtracks[0].ordering_confident is True


def test_single_surviving_rider_after_ghost_promoted_to_driver():
    """Regression for pair_018/back & pair_012/front: transient ghost does not leave sole rider as pillion."""
    pipe = CameraTrackingPipeline(
        camera_id="back",
        bytetrack_config=_gapc_bt_config(tentative_rider_coast_frames=2, min_role_consensus_frames=3),
        frame_dimensions=(3840, 2160),
    )
    for f in range(8):
        mx = 2000.0 - f * 30.0
        m_box = (mx, 500.0, mx + 450.0, 1100.0)
        heads = [_make_det(mx + 180.0, 350.0, mx + 260.0, 450.0, "helmet", 0.9, "back", f)]
        if f == 0:
            heads.append(_make_det(mx + 60.0, 260.0, mx + 120.0, 340.0, "no_helmet", 0.45, "back", f))
        pipe.step(_make_bundle("back", f, f * 0.0167, [_make_det(*m_box, "motorcycle", 0.9, "back", f)], [], heads))
    m = pipe.active_moto_tracks["back_moto_0001"]
    assert len(m.rider_subtracks) == 1
    sub_only = next(iter(m.rider_subtracks.values()))
    assert sub_only.slot_role == "driver"
    assert sub_only.ordering_confident is True
    assert sub_only.slot_role_provisional is False
    assert {e["slot_role"] for e in sub_only.observation_log} == {"driver"}


def test_single_rider_frame_preserves_established_ordering_confidence():
    """1-rider frame (obs.ordering_confident=False) must not erase multi-rider consensus confidence."""
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(min_role_consensus_frames=3), frame_dimensions=(1920, 1080))
    for f in range(6):
        mx = 400.0 + f * 25.0
        m_box = (mx, 400.0, mx + 400.0, 900.0)
        heads = [_make_det(mx + 240.0, 260.0, mx + 310.0, 360.0, "no_helmet", 0.9, "front", f)]
        if f < 4:
            heads.append(_make_det(mx + 100.0, 240.0, mx + 170.0, 340.0, "helmet", 0.9, "front", f))
        pipe.step(_make_bundle("front", f, f * 0.0167, [_make_det(*m_box, "motorcycle", 0.9, "front", f)], [], heads))
    m = pipe.active_moto_tracks["front_moto_0001"]
    assert m.rider_subtracks[0].ordering_confident is True
    assert m.rider_subtracks[1].ordering_confident is True


# ============================================================================
# GAP-C4: Orphan Lifecycle Gating
# ============================================================================


def test_low_conf_1hit_orphan_expired_to_discarded_tentative():
    pipe = CameraTrackingPipeline(camera_id="back", bytetrack_config=_gapc_bt_config(), frame_dimensions=(1920, 1080))
    # Frame 0: low-conf orphan head with no motorcycle; frames 1..5: empty
    pipe.step(_make_bundle("back", 0, 0.0, [], [], [_make_det(200.0, 200.0, 260.0, 270.0, "no_helmet", 0.28, "back", 0)]))
    for f in range(1, 6):
        pipe.step(_make_bundle("back", f, f * 0.0167, [], [], []))
    assert len(pipe.exported_orphan_tracks()) == 0
    assert len(pipe.discarded_tentative_orphan_tracks) == 1
    assert pipe.discarded_tentative_orphan_tracks[0].phase10_track.hits == 1


def test_confirmed_and_ge3_hit_orphans_preserved_in_export():
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(), frame_dimensions=(1920, 1080))
    for f in range(4):
        pipe.step(_make_bundle("front", f, f * 0.0167, [], [], [_make_det(300.0, 250.0, 360.0, 320.0, "helmet", 0.72, "front", f)]))
    for f in range(4, 15):
        pipe.step(_make_bundle("front", f, f * 0.0167, [], [], []))
    exported = pipe.exported_orphan_tracks()
    assert len(exported) == 1
    assert exported[0].phase10_track.hits == 4
    assert exported[0].track_helmet_state == "HELMET"
    assert len(pipe.discarded_tentative_orphan_tracks) == 0


def test_duplicate_orphan_suppression_prevents_double_spawn():
    pipe = CameraTrackingPipeline(camera_id="front", bytetrack_config=_gapc_bt_config(), frame_dimensions=(1920, 1080))
    pipe.step(_make_bundle("front", 0, 0.0, [], [], [_make_det(400.0, 300.0, 470.0, 380.0, "helmet", 0.85, "front", 0)]))
    pipe.step(_make_bundle("front", 1, 0.0167, [], [], [
        _make_det(402.0, 302.0, 472.0, 382.0, "helmet", 0.85, "front", 1),
        _make_det(405.0, 303.0, 475.0, 383.0, "no_helmet", 0.45, "front", 1),
    ]))
    assert pipe.duplicate_orphan_suppressed >= 1
    assert len(pipe.active_orphan_tracks) == 1


# ============================================================================
# YAML Config & Real Stage J Forensic Regressions
# ============================================================================


def test_tracking_config_yaml_loads_gapc_parameters():
    assert CONFIG_PATH.exists()
    pipe = CameraTrackingPipeline.from_yaml("front", yaml_path=CONFIG_PATH)
    assert pipe.bt_config.use_motorcycle_relative_kinematics is True
    assert pipe.bt_config.rel_damping_factor == pytest.approx(0.20)
    assert pipe.bt_config.rel_pos_alpha == pytest.approx(0.70)
    assert pipe.bt_config.rel_size_alpha == pytest.approx(0.50)
    assert pipe.bt_config.rel_vel_beta == pytest.approx(0.25)
    assert pipe.bt_config.rider_rel_match_max_dist == pytest.approx(0.58)
    assert pipe.bt_config.tentative_rider_coast_frames == 3
    assert pipe.bt_config.duplicate_head_iou_threshold == pytest.approx(0.55)
    assert pipe.bt_config.unlock_slot_conflicts is True
    assert pipe.bt_config.role_consensus_enabled is True
    assert pipe.bt_config.orphan_lifecycle_gating_enabled is True
    assert pipe.bt_config.export_confirmed_orphans_only is True


@pytest.mark.skipif(not STAGEJ_DIR.exists(), reason="Cached Stage J artifacts not present")
def test_stagej_forensic_focus_pairs_real_replay():
    """Verify GAP-C recovery on real cached Stage J failure fixtures without video decode."""
    results = {}
    for pair, cam, target_mid in [
        ("pair_002", "front", "front_moto_0001"),
        ("pair_006", "back", "back_moto_0002"),
        ("pair_012", "back", "back_moto_0002"),
        ("pair_019", "front", "front_moto_0002"),
        ("pair_021", "back", "back_moto_0001"),
    ]:
        bundles, wh, _, _ = _load_stagej_stream(pair, cam)
        pipe = CameraTrackingPipeline.from_yaml(cam, yaml_path=CONFIG_PATH, frame_dimensions=wh)
        for b, upd in bundles:
            pipe.step(b, should_update_state=upd)
        motos = list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values())
        m = next(x for x in motos if x.moto_track_id == target_mid)
        results[(pair, cam, target_mid)] = m
        for sub in m.rider_subtracks.values():
            assert {e["slot_role"] for e in sub.observation_log} == {sub.slot_role}
            for e in sub.observation_log:
                kb = e.get("kalman_bbox")
                if kb is not None:
                    assert -1e-6 <= kb[0] <= kb[2] <= wh[0] + 1e-6
                    assert -1e-6 <= kb[1] <= kb[3] <= wh[1] + 1e-6

    assert results[("pair_002", "front", "front_moto_0001")].rider_subtracks[0].phase10_track.hits >= 250
    assert results[("pair_002", "front", "front_moto_0001")].rider_subtracks[0].track_helmet_state == "NO_HELMET"
    assert results[("pair_006", "back", "back_moto_0002")].rider_subtracks[0].phase10_track.hits >= 100
    assert results[("pair_012", "back", "back_moto_0002")].rider_subtracks[0].phase10_track.hits >= 150
    assert results[("pair_012", "back", "back_moto_0002")].rider_subtracks[0].track_helmet_state == "HELMET"
    assert max(s.phase10_track.hits for s in results[("pair_019", "front", "front_moto_0002")].rider_subtracks.values()) >= 200
    assert sum(s.phase10_track.hits for s in results[("pair_021", "back", "back_moto_0001")].rider_subtracks.values()) >= 130
    assert results[("pair_021", "back", "back_moto_0001")].rider_abstained_updates <= 10

    # Verify pair_012 cross-camera rider association now LINKs cleanly (was INSUFFICIENT_EVIDENCE)
    f_b, f_wh, f_pts, tq = _load_stagej_stream("pair_012", "front")
    b_b, b_wh, b_pts, _ = _load_stagej_stream("pair_012", "back")
    fp = CameraTrackingPipeline.from_yaml("front", yaml_path=CONFIG_PATH, frame_dimensions=f_wh)
    bp = CameraTrackingPipeline.from_yaml("back", yaml_path=CONFIG_PATH, frame_dimensions=b_wh)
    for b, upd in f_b:
        fp.step(b, should_update_state=upd)
    for b, upd in b_b:
        bp.step(b, should_update_state=upd)
    assoc_res = CrossCameraAssociationEngine.from_yaml().associate_pipelines(
        pair_id="pair_012",
        front_pipeline=fp,
        back_pipeline=bp,
        timestamp_quality=tq,
        front_canonical_pts_by_frame=f_pts,
        back_canonical_pts_by_frame=b_pts,
        front_frame_dimensions=f_wh,
        back_frame_dimensions=b_wh,
    ).to_dict()
    linked_motos = [mr for mr in assoc_res["motorcycle_results"] if mr["outcome"] == "LINKED"]
    assert len(linked_motos) == 1
    linked_riders = [ra for ra in linked_motos[0]["rider_associations"] if ra["outcome"] == "LINKED"]
    assert len(linked_riders) == 1
    assert linked_riders[0]["front_subtrack_id"] == "front_moto_0004_rider_s0"
    assert linked_riders[0]["back_subtrack_id"] == "back_moto_0002_rider_s0"
