"""TASK-10 Cue C coordinate-space, resolution invariance, and real-video regression tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.association.appearance_features import (
    AppearanceConfig,
    bhattacharyya_histogram_similarity,
    extract_tracklet_appearance_descriptor,
    score_appearance_similarity,
)
from src.association.coordinate_space import (
    REJ_OUT_OF_FRAME,
    REJ_UNKNOWN_DIMS_MISMATCH,
)
from src.association.tracklet_models import ObservedFrameRecord
from src.pipeline.dual_camera_pipeline import _resize_keyframe
from src.video.video_reader import VideoReader

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGEJ_DIR = REPO_ROOT / "results" / "phase12_stagej"
DATA_ROOT = REPO_ROOT / "data" / "dual_camera_test_v1"
TASK10_RESULTS = REPO_ROOT / "results" / "phase12_task10" / "task10_cue_c_results.json"


def _find_first_moto_obs(pair_id: str, cam: str) -> tuple[int, tuple[float, float, float, float]]:
    det_path = STAGEJ_DIR / pair_id / f"{pair_id}_{cam}_detections.jsonl"
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            motos = row.get("motorcycles") or []
            if motos:
                return int(row["frame_index"]), tuple(float(v) for v in motos[0]["bbox"])
    raise AssertionError(f"No motorcycle detection found in {det_path}")


def test_task10_bounded_real_video_pair008_720p_vs_4k_invariance() -> None:
    """Decode <= 2 real keyframes from pair_008 via PTS-indexed VideoReader and verify 720p vs 4K parity."""
    video_front = DATA_ROOT / "front" / "pair_008_front.mp4"
    video_back = DATA_ROOT / "back" / "pair_008_back.mp4"
    if not (video_front.exists() and video_back.exists()):
        pytest.skip("Real video pair_008 files not present")

    cfg = AppearanceConfig()
    for cam, vpath in (("front", video_front), ("back", video_back)):
        fidx, bbox = _find_first_moto_obs("pair_008", cam)
        reader = VideoReader(vpath, use_pts_index=True)
        try:
            frame_4k = reader.get_frame_at_index(fidx)
            assert frame_4k is not None
            assert (reader.width, reader.height) == (3840, 2160)
            assert reader.decode_call_count == 1
        finally:
            reader.close()

        frame_720 = _resize_keyframe(frame_4k, 1280)
        assert frame_720.shape[:2] == (720, 1280)

        obs = [
            ObservedFrameRecord(
                frame_index=fidx,
                camera_pts_sec=fidx / 60.0,
                canonical_pts_sec=fidx / 60.0,
                bbox_xyxy=bbox,
                kalman_bbox_xyxy=bbox,
                confidence=0.85,
            )
        ]
        d4k = extract_tracklet_appearance_descriptor(
            obs,
            {fidx: frame_4k},
            cfg,
            is_motorcycle_tracklet=True,
            native_frame_dimensions=(3840, 2160),
        )
        d720 = extract_tracklet_appearance_descriptor(
            obs,
            {fidx: frame_720},
            cfg,
            is_motorcycle_tracklet=True,
            native_frame_dimensions=(3840, 2160),
        )
        assert d4k.available is True
        assert d720.available is True
        assert d4k.crop_rejection_counts.get(REJ_OUT_OF_FRAME, 0) == 0
        assert d720.crop_rejection_counts.get(REJ_OUT_OF_FRAME, 0) == 0

        sim = bhattacharyya_histogram_similarity(
            d4k.moto_chassis_histogram,
            d720.moto_chassis_histogram,
        )
        assert sim is not None and sim >= 0.95

        # Negative control: missing native_frame_dimensions on 4K box over 720p frame must be rejected
        if bbox[2] > 1280.0 or bbox[3] > 720.0:
            d_none = extract_tracklet_appearance_descriptor(
                obs,
                {fidx: frame_720},
                cfg,
                is_motorcycle_tracklet=True,
                native_frame_dimensions=None,
            )
            assert d_none.available is False
            assert d_none.unavailable_reason == REJ_UNKNOWN_DIMS_MISMATCH


def test_task10_zone_similarities_dict_structure_and_finite() -> None:
    """Ensure score_appearance_similarity populates zone_similarities cleanly with finite bounds."""
    cfg = AppearanceConfig()
    h = np.zeros(56, dtype=float)
    h[0] = 0.6
    h[48] = 0.4
    d1 = extract_tracklet_appearance_descriptor([], None, cfg)
    assert d1.available is False

    from src.association.tracklet_models import AppearanceDescriptorRecord

    a = AppearanceDescriptorRecord(
        available=True,
        valid_observed_crops_count=3,
        mean_crop_quality=0.95,
        cranial_cap_histogram=h.tolist(),
        upper_torso_histogram=h.tolist(),
        moto_chassis_histogram=h.tolist(),
    )
    ok, sim, qual, reason, diag = score_appearance_similarity(a, a, config=cfg)
    assert ok is True and reason is None
    assert sim == pytest.approx(1.0)
    assert qual == pytest.approx(0.95)
    assert set(diag["zone_similarities"].keys()) == {"cranial_cap", "upper_torso", "moto_chassis"}


def test_task10_committed_artifact_invariants() -> None:
    """Validate committed results/phase12_task10/task10_cue_c_results.json invariants."""
    if not TASK10_RESULTS.exists():
        pytest.skip("TASK-10 results JSON not generated yet")
    data = json.loads(TASK10_RESULTS.read_text(encoding="utf-8"))
    assert data["bounded_pairs"] == ["pair_005", "pair_008", "pair_009", "pair_019"]
    assert data["all_checks_passed"] is True
    assert all(data["checks"].values())
    assert data["aggregate_arms"]["C2_A4_720p_kf"]["by_entity_type"]["motorcycle"]["availability_rate"] == 1.0
    assert data["aggregate_arms"]["C2_A4_720p_kf"]["by_entity_type"]["rider"]["availability_rate"] == 1.0
    assert data["aggregate_arms"]["C2_A4_720p_kf"]["by_entity_type"]["orphan"]["availability_rate"] == 1.0
    assert data["resolution_invariance_summary"]["mean_composite_self_similarity_720p_vs_4k"] >= 0.98
