"""Integration and regression tests for final project defect fixes (D1-D6)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import scripts.reproducibility.evaluate as stagek_eval
import scripts.reproducibility.evaluate_task11_fusion as task11_eval
import src.detection.camera_detector as cam_det
import src.pipeline.dual_camera_pipeline as dcp
from src.association.orphan_association import OrphanAssociationConfig, OrphanAssociationEngine
from src.association.tracklet_models import (
    AppearanceDescriptorRecord,
    AssociationOutcome,
    MotorcycleAssociationResult,
    MotorcycleTracklet,
    OrphanRiderTracklet,
    RiderTracklet,
    TrackletCompleteness,
)
from src.association.transit_time_model import TransitTimeConfig, TransitTimeModel
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.fusion.evidence_fusion import FusionConfig
from src.temporal.temporal_engine import TemporalEngineConfig, TemporalVerificationEngine
from src.tracking.camera_tracker import CameraTrackingPipeline
from tests.test_phase12_pipeline_synthetic import FakePerCameraDetector, _build_dataset, _config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_rider_tracklet(
    cam: str,
    sub_id: str,
    parent_id: str,
    t0: float,
    t1: float,
    app: AppearanceDescriptorRecord,
    observed_frame_count: int = 10,
) -> RiderTracklet:
    return RiderTracklet(
        schema_version="1.0.0",
        camera_id=cam,
        subtrack_id=sub_id,
        parent_moto_track_id=parent_id,
        slot_index=0,
        slot_role="driver",
        ordering_confident=True,
        t_start_sec=t0,
        t_end_sec=t1,
        t_centroid_sec=0.5 * (t0 + t1),
        camera_pts_start_sec=t0,
        camera_pts_end_sec=t1,
        observed_frame_count=int(observed_frame_count),
        total_span_frames=int(observed_frame_count),
        missed_frame_count=0,
        observation_density=1.0,
        occlusion_intervals=[],
        entry_head_bbox=(400.0, 200.0, 440.0, 240.0),
        exit_head_bbox=(420.0, 200.0, 460.0, 240.0),
        mean_head_confidence=0.85,
        mean_coupling_score=0.90,
        abstained_updates_count=0,
        phase10_temporal_state="UNKNOWN",
        phase10_aggregated_confidence={"helmet": 0.85, "no_helmet": 0.15},
        phase10_consecutive_no_helmet_frames=0,
        phase10_consecutive_helmet_frames=5,
        phase10_alerts_count=0,
        completeness=TrackletCompleteness.COMPLETE,
        appearance_descriptor=app,
        genuine_helmet_evidence={"track_helmet_state": "HELMET", "coupled_observations_count": 10},
    )


def _make_moto_tracklet(cam: str, moto_id: str, t0: float, t1: float, rider: RiderTracklet) -> MotorcycleTracklet:
    return MotorcycleTracklet(
        schema_version="1.0.0",
        camera_id=cam,
        moto_track_id=moto_id,
        lifecycle_state="CONFIRMED",
        ever_confirmed=True,
        t_start_sec=t0,
        t_end_sec=t1,
        t_peak_sec=0.5 * (t0 + t1),
        t_centroid_sec=0.5 * (t0 + t1),
        camera_pts_start_sec=t0,
        camera_pts_end_sec=t1,
        first_frame_index=1,
        last_frame_index=10,
        observed_frame_count=10,
        predicted_only_frame_count=0,
        total_span_frames=10,
        observation_density=1.0,
        stage2_low_conf_matches=0,
        recovery_count=0,
        occlusion_intervals=[],
        entry_bbox=(360.0, 240.0, 500.0, 420.0),
        exit_bbox=(380.0, 240.0, 520.0, 420.0),
        peak_area_bbox=(370.0, 240.0, 510.0, 420.0),
        entry_velocity_xy=(2.0, 0.0),
        exit_velocity_xy=(2.0, 0.0),
        mean_velocity_xy=(2.0, 0.0),
        motion_direction_unit_xy=(1.0, 0.0),
        mean_detection_confidence=0.88,
        entry_near_border=False,
        exit_near_border=False,
        rider_count_distribution={1: 10},
        modal_rider_count=1,
        max_rider_count=1,
        rider_subtrack_count=1,
        slot_spatial_conflicts_detected=0,
        rider_abstained_updates=0,
        spatial_identity_overrides=0,
        completeness=TrackletCompleteness.COMPLETE,
        completeness_reasons=[],
        rider_tracklets={0: rider},
    )


def _make_orphan_tracklet(cam: str, oid: str, t0: float, t1: float, app: AppearanceDescriptorRecord) -> OrphanRiderTracklet:
    return OrphanRiderTracklet(
        schema_version="1.0.0",
        camera_id=cam,
        orphan_track_id=oid,
        lifecycle_state="CONFIRMED",
        last_rejection_reason="UNCOUPLED_NO_MOTO",
        t_start_sec=t0,
        t_end_sec=t1,
        t_centroid_sec=0.5 * (t0 + t1),
        camera_pts_start_sec=t0,
        camera_pts_end_sec=t1,
        observed_frame_count=6,
        total_span_frames=6,
        missed_frame_count=0,
        observation_density=1.0,
        occlusion_intervals=[],
        entry_head_bbox=(400.0, 200.0, 440.0, 240.0),
        exit_head_bbox=(420.0, 200.0, 460.0, 240.0),
        mean_velocity_xy=(2.0, 0.0),
        mean_head_confidence=0.85,
        phase10_temporal_state="UNKNOWN",
        phase10_aggregated_confidence={"helmet": 0.85, "no_helmet": 0.15},
        phase10_alerts_count=0,
        completeness=TrackletCompleteness.COMPLETE,
        appearance_descriptor=app,
        genuine_helmet_evidence={"track_helmet_state": "HELMET", "coupled_observations_count": 6},
    )


def test_d1_orphan_asymmetry_guard_recognizes_appearance_score_key() -> None:
    """D1: Asymmetric orphan<->rider (6 vs 50 frames) requires strong appearance (>= 0.85); weak (< 0.85) or missing appearance must block as UNMATCHED."""
    transit = TransitTimeModel(TransitTimeConfig(expected_transit_delay_sec=0.75, transit_sigma_sec=1.5))
    engine = OrphanAssociationEngine(
        transit_model=transit,
        config=OrphanAssociationConfig(min_orphan_association_score=0.60, min_strong_temporal_score_without_appearance=0.80),
    )
    app_strong_f = AppearanceDescriptorRecord(
        available=True,
        unavailable_reason=None,
        valid_observed_crops_count=4,
        mean_crop_quality=0.85,
        cranial_cap_histogram=[1.0, 0.0, 0.0, 0.0],
        upper_torso_histogram=[0.0, 1.0, 0.0, 0.0],
        moto_chassis_histogram=[],
    )
    front_orphan = _make_orphan_tracklet("front", "front_orphan_1", 1.0, 2.0, app_strong_f)
    back_rider = _make_rider_tracklet("back", "back_moto_1_rider_s0", "back_moto_1", 1.75, 2.75, app_strong_f, observed_frame_count=50)
    back_moto = _make_moto_tracklet("back", "back_moto_1", 1.75, 2.75, back_rider)
    unmatched_moto_res = MotorcycleAssociationResult(
        schema_version="1.0.0",
        global_motorcycle_id=None,
        front_moto_track_id=None,
        back_moto_track_id="back_moto_1",
        outcome=AssociationOutcome.UNMATCHED,
        association_score=0.0,
        second_best_score=0.0,
        ambiguity_margin=0.0,
        swap_margin_2x2=None,
        decision_reasons=[],
    )

    res_linked = engine.associate_orphans([front_orphan], [], [], [back_moto], [unmatched_moto_res])
    assert len(res_linked) == 1
    assert res_linked[0].outcome == AssociationOutcome.LINKED
    assert float(res_linked[0].cue_breakdown["appearance_score"]) >= 0.85
    assert not any("DURATION_ASYMMETRY" in r for r in res_linked[0].decision_reasons)

    # Weak uncalibrated appearance (< 0.85, e.g., ~0.65 opposing-view HSV overlap) must NOT bypass guard
    app_weak_b = AppearanceDescriptorRecord(
        available=True,
        unavailable_reason=None,
        valid_observed_crops_count=4,
        mean_crop_quality=0.85,
        cranial_cap_histogram=[0.4225, 0.5775, 0.0, 0.0],
        upper_torso_histogram=[0.5775, 0.4225, 0.0, 0.0],
        moto_chassis_histogram=[],
    )
    front_weak = _make_orphan_tracklet("front", "front_orphan_1", 1.0, 2.0, app_strong_f)
    back_rider_weak = _make_rider_tracklet("back", "back_moto_1_rider_s0", "back_moto_1", 1.75, 2.75, app_weak_b, observed_frame_count=50)
    back_moto_weak = _make_moto_tracklet("back", "back_moto_1", 1.75, 2.75, back_rider_weak)
    res_weak = engine.associate_orphans([front_weak], [], [], [back_moto_weak], [unmatched_moto_res])
    assert len(res_weak) == 1
    assert res_weak[0].outcome == AssociationOutcome.UNMATCHED
    assert "UNMATCHED_MOTO_RIDER_DURATION_ASYMMETRY_6vs50_WEAK_APPEARANCE" in res_weak[0].decision_reasons

    # Without appearance descriptor, asymmetric cross-kind match (6 vs 50 frames) must still be blocked
    app_missing = AppearanceDescriptorRecord(available=False, unavailable_reason="NO_CROPS_PROVIDED")
    front_no_app = _make_orphan_tracklet("front", "front_orphan_1", 1.0, 2.0, app_missing)
    back_rider_no_app = _make_rider_tracklet("back", "back_moto_1_rider_s0", "back_moto_1", 1.75, 2.75, app_missing, observed_frame_count=50)
    back_moto_no_app = _make_moto_tracklet("back", "back_moto_1", 1.75, 2.75, back_rider_no_app)
    res_blocked = engine.associate_orphans([front_no_app], [], [], [back_moto_no_app], [unmatched_moto_res])
    assert len(res_blocked) == 1
    assert res_blocked[0].outcome == AssociationOutcome.UNMATCHED
    assert "UNMATCHED_MOTO_RIDER_DURATION_ASYMMETRY_6vs50_WITHOUT_APPEARANCE" in res_blocked[0].decision_reasons


def test_d2_shared_detector_registry_keys_by_device_and_conf_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """D2: SharedDetectorRegistry isolates cached instances per (device, conf_threshold)."""
    created_veh: list[tuple[str | None, float]] = []
    created_helm: list[tuple[str | None, float]] = []

    class _DummyVeh:
        def __init__(self, device=None, confidence_threshold=0.18) -> None:
            self.device = device
            self.confidence_threshold = confidence_threshold
            created_veh.append((device, float(confidence_threshold)))

    class _DummyHelm:
        def __init__(self, device=None, confidence_threshold=0.25) -> None:
            self.device = device
            self.confidence_threshold = confidence_threshold
            created_helm.append((device, float(confidence_threshold)))

    old_veh = dict(cam_det.SharedDetectorRegistry._vehicle_detectors)
    old_helm = dict(cam_det.SharedDetectorRegistry._helmet_detectors)
    cam_det.SharedDetectorRegistry._vehicle_detectors.clear()
    cam_det.SharedDetectorRegistry._helmet_detectors.clear()
    try:
        monkeypatch.setattr(cam_det, "VehiclePersonDetectorAdapter", _DummyVeh)
        monkeypatch.setattr(cam_det, "HelmetDetectorAdapter", _DummyHelm)

        v1 = cam_det.SharedDetectorRegistry.get_vehicle_detector(device="cpu", conf_threshold=0.18)
        v1_again = cam_det.SharedDetectorRegistry.get_vehicle_detector(device="cpu", conf_threshold=0.18)
        v2 = cam_det.SharedDetectorRegistry.get_vehicle_detector(device="cpu", conf_threshold=0.35)
        assert v1 is v1_again
        assert v1 is not v2
        assert v2.confidence_threshold == pytest.approx(0.35)
        assert len(created_veh) == 2

        h1 = cam_det.SharedDetectorRegistry.get_helmet_detector(device="cpu", conf_threshold=0.25)
        h2 = cam_det.SharedDetectorRegistry.get_helmet_detector(device="cpu", conf_threshold=0.50)
        assert h1 is not h2
        assert h2.confidence_threshold == pytest.approx(0.50)
        assert len(created_helm) == 2
    finally:
        cam_det.SharedDetectorRegistry._vehicle_detectors.clear()
        cam_det.SharedDetectorRegistry._vehicle_detectors.update(old_veh)
        cam_det.SharedDetectorRegistry._helmet_detectors.clear()
        cam_det.SharedDetectorRegistry._helmet_detectors.update(old_helm)


def test_d3_temporal_engine_forwards_ema_alpha_to_tracks() -> None:
    """D3: TemporalVerificationEngine forwards TemporalEngineConfig.ema_alpha into created Track aggregators."""
    cfg = TemporalEngineConfig(aggregator_method="ema", ema_alpha=0.75)
    engine = TemporalVerificationEngine(config=cfg)
    assert engine.tracker.ema_alpha == pytest.approx(0.75)

    engine.process_frame(
        frame_index=0,
        raw_detections=[{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0, "class_name": "no_helmet"}],
    )
    engine.process_frame(
        frame_index=1,
        raw_detections=[
            {"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0, "class_name": "no_helmet"},
            {"bbox": [200.0, 200.0, 250.0, 250.0], "confidence": 0.85, "class_id": 1, "class_name": "helmet"},
        ],
    )
    assert len(engine.tracker.active_tracks) == 2
    for track in engine.tracker.active_tracks.values():
        assert track.aggregator.alpha == pytest.approx(0.75)


def test_d4_pipeline_fusion_serialization_cli_role_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D4: fusion_mode & fusion_config serialize cleanly, CLI exposes --fusion-mode, _track_records exports consensus_slot_role, and non-legacy run_pair writes fusion summary."""
    fc = FusionConfig(conflict_threshold=0.45, min_observations_for_decision=2)
    cfg = dcp.PipelineConfig(
        data_root=tmp_path / "data",
        metadata_path=tmp_path / "meta.json",
        tracking_config_path=REPO_ROOT / "configs" / "tracking_config.yaml",
        association_config_path=REPO_ROOT / "configs" / "cross_camera_association_config.yaml",
        output_root=tmp_path / "out",
        device="cpu",
        fusion_mode="dual_fused",
        fusion_config=fc,
    )
    roundtrip = dcp._config_from_dict(dcp._serialize_config(cfg))
    assert roundtrip.fusion_mode == "dual_fused"
    assert isinstance(roundtrip.fusion_config, FusionConfig)
    assert roundtrip.fusion_config.conflict_threshold == pytest.approx(0.45)
    assert roundtrip.fusion_config.min_observations_for_decision == 2

    # _track_records propagates distinct consensus_slot_role
    sub_stub = SimpleNamespace(
        subtrack_id="front_moto_0001_rider_s0",
        slot_role="pillion_1",
        consensus_slot_role="driver",
        track_helmet_state="NO_HELMET",
        alerts=[],
    )
    moto_stub = SimpleNamespace(
        moto_track_id="front_moto_0001",
        first_pts_sec=1.0,
        last_pts_sec=2.0,
        rider_subtracks={0: sub_stub},
    )
    cam_stub = SimpleNamespace(
        camera_id="front",
        canonical_shift_sec=0.0,
        pipeline=SimpleNamespace(terminated_moto_tracks=[moto_stub], active_moto_tracks={}),
    )
    recs = dcp._track_records(cam_stub)
    assert recs["front_moto_0001"]["rider_subtracks"][0]["consensus_slot_role"] == "driver"
    canon_idx, canon_role = dcp._canonical_slot_and_role(0, recs["front_moto_0001"]["rider_subtracks"][0])
    assert (canon_idx, canon_role) == (0, "driver")

    # CLI --fusion-mode argument wiring (intercept run_all to verify parsed config)
    captured_cfgs: list[dcp.PipelineConfig] = []

    def _fake_run_all(pair_ids: list[str], run_cfg: dcp.PipelineConfig) -> list[dcp.PairRunResult]:
        captured_cfgs.append(run_cfg)
        return [dcp.PairRunResult(pair_id=pid, status=dcp.PairStatus.COMPLETE, summary={}, error=None) for pid in pair_ids]

    monkeypatch.setattr(dcp, "run_all", _fake_run_all)
    rc = dcp.main(["--pairs", "pair_001", "--output-root", str(tmp_path / "cli_out"), "--fusion-mode", "dual_fused"])
    assert rc == 0
    assert captured_cfgs[0].fusion_mode == "dual_fused"
    rc_def = dcp.main(["--pairs", "pair_001", "--output-root", str(tmp_path / "cli_def")])
    assert rc_def == 0
    assert captured_cfgs[1].fusion_mode == "legacy"

    # Real synthetic run_pair check: dual_fused emits pair_fusion_summary.json, legacy does not
    FakePerCameraDetector.calls.clear()
    monkeypatch.setattr(dcp, "PerCameraDetector", FakePerCameraDetector)
    data_root = tmp_path / "syn_data"
    meta_path = _build_dataset(
        data_root,
        [{"pair_id": "pair_syn", "front": (45, 30.0), "back": (45, 30.0), "meta": {"offset": 0.0}}],
    )
    out_leg = tmp_path / "out_leg"
    res_leg = dcp.run_pair("pair_syn", _config(data_root, meta_path, out_leg, resume=False))
    assert res_leg.status == dcp.PairStatus.COMPLETE
    assert not (out_leg / "pair_syn" / "pair_syn_fusion_summary.json").exists()

    out_fused = tmp_path / "out_fused"
    cfg_fused = dcp.PipelineConfig(
        data_root=data_root,
        metadata_path=meta_path,
        tracking_config_path=REPO_ROOT / "configs" / "tracking_config.yaml",
        association_config_path=REPO_ROOT / "configs" / "cross_camera_association_config.yaml",
        output_root=out_fused,
        device="cpu",
        resume=False,
        fusion_mode="dual_fused",
        fusion_config=FusionConfig(),
    )
    res_fused = dcp.run_pair("pair_syn", cfg_fused)
    assert res_fused.status == dcp.PairStatus.COMPLETE
    fsum_path = out_fused / "pair_syn" / "pair_syn_fusion_summary.json"
    assert fsum_path.exists()
    fsum = json.loads(fsum_path.read_text(encoding="utf-8"))
    assert fsum["fusion_mode"] == "dual_fused"
    assert "stats" in fsum and "orphan_results" in fsum


def test_d5_compute_tiou_validation_boundary_and_task11_delegation() -> None:
    """D5: compute_tiou validates mode, handles boundary/tol edge cases, and shares implementation with Task 11."""
    assert stagek_eval.compute_tiou((2.0, 2.0), (2.0 + 5e-10, 2.0 + 5e-10), mode="fixed") == 1.0
    assert stagek_eval.compute_tiou((2.0, 2.0), (2.0 + 1e-6, 2.0 + 1e-6), mode="fixed") == 0.0
    assert stagek_eval.compute_tiou((0.0, 2.0), (2.0, 4.0), mode="fixed") == 0.0
    assert stagek_eval.compute_tiou((0.0, 2.0), (0.0, 4.0), mode="fixed") == pytest.approx(0.5)
    assert stagek_eval.compute_tiou((1.0, 3.0), (1.0, 3.0), mode="fixed") == 1.0
    with pytest.raises(ValueError, match="Unsupported tIoU mode"):
        stagek_eval.compute_tiou((1.0, 2.0), (1.0, 2.0), mode="unknown")

    for pair_a, pair_b in [
        ((1.428344, 1.428344), (1.428344, 1.428344)),
        ((0.0, 2.0), (1.0, 3.0)),
        ((None, None), (1.0, 2.0)),
    ]:
        assert task11_eval._tiou_fixed(pair_a, pair_b) == stagek_eval.compute_tiou(pair_a, pair_b, mode="fixed")
        assert task11_eval._tiou_legacy(pair_a, pair_b) == stagek_eval.compute_tiou(pair_a, pair_b, mode="legacy")


def test_d6_camera_tracker_fallback_defaults_match_yaml(tmp_path: Path) -> None:
    """D6: Empty-YAML fallback ByteTrack defaults match configs/tracking_config.yaml."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("{}\n", encoding="utf-8")
    pipe_empty = CameraTrackingPipeline.from_yaml("back", yaml_path=empty_yaml)
    pipe_real = CameraTrackingPipeline.from_yaml("back", yaml_path=REPO_ROOT / "configs" / "tracking_config.yaml")
    assert pipe_empty.bt_config.role_min_moto_speed_px == pipe_real.bt_config.role_min_moto_speed_px == pytest.approx(8.0)
    assert pipe_empty.bt_config.orphan_tentative_max_lost_frames == pipe_real.bt_config.orphan_tentative_max_lost_frames == 4
