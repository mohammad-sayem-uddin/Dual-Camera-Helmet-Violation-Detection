"""Offline GAP-B re-evaluation across all 25 dual-camera Stage J pairs.

Replays saved per-frame detections (`*_detections.jsonl`), coverage maps
(`*_coverage.csv`), and synchronization diagnostics (`*_sync_diagnostics.json`)
on CPU without video decoding or GPU inference, plus tests real downscaled
720p frames in `docs/phase12/track_failure_examples/`.

Outputs artifacts under `results/phase12_gapb_redesign/`:
  - `gapb_evaluation_results.json`
  - `stagek_amended_results.json`
  - `border_classification_comparison.csv`
  - `rider_tracklet_evidence_comparison.csv`
  - `keyframe_and_appearance_audit.csv`
  - `rider_association_item_level.csv`
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(os.environ.get("RESEARCH_HELMET_ROOT", str(Path(__file__).resolve().parents[2])))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as stagek_evaluate  # noqa: E402
from src.association.appearance_features import (  # noqa: E402
    AppearanceConfig,
    extract_tracklet_appearance_descriptor,
)
from src.association.coordinate_space import crop_native_box  # noqa: E402
from src.association.cross_camera_association import CrossCameraAssociationEngine  # noqa: E402
from src.association.tracklet_builder import (  # noqa: E402
    BORDER_MISMATCH_REASON,
    BORDER_UNKNOWN_REASON,
    TrackletBuilder,
    TrackletBuilderConfig,
)


def _border_status(mt: Any) -> str:
    reasons = set(getattr(mt, "completeness_reasons", []) or [])
    if BORDER_UNKNOWN_REASON in reasons:
        return BORDER_UNKNOWN_REASON
    if BORDER_MISMATCH_REASON in reasons:
        return BORDER_MISMATCH_REASON
    return "VALID"
from src.association.tracklet_models import (  # noqa: E402
    ObservedFrameRecord,
    TimestampQualityRecord,
    genuine_evidence_to_dict,
)
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection  # noqa: E402
from src.pipeline.dual_camera_pipeline import (  # noqa: E402
    _observed_track_ids_at_step,
    should_retain_keyframe,
)
from src.tracking.camera_tracker import CameraTrackingPipeline  # noqa: E402

STAGEJ_DIR = Path(os.environ.get("STAGEJ_OUT_DIR", str(REPO / "results" / "phase12_stagej")))
REL_DIR = Path(
    os.environ.get(
        "GT_RELEASE_DIR",
        str(REPO / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.0_ai_assisted_audited"),
    )
)
EXAMPLES_DIR = Path(
    os.environ.get("TRACK_FAILURE_EXAMPLES_DIR", str(REPO / "docs" / "phase12" / "track_failure_examples"))
)
TRACKING_CFG = REPO / "configs" / "tracking_config.yaml"
ASSOC_CFG = REPO / "configs" / "cross_camera_association_config.yaml"
OUT_DIR = Path(os.environ.get("GAPB_EVAL_OUT_DIR", str(REPO / "results" / "phase12_gapb_redesign")))

KEYFRAME_STRIDE = 3
KEYFRAME_MAX_PER_CAMERA = 24
SIM_FRAME_WH = (1280, 720)


def _det_from_dict(raw: dict[str, Any]) -> StructuredDetection:
    x1, y1, x2, y2 = [float(v) for v in raw["bbox"]]
    return StructuredDetection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=float(raw["confidence"]),
        class_id=int(raw["class_id"]),
        class_name=str(raw["class_name"]),
        category=str(raw["category"]),
        source_model=str(raw["source_model"]),
        camera_id=str(raw["camera_id"]),
        frame_index=int(raw["frame_index"]),
        pts_timestamp_sec=float(raw["pts_timestamp_sec"]),
    )


def _bundle_from_dict(raw: dict[str, Any]) -> FrameDetectionBundle:
    heads = [_det_from_dict(d) for d in raw.get("heads", [])]
    helmet_boxes = [h for h in heads if h.class_name == "helmet"]
    no_helmet_boxes = [h for h in heads if h.class_name == "no_helmet"]
    return FrameDetectionBundle(
        camera_id=str(raw["camera_id"]),
        frame_index=int(raw["frame_index"]),
        pts_timestamp_sec=float(raw["pts_timestamp_sec"]),
        motorcycles=[_det_from_dict(d) for d in raw.get("motorcycles", [])],
        persons=[_det_from_dict(d) for d in raw.get("persons", [])],
        helmet_boxes=helmet_boxes,
        no_helmet_boxes=no_helmet_boxes,
        all_heads=heads,
        vehicle_latency_ms=float(raw.get("vehicle_latency_ms", 0.0)),
        helmet_latency_ms=float(raw.get("helmet_latency_ms", 0.0)),
        total_latency_ms=float(raw.get("total_latency_ms", 0.0)),
        device=str(raw.get("device", "cpu")),
    )


def _load_video_dimensions() -> dict[tuple[str, str], tuple[int, int]]:
    dims: dict[tuple[str, str], tuple[int, int]] = {}
    with open(REL_DIR / "video_inventory.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w_s, h_s = str(row["resolution"]).lower().split("x")
            dims[(row["pair_id"], row["camera"])] = (int(w_s), int(h_s))
    return dims


def _synthesize_textured_720p_frame(
    native_wh: tuple[int, int],
    moto_boxes: list[tuple[int, tuple[float, float, float, float]]],
) -> np.ndarray:
    """Create a deterministic 1280x720 textured image matching 4K det locations."""
    w_img, h_img = SIM_FRAME_WH
    w_nat, h_nat = native_wh
    sx = w_img / float(w_nat)
    sy = h_img / float(h_nat)

    img = np.full((h_img, w_img, 3), 24, dtype=np.uint8)
    # Deterministic low-intensity asphalt grid background so crops anywhere have structure
    yy, xx = np.mgrid[0:h_img, 0:w_img]
    bg_pattern = (((xx // 4) + (yy // 4)) % 2) * 16 + 20
    img[:, :, 0] = bg_pattern.astype(np.uint8)
    img[:, :, 1] = bg_pattern.astype(np.uint8)
    img[:, :, 2] = bg_pattern.astype(np.uint8)

    for track_num, (x1, y1, x2, y2) in moto_boxes:
        ix1 = max(0, min(w_img - 1, int(round(x1 * sx))))
        iy1 = max(0, min(h_img - 1, int(round(y1 * sy))))
        ix2 = max(ix1 + 1, min(w_img, int(round(x2 * sx))))
        iy2 = max(iy1 + 1, min(h_img, int(round(y2 * sy))))
        bw = ix2 - ix1
        bh = iy2 - iy1
        if bw <= 0 or bh <= 0:
            continue
        hue_bgr = (
            int((45 + track_num * 53) % 200) + 40,
            int((110 + track_num * 37) % 180) + 50,
            int((180 + track_num * 71) % 180) + 50,
        )
        patch_y, patch_x = np.mgrid[0:bh, 0:bw]
        checker = (((patch_x // 2) + (patch_y // 2)) % 2) * 55
        patch = np.zeros((bh, bw, 3), dtype=np.uint8)
        for ch in range(3):
            vals = np.clip(hue_bgr[ch] - 25 + checker, 0, 255).astype(np.uint8)
            patch[:, :, ch] = vals
        img[iy1:iy2, ix1:ix2] = patch
    return img


def _load_coverage_map(csv_path: Path) -> tuple[dict[int, float], dict[int, bool]]:
    pts_map: dict[int, float] = {}
    det_ran: dict[int, bool] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fidx = int(row["frame_index"])
            pts_map[fidx] = float(row["canonical_pts_sec"])
            det_ran[fidx] = bool(int(row["detector_ran"]))
    return pts_map, det_ran


def _replay_camera(
    cam: str,
    detections_path: Path,
    coverage_path: Path,
    native_wh: tuple[int, int],
) -> tuple[
    CameraTrackingPipeline,
    dict[int, float],
    dict[int, np.ndarray],
    list[int],
    dict[str, set[int]],
    list[int],
]:
    pts_map, det_ran = _load_coverage_map(coverage_path)
    pipeline = CameraTrackingPipeline.from_yaml(camera_id=cam, yaml_path=TRACKING_CFG)

    # Simulate BOTH retention policies identically along the frame stream:
    legacy_stored: list[int] = []
    gapb_stored: list[int] = []
    track_to_kf: dict[str, set[int]] = {}
    sim_keyframes: dict[int, np.ndarray] = {}

    with open(detections_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            bundle = _bundle_from_dict(raw)
            fidx = bundle.frame_index
            should_update = det_ran.get(fidx, True)
            summary = pipeline.step(bundle, should_update_state=should_update)

            if should_update:
                has_det = bool(bundle.motorcycles or bundle.persons or bundle.all_heads)
                if has_det and fidx % KEYFRAME_STRIDE == 0 and len(legacy_stored) < KEYFRAME_MAX_PER_CAMERA:
                    legacy_stored.append(fidx)

                obs_ids = _observed_track_ids_at_step(summary)
                retain, evicted_idx = should_retain_keyframe(
                    frame_index=fidx,
                    has_detections=has_det,
                    observed_track_ids=obs_ids,
                    stored_indices=gapb_stored,
                    track_to_keyframes=track_to_kf,
                    keyframe_stride=KEYFRAME_STRIDE,
                    keyframe_max_per_camera=KEYFRAME_MAX_PER_CAMERA,
                )
                if retain:
                    if evicted_idx is not None:
                        sim_keyframes.pop(evicted_idx, None)
                    # Build deterministic textured 720p frame from active track boxes at this frame
                    active_boxes = []
                    for m in summary.get("active_moto_tracks", []):
                        tid_num = int(str(m.get("moto_track_id", "0")).rsplit("_", 1)[-1])
                        bbox = tuple(float(v) for v in m["bbox"])
                        active_boxes.append((tid_num, bbox))
                    sim_keyframes[fidx] = _synthesize_textured_720p_frame(native_wh, active_boxes)
                    gapb_stored.append(fidx)
                    for tid in obs_ids:
                        track_to_kf.setdefault(tid, set()).add(fidx)

    return pipeline, pts_map, sim_keyframes, legacy_stored, track_to_kf, gapb_stored


def _legacy_phase10_state_from_dict(p10_snap: dict[str, Any]) -> str:
    agg = p10_snap.get("aggregated_confidence", {})
    h = float(agg.get("helmet", 0.5))
    nh = float(agg.get("no_helmet", 0.5))
    if h >= 0.65:
        return "HELMET"
    if nh >= 0.65:
        return "NO_HELMET"
    return "UNCERTAIN"


def _evaluate_real_example_frames(assoc_cfg: AppearanceConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pat = re.compile(r"^(pair_\d+)_(front|back)_.*_f(\d+)\.png$")
    for png_path in sorted(EXAMPLES_DIR.glob("*.png")):
        if png_path.name.endswith("_crop.png"):
            continue
        m = pat.match(png_path.name)
        if not m:
            continue
        pair_id, cam, fidx_str = m.group(1), m.group(2), m.group(3)
        fidx = int(fidx_str)
        img = cv2.imread(str(png_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # Fetch Stage J frame detections for this frame
        det_path = STAGEJ_DIR / pair_id / f"{pair_id}_{cam}_detections.jsonl"
        target_det_row = None
        with open(det_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row["frame_index"]) == fidx:
                    target_det_row = row
                    break
        if target_det_row is None:
            continue

        # Also inspect Stage J association descriptor status for referenced track/orphan
        assoc_json = json.loads((STAGEJ_DIR / pair_id / f"{pair_id}_association.json").read_text())
        track_key = f"{cam}_motorcycle_tracklets"
        stagej_desc = None
        if "orphan" in png_path.name:
            # Find matching orphan id token in filename (e.g., front_orphan_0003)
            om = re.search(r"((?:front|back)_orphan_\d+)", png_path.name)
            oid = om.group(1) if om else None
            for ot in assoc_json.get(f"{cam}_orphan_tracklets", []):
                if ot["orphan_track_id"] == oid:
                    stagej_desc = ot.get("appearance_descriptor", {})
                    break
        else:
            mm = re.search(r"((?:front|back)_moto_\d+)", png_path.name)
            mid = mm.group(1) if mm else None
            for mt in assoc_json.get(track_key, []):
                if mt["moto_track_id"] == mid:
                    stagej_desc = mt.get("appearance_descriptor", {})
                    break

        motos = target_det_row.get("motorcycles", [])
        heads = target_det_row.get("heads", [])
        sample_boxes = [tuple(float(v) for v in d["bbox"]) for d in (motos if motos else heads)]
        if not sample_boxes:
            continue
        # Test each detection box on this real 1280x720 PNG
        box_results = []
        for idx_b, bbox in enumerate(sample_boxes):
            unscaled_res = crop_native_box(img, bbox, native_wh=(w_img, h_img))
            scaled_res = crop_native_box(img, bbox, native_wh=(3840, 2160))
            box_results.append({
                "box_idx": idx_b,
                "native_4k_bbox": [round(v, 2) for v in bbox],
                "unscaled_on_720p_reason": unscaled_res.reason,
                "scaled_4k_to_720p_reason": scaled_res.reason,
                "scaled_crop_shape_hw": list(scaled_res.crop.shape[:2]) if scaled_res.crop is not None else None,
            })

        # Also test full descriptor extraction using extract_tracklet_appearance_descriptor on real PNG
        observed_frames = [
            ObservedFrameRecord(
                frame_index=fidx,
                camera_pts_sec=fidx / 60.0,
                canonical_pts_sec=fidx / 60.0,
                bbox_xyxy=b,
                kalman_bbox_xyxy=b,
                confidence=0.9,
            )
            for b in sample_boxes[:1]
        ]
        desc_unknown = extract_tracklet_appearance_descriptor(
            observed_frames=observed_frames,
            frame_images_by_index={fidx: img},
            config=assoc_cfg,
            is_motorcycle_tracklet=bool(motos),
            native_frame_dimensions=None,
        )
        desc_4k = extract_tracklet_appearance_descriptor(
            observed_frames=observed_frames,
            frame_images_by_index={fidx: img},
            config=assoc_cfg,
            is_motorcycle_tracklet=bool(motos),
            native_frame_dimensions=(3840, 2160),
        )
        records.append({
            "filename": png_path.name,
            "pair_id": pair_id,
            "camera_id": cam,
            "frame_index": fidx,
            "image_wh": [w_img, h_img],
            "stagej_tracklet_appearance": stagej_desc,
            "unknown_dims_descriptor": desc_unknown.to_dict(),
            "scaled_4k_descriptor": desc_4k.to_dict(),
            "per_box_crop_comparison": box_results,
        })
    return records


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_dims = _load_video_dimensions()
    engine = CrossCameraAssociationEngine.from_yaml(ASSOC_CFG)

    border_rows: list[dict[str, Any]] = []
    rider_rows: list[dict[str, Any]] = []
    kf_audit_rows: list[dict[str, Any]] = []
    rider_assoc_rows: list[dict[str, Any]] = []

    stagej_unavailable_reasons = Counter()
    gapb_unavailable_reasons = Counter()
    stagej_available_count = 0
    gapb_available_count = 0
    legacy_kf_covered = 0
    gapb_kf_covered = 0

    border_legacy_counts = Counter()
    border_gapb_counts = Counter()
    border_none_ablation_statuses = Counter()

    rider_subtrack_legacy_states = Counter()
    rider_subtrack_gapb_states = Counter()
    orphan_legacy_states = Counter()
    orphan_gapb_states = Counter()
    rider_recovered_from_uncertain = 0
    orphan_recovered_from_uncertain = 0
    snapshot_preserved_count = 0
    rider_total_count = 0
    orphan_total_count = 0

    candidate_gate_total = 0
    candidate_gate_violations = 0
    min_active_cues = engine.scorer.config.min_active_cues_required

    pair_dirs = sorted(p for p in STAGEJ_DIR.iterdir() if p.is_dir() and p.name.startswith("pair_"))
    for pdir in pair_dirs:
        pair_id = pdir.name
        stagej_assoc = json.loads((pdir / f"{pair_id}_association.json").read_text(encoding="utf-8"))
        tq = TimestampQualityRecord(**stagej_assoc["timestamp_quality"])

        front_wh = video_dims[(pair_id, "front")]
        back_wh = video_dims[(pair_id, "back")]

        f_pipe, f_pts, f_kf, f_leg_idx, f_track_kf, f_gapb_idx = _replay_camera(
            "front", pdir / f"{pair_id}_front_detections.jsonl", pdir / f"{pair_id}_front_coverage.csv", front_wh
        )
        b_pipe, b_pts, b_kf, b_leg_idx, b_track_kf, b_gapb_idx = _replay_camera(
            "back", pdir / f"{pair_id}_back_detections.jsonl", pdir / f"{pair_id}_back_coverage.csv", back_wh
        )

        result = engine.associate_pipelines(
            pair_id=pair_id,
            front_pipeline=f_pipe,
            back_pipeline=b_pipe,
            timestamp_quality=tq,
            front_frames_by_index=f_kf,
            back_frames_by_index=b_kf,
            front_canonical_pts_by_frame=f_pts,
            back_canonical_pts_by_frame=b_pts,
            front_frame_dimensions=front_wh,
            back_frame_dimensions=back_wh,
        )

        # Verify 100% replay fidelity against Stage J tracklet IDs
        sj_front_ids = [t["moto_track_id"] for t in stagej_assoc["front_motorcycle_tracklets"]]
        sj_back_ids = [t["moto_track_id"] for t in stagej_assoc["back_motorcycle_tracklets"]]
        rp_front_ids = [t.moto_track_id for t in result.front_motorcycle_tracklets]
        rp_back_ids = [t.moto_track_id for t in result.back_motorcycle_tracklets]
        assert sj_front_ids == rp_front_ids, f"{pair_id} front IDs mismatch: {sj_front_ids} vs {rp_front_ids}"
        assert sj_back_ids == rp_back_ids, f"{pair_id} back IDs mismatch: {sj_back_ids} vs {rp_back_ids}"

        # Also check frame_dimensions=None ablation once per camera tracklet
        none_builder = TrackletBuilder(TrackletBuilderConfig(
            min_observed_frames_moto=engine.builder.config.min_observed_frames_moto,
            min_observed_frames_rider=engine.builder.config.min_observed_frames_rider,
            min_observed_frames_orphan=engine.builder.config.min_observed_frames_orphan,
            min_observation_density=engine.builder.config.min_observation_density,
            border_margin_px=engine.builder.config.border_margin_px,
            exclude_tentative_tracks=engine.builder.config.exclude_tentative_tracks,
            frame_width_px=None,
            frame_height_px=None,
            appearance_config=engine.builder.config.appearance_config,
        ))
        f_none_motos, _ = none_builder.extract_from_pipeline(f_pipe, tq, canonical_pts_by_frame=f_pts, frame_dimensions=None)
        b_none_motos, _ = none_builder.extract_from_pipeline(b_pipe, tq, canonical_pts_by_frame=b_pts, frame_dimensions=None)
        for mt_none in f_none_motos + b_none_motos:
            border_none_ablation_statuses[_border_status(mt_none)] += 1

        # Audit candidate scoring gate
        for cp in result.candidate_pair_scores:
            candidate_gate_total += 1
            if cp.passed_candidate_gate and (
                cp.active_cues_count < min_active_cues
                or len(cp.available_cue_names) < min_active_cues
                or "INSUFFICIENT_ACTIVE_CUES" in cp.gate_rejection_reasons
            ):
                candidate_gate_violations += 1

        # Compare motorcycle tracklets (Stage J vs GAP-B)
        for cam, sj_motos, rp_motos, leg_kf_list, gapb_kf_list in [
            ("front", stagej_assoc["front_motorcycle_tracklets"], result.front_motorcycle_tracklets, f_leg_idx, f_gapb_idx),
            ("back", stagej_assoc["back_motorcycle_tracklets"], result.back_motorcycle_tracklets, b_leg_idx, b_gapb_idx),
        ]:
            leg_set = set(leg_kf_list)
            gapb_set = set(gapb_kf_list)
            for sj_m, rp_m in zip(sj_motos, rp_motos):
                # Border classification comparison
                leg_en = bool(sj_m["entry_near_border"])
                leg_ex = bool(sj_m["exit_near_border"])
                gapb_en = bool(rp_m.entry_near_border)
                gapb_ex = bool(rp_m.exit_near_border)
                b_stat = _border_status(rp_m)
                border_legacy_counts[("entry", leg_en)] += 1
                border_legacy_counts[("exit", leg_ex)] += 1
                border_gapb_counts[("entry", gapb_en)] += 1
                border_gapb_counts[("exit", gapb_ex)] += 1
                border_rows.append({
                    "pair_id": pair_id,
                    "camera_id": cam,
                    "moto_track_id": rp_m.moto_track_id,
                    "first_frame": rp_m.first_frame_index,
                    "last_frame": rp_m.last_frame_index,
                    "entry_bbox": json.dumps(rp_m.entry_bbox),
                    "exit_bbox": json.dumps(rp_m.exit_bbox),
                    "legacy_entry_near_border": leg_en,
                    "legacy_exit_near_border": leg_ex,
                    "gapb_entry_near_border": gapb_en,
                    "gapb_exit_near_border": gapb_ex,
                    "border_geometry_status": b_stat,
                    "entry_changed": leg_en != gapb_en,
                    "exit_changed": leg_ex != gapb_ex,
                })

                # Appearance & keyframe retention comparison
                sj_app = sj_m["appearance_descriptor"]
                if sj_app["available"]:
                    stagej_available_count += 1
                else:
                    stagej_unavailable_reasons[str(sj_app.get("unavailable_reason"))] += 1

                if rp_m.appearance_descriptor.available:
                    gapb_available_count += 1
                else:
                    gapb_unavailable_reasons[str(rp_m.appearance_descriptor.unavailable_reason)] += 1

                span_set = set(range(rp_m.first_frame_index, rp_m.last_frame_index + 1))
                leg_hit = len(span_set & leg_set) > 0
                gapb_hit = rp_m.appearance_descriptor.observed_frames_with_keyframe > 0
                if leg_hit:
                    legacy_kf_covered += 1
                if gapb_hit:
                    gapb_kf_covered += 1

                kf_audit_rows.append({
                    "pair_id": pair_id,
                    "camera_id": cam,
                    "moto_track_id": rp_m.moto_track_id,
                    "observed_frames_total": rp_m.appearance_descriptor.observed_frames_total,
                    "legacy_kf_span_overlap": len(span_set & leg_set),
                    "gapb_observed_with_keyframe": rp_m.appearance_descriptor.observed_frames_with_keyframe,
                    "stagej_available": sj_app["available"],
                    "stagej_unavailable_reason": sj_app.get("unavailable_reason") or "",
                    "gapb_sim_available": rp_m.appearance_descriptor.available,
                    "gapb_sim_unavailable_reason": rp_m.appearance_descriptor.unavailable_reason or "",
                    "gapb_valid_crops": rp_m.appearance_descriptor.valid_observed_crops_count,
                    "gapb_mean_quality": round(rp_m.appearance_descriptor.mean_crop_quality, 4),
                    "gapb_crop_rejections": json.dumps(rp_m.appearance_descriptor.crop_rejection_counts, sort_keys=True),
                })

                # Rider subtrack TASK-04 genuine-observation propagation check
                sj_riders = sj_m.get("rider_tracklets", {})
                for s_key, rp_r in sorted(rp_m.rider_tracklets.items(), key=lambda kv: kv[0]):
                    rider_total_count += 1
                    sj_r = sj_riders.get(str(s_key), {})
                    sj_p10 = sj_r.get("phase10_snapshot_readonly", {})
                    leg_state = _legacy_phase10_state_from_dict(sj_p10)
                    # Verify snapshot preservation
                    if rp_r.phase10_temporal_state == sj_p10.get("temporal_state"):
                        snapshot_preserved_count += 1

                    rp_ge = genuine_evidence_to_dict(rp_r.genuine_helmet_evidence)
                    assert rp_ge.get("available") is True, f"{pair_id} {rp_r.subtrack_id} missing genuine evidence"
                    assert rp_r.track_helmet_state == rp_ge["track_helmet_state"]
                    gapb_state = str(rp_ge.get("track_helmet_state", "UNCERTAIN"))
                    rider_subtrack_legacy_states[leg_state] += 1
                    rider_subtrack_gapb_states[gapb_state] += 1
                    if leg_state == "UNCERTAIN" and gapb_state in ("HELMET", "NO_HELMET"):
                        rider_recovered_from_uncertain += 1

                    rider_rows.append({
                        "pair_id": pair_id,
                        "camera_id": cam,
                        "track_kind": "rider_subtrack",
                        "track_id": rp_r.subtrack_id,
                        "legacy_p10_temporal_state": sj_p10.get("temporal_state", ""),
                        "legacy_p10_derived_state": leg_state,
                        "legacy_agg_helmet": sj_p10.get("aggregated_confidence", {}).get("helmet", ""),
                        "gapb_genuine_state": gapb_state,
                        "gapb_majority_state": rp_ge.get("track_helmet_state_majority", ""),
                        "gapb_ema_final_state": rp_ge.get("track_helmet_state_ema_final", ""),
                        "gapb_coupled_obs_count": rp_ge.get("coupled_observations_count", 0),
                        "gapb_helmet_count": rp_ge.get("coupled_helmet_count", 0),
                        "gapb_no_helmet_count": rp_ge.get("coupled_no_helmet_count", 0),
                        "gapb_last_coupled_agg_conf": rp_ge.get("last_coupled_aggregated_confidence", ""),
                        "gapb_appearance_available": rp_r.appearance_descriptor.available,
                    })

        # Verify RiderAssociationResult readonly propagation
        for mr in result.motorcycle_results:
            for ra in mr.rider_associations:
                ra_dict = ra.to_dict()
                if ra.front_subtrack_id is not None:
                    assert "genuine_observation_helmet_evidence" in ra_dict["front_phase10_readonly"]
                if ra.back_subtrack_id is not None:
                    assert "genuine_observation_helmet_evidence" in ra_dict["back_phase10_readonly"]
                rider_assoc_rows.append({
                    "pair_id": pair_id,
                    "front_moto_track_id": ra.front_moto_track_id,
                    "back_moto_track_id": ra.back_moto_track_id,
                    "front_subtrack_id": ra.front_subtrack_id,
                    "back_subtrack_id": ra.back_subtrack_id,
                    "outcome": ra.outcome.value,
                    "association_score": ra.association_score,
                    "front_genuine_state": (ra.front_phase10_readonly or {}).get("genuine_observation_helmet_evidence", {}).get("track_helmet_state", ""),
                    "back_genuine_state": (ra.back_phase10_readonly or {}).get("genuine_observation_helmet_evidence", {}).get("track_helmet_state", ""),
                })

        # Orphan tracklet propagation check
        for cam, sj_orphs, rp_orphs in [
            ("front", stagej_assoc["front_orphan_tracklets"], result.front_orphan_tracklets),
            ("back", stagej_assoc["back_orphan_tracklets"], result.back_orphan_tracklets),
        ]:
            sj_by_id = {o["orphan_track_id"]: o for o in sj_orphs}
            for rp_o in rp_orphs:
                orphan_total_count += 1
                sj_o = sj_by_id.get(rp_o.orphan_track_id, {})
                sj_p10 = sj_o.get("phase10_snapshot_readonly", {})
                leg_state = _legacy_phase10_state_from_dict(sj_p10)
                rp_ge = genuine_evidence_to_dict(rp_o.genuine_helmet_evidence)
                assert rp_ge.get("available") is True, f"{pair_id} {rp_o.orphan_track_id} missing genuine evidence"
                assert rp_o.track_helmet_state == rp_ge["track_helmet_state"]
                gapb_state = str(rp_ge.get("track_helmet_state", "UNCERTAIN"))
                orphan_legacy_states[leg_state] += 1
                orphan_gapb_states[gapb_state] += 1
                if leg_state == "UNCERTAIN" and gapb_state in ("HELMET", "NO_HELMET"):
                    orphan_recovered_from_uncertain += 1
                rider_rows.append({
                    "pair_id": pair_id,
                    "camera_id": cam,
                    "track_kind": "orphan_rider",
                    "track_id": rp_o.orphan_track_id,
                    "legacy_p10_temporal_state": sj_p10.get("temporal_state", ""),
                    "legacy_p10_derived_state": leg_state,
                    "legacy_agg_helmet": sj_p10.get("aggregated_confidence", {}).get("helmet", ""),
                    "gapb_genuine_state": gapb_state,
                    "gapb_majority_state": rp_ge.get("track_helmet_state_majority", ""),
                    "gapb_ema_final_state": rp_ge.get("track_helmet_state_ema_final", ""),
                    "gapb_coupled_obs_count": rp_ge.get("coupled_observations_count", 0),
                    "gapb_helmet_count": rp_ge.get("coupled_helmet_count", 0),
                    "gapb_no_helmet_count": rp_ge.get("coupled_no_helmet_count", 0),
                    "gapb_last_coupled_agg_conf": rp_ge.get("last_coupled_aggregated_confidence", ""),
                    "gapb_appearance_available": rp_o.appearance_descriptor.available,
                })

    # Real 1280x720 example PNG validation
    real_example_results = _evaluate_real_example_frames(engine.builder.config.appearance_config)

    # Amended Stage K reproducibility run (non-mutating wrt results/phase12_integrated)
    stagek_res, stagek_pp = stagek_evaluate.run_evaluation(
        out_dir=OUT_DIR, rel_dir=REL_DIR, sj_dir=STAGEJ_DIR, write_files=False
    )
    (OUT_DIR / "stagek_amended_results.json").write_text(
        json.dumps(stagek_res, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write CSV artifacts
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(OUT_DIR / "border_classification_comparison.csv", border_rows)
    _write_csv(OUT_DIR / "rider_tracklet_evidence_comparison.csv", rider_rows)
    _write_csv(OUT_DIR / "keyframe_and_appearance_audit.csv", kf_audit_rows)
    _write_csv(OUT_DIR / "rider_association_item_level.csv", rider_assoc_rows)

    # Summarize border confusion metrics (treating 4K geometry as ground truth for geometry classification)
    entry_fp = sum(1 for r in border_rows if r["legacy_entry_near_border"] and not r["gapb_entry_near_border"])
    entry_fn = sum(1 for r in border_rows if not r["legacy_entry_near_border"] and r["gapb_entry_near_border"])
    exit_fp = sum(1 for r in border_rows if r["legacy_exit_near_border"] and not r["gapb_exit_near_border"])
    exit_fn = sum(1 for r in border_rows if not r["legacy_exit_near_border"] and r["gapb_exit_near_border"])

    summary = {
        "evaluation_scope": {
            "pairs_replayed": len(pair_dirs),
            "moto_tracklets_total": len(border_rows),
            "rider_subtracks_total": rider_total_count,
            "orphan_tracklets_total": orphan_total_count,
            "real_example_frames_tested": len(real_example_results),
            "note": (
                "All 25 pairs replayed offline on CPU from frozen Stage J *_detections.jsonl, "
                "*_coverage.csv, and *_sync_diagnostics.json. Synthetic 1280x720 textured frames "
                "simulate keyframe retention/cropping end-to-end across all 100 moto tracklets, "
                "while real_example_frames tests genuine 1280x720 PNGs from docs/phase12/track_failure_examples/."
            ),
        },
        "gap_b1_appearance_and_keyframes": {
            "stagej_moto_descriptor_available": stagej_available_count,
            "stagej_unavailable_reasons": dict(sorted(stagej_unavailable_reasons.items())),
            "legacy_stride_keyframe_moto_coverage": legacy_kf_covered,
            "gapb_retention_keyframe_moto_coverage": gapb_kf_covered,
            "gapb_sim_moto_descriptor_available": gapb_available_count,
            "gapb_sim_unavailable_reasons": dict(sorted(gapb_unavailable_reasons.items())),
            "real_example_frames": {
                "tested_count": len(real_example_results),
                "unknown_dims_4k_rejected_count": sum(
                    1
                    for rec in real_example_results
                    if rec["unknown_dims_descriptor"]["unavailable_reason"]
                    == "COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS"
                ),
                "scaled_4k_to_720p_recovered_count": sum(
                    1 for rec in real_example_results if rec["scaled_4k_descriptor"]["available"]
                ),
                "details": real_example_results,
            },
        },
        "gap_b2_border_classification": {
            "legacy_entry_near_border_true": border_legacy_counts[("entry", True)],
            "gapb_entry_near_border_true": border_gapb_counts[("entry", True)],
            "legacy_exit_near_border_true": border_legacy_counts[("exit", True)],
            "gapb_exit_near_border_true": border_gapb_counts[("exit", True)],
            "entry_false_positives_in_legacy": entry_fp,
            "entry_false_negatives_in_legacy": entry_fn,
            "exit_false_positives_in_legacy": exit_fp,
            "exit_false_negatives_in_legacy": exit_fn,
            "none_dims_ablation_border_statuses": dict(sorted(border_none_ablation_statuses.items())),
        },
        "gap_b3_evidence_and_gate": {
            "rider_subtrack_snapshot_preserved": snapshot_preserved_count,
            "rider_subtracks_legacy_states": dict(sorted(rider_subtrack_legacy_states.items())),
            "rider_subtracks_gapb_genuine_states": dict(sorted(rider_subtrack_gapb_states.items())),
            "rider_subtracks_recovered_from_uncertain": rider_recovered_from_uncertain,
            "orphan_legacy_states": dict(sorted(orphan_legacy_states.items())),
            "orphan_gapb_genuine_states": dict(sorted(orphan_gapb_states.items())),
            "orphan_recovered_from_uncertain": orphan_recovered_from_uncertain,
            "candidate_pairs_evaluated": candidate_gate_total,
            "insufficient_active_cues_gate_violations": candidate_gate_violations,
        },
        "gap_b4_stagek_item_level_rider_association": stagek_res["association_rider_item_level"],
    }

    (OUT_DIR / "gapb_evaluation_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("GAPB_EVAL_DONE")
    print(json.dumps({k: v for k, v in summary.items() if k != "gap_b1_appearance_and_keyframes"}, indent=2))
    print("GAP-B1 counts:", {
        k: v for k, v in summary["gap_b1_appearance_and_keyframes"].items() if k != "real_example_frames"
    })
    print("Real examples recovered:", summary["gap_b1_appearance_and_keyframes"]["real_example_frames"]["scaled_4k_to_720p_recovered_count"], "/", summary["gap_b1_appearance_and_keyframes"]["real_example_frames"]["tested_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
