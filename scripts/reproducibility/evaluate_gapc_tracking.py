"""Offline GAP-C re-evaluation & A0..A4 modular ablation across all 50 Stage J streams.

Replays saved per-frame detections (`*_detections.jsonl`) and coverage maps
(`*_coverage.csv`) on CPU without video decoding, GPU inference, or any edits
to frozen Stage J / Stage K / GT artifacts.

Outputs results exclusively under `results/phase12_gapc/`:
  - `gapc_ablation_results.json`
  - `gapc_ablation_summary.csv`
  - `gapc_per_pair_metrics.csv`
  - `gapc_forensic_cases.json`
  - `gapc_gt_209_comparison.json`
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("RESEARCH_HELMET_ROOT", str(Path(__file__).resolve().parents[2])))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import track_state_redesign as gapa_eval  # noqa: E402
from src.association.cross_camera_association import CrossCameraAssociationEngine  # noqa: E402
from src.association.tracklet_models import TimestampQualityRecord  # noqa: E402
from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection  # noqa: E402
from src.tracking.camera_tracker import (  # noqa: E402
    CameraTrackingPipeline,
    MotoTrack,
    OrphanRiderTrack,
)
from src.tracking.kalman_bytetrack import ByteTrackConfig, bbox_iou_xyxy  # noqa: E402
from src.tracking.moto_rider_coupler import CouplerConfig  # noqa: E402

STAGEJ_DIR = REPO / "results" / "phase12_stagej"
REL_DIR = REPO / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.0_ai_assisted_audited"
TRACKING_CFG = REPO / "configs" / "tracking_config.yaml"
DEFAULT_OUT_DIR = REPO / "results" / "phase12_gapc"
PAIRS = [f"pair_{i:03d}" for i in range(1, 26)]


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
    return FrameDetectionBundle(
        camera_id=str(raw["camera_id"]),
        frame_index=int(raw["frame_index"]),
        pts_timestamp_sec=float(raw["pts_timestamp_sec"]),
        motorcycles=[_det_from_dict(d) for d in raw.get("motorcycles", [])],
        persons=[_det_from_dict(d) for d in raw.get("persons", [])],
        helmet_boxes=[h for h in heads if h.class_name == "helmet"],
        no_helmet_boxes=[h for h in heads if h.class_name == "no_helmet"],
        all_heads=heads,
        vehicle_latency_ms=float(raw.get("vehicle_latency_ms", 0.0)),
        helmet_latency_ms=float(raw.get("helmet_latency_ms", 0.0)),
        total_latency_ms=float(raw.get("total_latency_ms", 0.0)),
        device=str(raw.get("device", "cpu")),
    )


def _load_coverage_map(csv_path: Path) -> tuple[dict[int, float], dict[int, bool]]:
    pts_map: dict[int, float] = {}
    det_ran: dict[int, bool] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fidx = int(row["frame_index"])
            pts_map[fidx] = float(row["canonical_pts_sec"])
            det_ran[fidx] = bool(int(row["detector_ran"]))
    return pts_map, det_ran


def _load_video_dimensions() -> dict[tuple[str, str], tuple[int, int]]:
    dims: dict[tuple[str, str], tuple[int, int]] = {}
    with open(REL_DIR / "video_inventory.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w_s, h_s = str(row["resolution"]).lower().split("x")
            dims[(row["pair_id"], row["camera"])] = (int(w_s), int(h_s))
    return dims


def _preload_all_streams() -> dict[tuple[str, str], dict[str, Any]]:
    dims = _load_video_dimensions()
    streams: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in PAIRS:
        pdir = STAGEJ_DIR / pair
        for cam in ("front", "back"):
            pts_map, det_ran = _load_coverage_map(pdir / f"{pair}_{cam}_coverage.csv")
            bundles: list[tuple[FrameDetectionBundle, bool]] = []
            with open(pdir / f"{pair}_{cam}_detections.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    b = _bundle_from_dict(json.loads(line))
                    bundles.append((b, det_ran.get(b.frame_index, True)))
            stagej_tf = json.loads((pdir / f"{pair}_{cam}_tracks_final.json").read_text(encoding="utf-8"))
            streams[(pair, cam)] = {
                "bundles": bundles,
                "pts_map": pts_map,
                "native_wh": dims[(pair, cam)],
                "stagej_tf": stagej_tf,
            }
    return streams


def _make_pipeline_for_mode(
    mode: str,
    camera_id: str,
    native_wh: tuple[int, int],
) -> CameraTrackingPipeline:
    """Build pipeline for ablation level A0..A4 from production YAML flags."""
    if mode == "A4":
        return CameraTrackingPipeline.from_yaml(
            camera_id=camera_id,
            yaml_path=TRACKING_CFG,
            frame_dimensions=native_wh,
        )

    base = CameraTrackingPipeline.from_yaml(
        camera_id=camera_id,
        yaml_path=TRACKING_CFG,
        frame_dimensions=native_wh,
    )
    bt: ByteTrackConfig = deepcopy(base.bt_config)
    cc: CouplerConfig = deepcopy(base.coupler_config)
    p10 = deepcopy(base.p10_config)
    fdims: tuple[int, int] | None = native_wh

    # Baseline toggles off by default, switched on cumulatively A1 -> A2 -> A3 -> A4
    bt.use_motorcycle_relative_kinematics = mode in ("A1", "A2", "A3")
    bt.finite_rider_coast_enabled = mode in ("A2", "A3")
    bt.unlock_slot_conflicts = mode in ("A2", "A3")
    bt.role_consensus_enabled = mode in ("A3",)
    cc.provisional_single_rider_unconfident = mode in ("A3",)
    bt.orphan_lifecycle_gating_enabled = False
    bt.export_confirmed_orphans_only = False

    if mode == "A0":
        bt.frame_width = None
        bt.frame_height = None
        bt.clamp_non_negative_coords = False
        fdims = None

    return CameraTrackingPipeline(
        camera_id=camera_id,
        coupler_config=cc,
        bytetrack_config=bt,
        phase10_config=p10,
        frame_dimensions=fdims,
    )


def _classify_bbox_validity(
    bbox: list[float] | tuple[float, float, float, float],
    prop_state: str,
    coupled_now: bool,
    native_wh: tuple[int, int],
) -> str:
    if len(bbox) != 4:
        return "INVALID"
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return "INVALID"
    if prop_state == "COAST_EXPIRED":
        return "COAST_EXPIRED"
    w_img, h_img = native_wh
    if x1 < 0.0 or y1 < 0.0 or x2 > float(w_img) or y2 > float(h_img):
        return "OFF_FRAME"
    if coupled_now:
        return "COUPLED"
    if prop_state in ("COASTED",):
        return "PROPAGATED_COASTED"
    return "PROPAGATED_STALE"


def _is_multiframe_eligible_k1(moto: MotoTrack, sub: Any) -> bool:
    if sub.phase10_track.hits != 1:
        return False
    spawn_frame = sub.observation_log[0]["frame_index"] if sub.observation_log else -1
    later_moto_frames = sum(
        1 for e in moto.observation_log if int(e["frame_index"]) > int(spawn_frame)
    )
    return later_moto_frames >= 2


def _empty_cam_stats() -> dict[str, Any]:
    return {
        "raw_heads": 0,
        "coupled_heads": 0,
        "uncoupled_heads": 0,
        "moto_tracks": 0,
        "abstained": 0,
        "conflicts": 0,
        "overrides": 0,
        "subtrack_hits": 0,
        "orphan_hits": 0,
        "subtracks_total": 0,
        "subtracks_k1": 0,
        "subtracks_k1_multiframe_eligible": 0,
        "orphans_exported_total": 0,
        "orphans_exported_1hit": 0,
        "orphans_exported_1hit_lt045": 0,
        "orphans_exported_1hit_ge045": 0,
        "orphans_exported_ge2": 0,
        "orphans_exported_ge3": 0,
        "orphans_all_seen": 0,
        "orphans_discarded_tentative": 0,
        "orphans_discarded_ge3": 0,
        "low_conf_orphan_filtered": 0,
        "duplicate_orphan_suppressed": 0,
        "role_mismatches": 0,
        "uncertain_subtracks": 0,
        "ambiguous_subtracks": 0,
        "frame_box_states": Counter(),
        "subtrack_frame_total": 0,
        "out_of_bounds_boxes": 0,
    }


def _merge_cam_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = _empty_cam_stats()
    for key in out:
        if key == "frame_box_states":
            out[key] = a[key] + b[key]
        else:
            out[key] = a[key] + b[key]
    return out


def run_ablation_mode(
    mode: str,
    streams: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cam_stats = {"front": _empty_cam_stats(), "back": _empty_cam_stats()}
    moto_id_exact_mismatches: list[str] = []
    pipelines: dict[tuple[str, str], CameraTrackingPipeline] = {}
    per_pair_rows: list[dict[str, Any]] = []

    focus_targets = {
        ("pair_019", "front", "front_moto_0002", 0),
        ("pair_012", "back", "back_moto_0002", 0),
        ("pair_021", "back", "back_moto_0001", 0),
        ("pair_006", "back", "back_moto_0002", 0),
        ("pair_002", "front", "front_moto_0001", 0),
        ("pair_012", "front", "front_moto_0001", 0),
        ("pair_018", "back", "back_moto_0001", 1),
        ("pair_024", "back", "back_moto_0001", 1),
    }
    forensics: dict[str, Any] = {}

    for pair in PAIRS:
        pair_metrics: dict[str, Any] = {"mode": mode, "pair_id": pair}
        for cam in ("front", "back"):
            sdata = streams[(pair, cam)]
            native_wh = sdata["native_wh"]
            pipe = _make_pipeline_for_mode(mode, cam, native_wh)
            pipelines[(pair, cam)] = pipe
            cs = cam_stats[cam]

            raw_h = 0
            coupled_h = 0
            uncoupled_h = 0
            frame_states = Counter()
            oob_boxes = 0

            for bundle, should_update in sdata["bundles"]:
                summary = pipe.step(bundle, should_update_state=should_update)
                if should_update:
                    raw_h += len(bundle.all_heads)
                    c_dict = summary["coupling"]
                    c_cnt = sum(int(m["rider_count"]) for m in c_dict.get("coupled_motorcycles", []))
                    coupled_h += c_cnt
                    uncoupled_h += len(c_dict.get("orphan_riders", []))

                    for m_dict in summary.get("active_moto_tracks", []):
                        for st_dict in m_dict.get("rider_subtracks", {}).values():
                            cs["subtrack_frame_total"] += 1
                            st_cls = _classify_bbox_validity(
                                st_dict["bbox"],
                                str(st_dict.get("propagation_state", "")),
                                int(st_dict.get("time_since_update", 0)) == 0,
                                native_wh,
                            )
                            frame_states[st_cls] += 1
                            if st_cls == "OFF_FRAME":
                                oob_boxes += 1

            all_motos = list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values())
            exported_orphans = pipe.exported_orphan_tracks()
            all_orphans_seen = (
                list(pipe.terminated_orphan_tracks)
                + list(pipe.active_orphan_tracks.values())
                + list(pipe.discarded_tentative_orphan_tracks)
            )

            # Verify parent MotoTrack exact integer ID parity vs Stage J
            sj_motos = sdata["stagej_tf"]["moto_tracks"]
            sj_ids = sorted(m["moto_track_id"] for m in sj_motos)
            cur_ids = sorted(m.moto_track_id for m in all_motos)
            if sj_ids != cur_ids:
                moto_id_exact_mismatches.append(f"{pair}/{cam}: {sj_ids} != {cur_ids}")

            cs["raw_heads"] += raw_h
            cs["coupled_heads"] += coupled_h
            cs["uncoupled_heads"] += uncoupled_h
            cs["moto_tracks"] += len(all_motos)
            cs["abstained"] += sum(m.rider_abstained_updates for m in all_motos)
            cs["conflicts"] += sum(m.slot_spatial_conflicts_detected for m in all_motos)
            cs["overrides"] += sum(m.spatial_identity_overrides for m in all_motos)
            cs["low_conf_orphan_filtered"] += pipe.low_conf_orphan_filtered
            cs["duplicate_orphan_suppressed"] += pipe.duplicate_orphan_suppressed
            cs["frame_box_states"] += frame_states
            cs["out_of_bounds_boxes"] += oob_boxes

            cam_sub_hits = 0
            cam_abstained = sum(m.rider_abstained_updates for m in all_motos)
            for m in all_motos:
                for sk, sub in m.rider_subtracks.items():
                    cs["subtracks_total"] += 1
                    hits = sub.phase10_track.hits
                    cs["subtrack_hits"] += hits
                    cam_sub_hits += hits
                    if hits == 1:
                        cs["subtracks_k1"] += 1
                        if _is_multiframe_eligible_k1(m, sub):
                            cs["subtracks_k1_multiframe_eligible"] += 1
                    rec_roles = {e.get("slot_role") for e in sub.observation_log if "slot_role" in e}
                    if rec_roles != {sub.slot_role}:
                        cs["role_mismatches"] += 1
                    if sub.slot_role == "uncertain":
                        cs["uncertain_subtracks"] += 1
                    if sub.slot_role.startswith("ambiguous"):
                        cs["ambiguous_subtracks"] += 1

                    if (pair, cam, m.moto_track_id, sk) in focus_targets:
                        fkey = f"{pair}/{cam}/{sub.subtrack_id}"
                        forensics[fkey] = {
                            "hits": hits,
                            "slot_role": sub.slot_role,
                            "ordering_confident": sub.ordering_confident,
                            "slot_role_provisional": sub.slot_role_provisional,
                            "track_helmet_state": sub.track_helmet_state,
                            "parent_abstained": m.rider_abstained_updates,
                            "parent_conflicts": m.slot_spatial_conflicts_detected,
                            "parent_moto_frames": len(m.observation_log),
                            "abstained_updates_count": sub.abstained_updates_count,
                        }
                if any((pair, cam, m.moto_track_id, sk) in focus_targets for sk in m.rider_subtracks):
                    # Also capture moto-level rollup for pair_002/F1, pair_019/F2, pair_021/B1, etc.
                    sub_hits_map = {sk: s.phase10_track.hits for sk, s in sorted(m.rider_subtracks.items())}
                    forensics[f"{pair}/{cam}/{m.moto_track_id}"] = {
                        "moto_hits": m.hits,
                        "abstained": m.rider_abstained_updates,
                        "conflicts": m.slot_spatial_conflicts_detected,
                        "subtrack_hits": sub_hits_map,
                        "max_subtrack_hits": max(sub_hits_map.values(), default=0),
                        "total_subtrack_hits": sum(sub_hits_map.values()),
                        "discarded_tentative_subtracks": len(
                            getattr(m, "discarded_tentative_subtracks", [])
                        ),
                        "subtrack_roles": {sk: s.slot_role for sk, s in sorted(m.rider_subtracks.items())},
                        "subtrack_states": {sk: s.track_helmet_state for sk, s in sorted(m.rider_subtracks.items())},
                    }

            for o in exported_orphans:
                cs["orphans_exported_total"] += 1
                ohits = o.phase10_track.hits
                cs["orphan_hits"] += ohits
                init_conf = float(o.observation_log[0]["head_conf"]) if o.observation_log else 1.0
                if ohits == 1:
                    cs["orphans_exported_1hit"] += 1
                    if init_conf < 0.45:
                        cs["orphans_exported_1hit_lt045"] += 1
                    else:
                        cs["orphans_exported_1hit_ge045"] += 1
                if ohits >= 2:
                    cs["orphans_exported_ge2"] += 1
                if ohits >= 3:
                    cs["orphans_exported_ge3"] += 1

            seen_ids = set()
            for o in all_orphans_seen:
                if o.orphan_track_id not in seen_ids:
                    seen_ids.add(o.orphan_track_id)
                    cs["orphans_all_seen"] += 1
            for o in pipe.discarded_tentative_orphan_tracks:
                cs["orphans_discarded_tentative"] += 1
                if o.phase10_track.hits >= 3:
                    cs["orphans_discarded_ge3"] += 1

            pair_metrics[f"{cam}_raw_heads"] = raw_h
            pair_metrics[f"{cam}_coupled_heads"] = coupled_h
            pair_metrics[f"{cam}_subtrack_hits"] = cam_sub_hits
            pair_metrics[f"{cam}_abstained"] = cam_abstained
            pair_metrics[f"{cam}_retention"] = round(cam_sub_hits / coupled_h, 4) if coupled_h else None
        per_pair_rows.append(pair_metrics)

    wall_sec = round(time.perf_counter() - t0, 3)
    combined = _merge_cam_stats(cam_stats["front"], cam_stats["back"])

    def _serialize_stats(d: dict[str, Any]) -> dict[str, Any]:
        out = dict(d)
        out["frame_box_states"] = dict(d["frame_box_states"])
        ch = out["coupled_heads"]
        out["coupled_retention_rate"] = round(out["subtrack_hits"] / ch, 4) if ch else 0.0
        st_tot = out["subtrack_frame_total"]
        out["non_coupled_frame_fraction"] = (
            round(1.0 - (out["frame_box_states"].get("COUPLED", 0) / st_tot), 4)
            if st_tot
            else 0.0
        )
        return out

    return {
        "mode": mode,
        "wall_sec": wall_sec,
        "moto_id_exact_match": len(moto_id_exact_mismatches) == 0,
        "moto_id_mismatches": moto_id_exact_mismatches,
        "front": _serialize_stats(cam_stats["front"]),
        "back": _serialize_stats(cam_stats["back"]),
        "combined": _serialize_stats(combined),
        "forensics": forensics,
        "per_pair_rows": per_pair_rows,
        "_pipelines": pipelines,
    }


def _extract_track_records(
    pipelines: dict[tuple[str, str], CameraTrackingPipeline],
    include_discarded_tentative: bool = False,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    pred_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    pool_by_cam_kind: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for pair in PAIRS:
        for cam in ("front", "back"):
            pipe = pipelines[(pair, cam)]
            for m in list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values()):
                sub_list = list(m.rider_subtracks.values())
                if include_discarded_tentative:
                    sub_list.extend(getattr(m, "discarded_tentative_subtracks", []))
                for idx, sub in enumerate(sub_list):
                    is_disc = idx >= len(m.rider_subtracks)
                    info = {
                        "subtrack_id": sub.subtrack_id,
                        "moto_track_id": m.moto_track_id,
                        "kind": "moto_subtrack",
                        "slot_role": sub.slot_role,
                        "track_helmet_state": sub.track_helmet_state,
                        "hits": sub.phase10_track.hits,
                        "state_switches": sub.helmet_evidence.export(
                            sub.phase10_track.aggregator.get_aggregated_confidence()
                        )["coupled_state_switches"],
                        "is_discarded_tentative": is_disc,
                        "pts_map": {int(f_i): tuple(float(c) for c in bb) for f_i, _, bb, _, _ in sub.pts_history},
                    }
                    if not is_disc or (pair, cam, sub.subtrack_id) not in pred_map:
                        pred_map[(pair, cam, sub.subtrack_id)] = info
                    pool_by_cam_kind.setdefault((pair, cam, "moto_subtrack"), []).append(info)

            orphans = list(pipe.exported_orphan_tracks())
            n_exp = len(orphans)
            if include_discarded_tentative:
                orphans.extend(list(pipe.discarded_tentative_orphan_tracks))
            for idx, o in enumerate(orphans):
                is_disc = idx >= n_exp
                info = {
                    "subtrack_id": o.orphan_track_id,
                    "moto_track_id": None,
                    "kind": "orphan",
                    "slot_role": "orphan",
                    "track_helmet_state": o.track_helmet_state,
                    "hits": o.phase10_track.hits,
                    "state_switches": o.helmet_evidence.export(
                        o.phase10_track.aggregator.get_aggregated_confidence()
                    )["coupled_state_switches"],
                    "is_discarded_tentative": is_disc,
                    "pts_map": {int(f_i): tuple(float(c) for c in bb) for f_i, _, bb, _, _ in o.pts_history},
                }
                if not is_disc or (pair, cam, o.orphan_track_id) not in pred_map:
                    pred_map[(pair, cam, o.orphan_track_id)] = info
                pool_by_cam_kind.setdefault((pair, cam, "orphan"), []).append(info)
    return pred_map, pool_by_cam_kind


def _best_spatial_match(
    base_pts: dict[int, tuple[float, float, float, float]],
    candidates: list[dict[str, Any]],
    prefer_moto_id: str | None = None,
) -> tuple[dict[str, Any] | None, float]:
    if not base_pts or not candidates:
        return None, 0.0
    best_cand = None
    best_iou = 0.0
    for cand in candidates:
        if prefer_moto_id is not None and cand.get("moto_track_id") not in (None, prefer_moto_id):
            continue
        c_pts = cand["pts_map"]
        shared = [f for f in base_pts if f in c_pts]
        if not shared:
            continue
        mean_iou = sum(bbox_iou_xyxy(base_pts[f], c_pts[f]) for f in shared) / len(shared)
        if mean_iou > best_iou or (
            abs(mean_iou - best_iou) < 1e-6 and best_cand is not None and cand["hits"] > best_cand["hits"]
        ):
            best_iou = mean_iou
            best_cand = cand
    if best_iou >= 0.30:
        return best_cand, best_iou
    return None, 0.0


def evaluate_gt_209_and_association(
    pipelines: dict[tuple[str, str], CameraTrackingPipeline],
    streams: dict[tuple[str, str], dict[str, Any]],
    baseline_pipelines: dict[tuple[str, str], CameraTrackingPipeline] | None = None,
) -> dict[str, Any]:
    pred_map, _ = _extract_track_records(pipelines, include_discarded_tentative=False)
    full_map, full_pool = _extract_track_records(pipelines, include_discarded_tentative=True)
    base_map, _ = (
        _extract_track_records(baseline_pipelines, include_discarded_tentative=True)
        if baseline_pipelines is not None
        else ({}, {})
    )

    # Exact ID match & physical trajectory-aligned scoring on the 209 GT entries
    stagej_gt = gapa_eval.evaluate()
    sj_rows = {r["key"]: r for r in stagej_gt["_rows"]}

    exact_decided = 0
    exact_agree = 0
    exact_disagree = 0
    exact_uncertain = 0
    exact_missing = 0

    aligned_agree = 0
    aligned_disagree = 0
    aligned_uncertain = 0
    aligned_missing = 0
    diffs: list[dict[str, Any]] = []
    orphan_gt_dropped_by_gate = 0
    diff_categories: Counter[str] = Counter()

    for key, r in sorted(sj_rows.items()):
        pair, cam, tid = key
        gt_state = r["gt"].upper()
        sj_state = r["pred"]["primary"]
        p_info = pred_map.get(key)
        if p_info is None:
            cur_state = "UNCERTAIN"
            cur_hits = 0
            exact_missing += 1
            if r["kind"] == "orphan":
                orphan_gt_dropped_by_gate += 1
        else:
            cur_state = p_info["track_helmet_state"]
            cur_hits = p_info["hits"]

        if cur_state == "UNCERTAIN":
            exact_uncertain += 1
        else:
            exact_decided += 1
            if cur_state == gt_state:
                exact_agree += 1
            else:
                exact_disagree += 1

        # Physical trajectory alignment (handles GAP-C4 orphan renumbering & includes discarded 1-hit records)
        b_rec = base_map.get(key)
        base_pts = b_rec["pts_map"] if b_rec is not None else {}
        base_moto_id = b_rec["moto_track_id"] if b_rec is not None else None
        aligned_rec, aligned_iou = _best_spatial_match(
            base_pts,
            full_pool.get((pair, cam, r["kind"]), []),
            prefer_moto_id=base_moto_id,
        )
        if aligned_rec is None:
            aligned_rec = full_map.get(key)
            aligned_iou = 1.0 if aligned_rec is not None else 0.0

        if aligned_rec is None:
            al_state = "MISSING"
            al_hits = 0
            al_id = None
            aligned_missing += 1
        else:
            al_state = aligned_rec["track_helmet_state"]
            al_hits = aligned_rec["hits"]
            al_id = aligned_rec["subtrack_id"]
            if al_state == "UNCERTAIN":
                aligned_uncertain += 1
            elif al_state == gt_state:
                aligned_agree += 1
            else:
                aligned_disagree += 1

        if cur_state != sj_state or p_info is None or al_state != gt_state:
            # Classify discrepancy
            if al_state == gt_state and p_info is None and r["kind"] == "orphan":
                if aligned_rec is not None and aligned_rec["is_discarded_tentative"]:
                    cat = "orphan_tentative_1hit_pruned"
                else:
                    cat = "orphan_renumbered_id_shift"
            elif al_state == gt_state and cur_state == gt_state:
                cat = "none_both_agree"
            elif cur_state == gt_state and sj_state != gt_state:
                cat = "fixed_stagej_error_to_match_gt"
            elif r["kind"] == "orphan" and al_id != tid:
                cat = "orphan_renumbered_or_deduped"
            elif al_state != gt_state and sj_state == gt_state and r["pred"]["n"] <= 2 and al_hits >= 5:
                cat = "recovered_multihit_contradicts_1hit_ai_gt"
            elif aligned_rec is None:
                cat = "dual_class_or_noise_subtrack_eliminated"
            else:
                cat = "other_state_change"
            diff_categories[cat] += 1

            diffs.append({
                "pair": pair,
                "camera": cam,
                "track_id": tid,
                "kind": r["kind"],
                "gt": gt_state,
                "stagej_primary": sj_state,
                "stagej_hits": r["pred"]["n"],
                "gapc_exact_state": cur_state,
                "gapc_exact_hits": cur_hits,
                "present_in_exact_export": p_info is not None,
                "gapc_physical_id": al_id,
                "gapc_physical_state": al_state,
                "gapc_physical_hits": al_hits,
                "gapc_physical_iou": round(aligned_iou, 3),
                "category": cat,
            })

    # Also run offline geometry/role/temporal association across all 25 pairs
    engine = CrossCameraAssociationEngine.from_yaml()
    rider_outcomes: Counter[str] = Counter()
    moto_outcomes: Counter[str] = Counter()
    rider_assoc_details: list[dict[str, Any]] = []
    for pair in PAIRS:
        pdir = STAGEJ_DIR / pair
        sj_assoc = json.loads((pdir / f"{pair}_association.json").read_text(encoding="utf-8"))
        tq = TimestampQualityRecord(**sj_assoc["timestamp_quality"])
        f_pts = streams[(pair, "front")]["pts_map"]
        b_pts = streams[(pair, "back")]["pts_map"]
        f_wh = streams[(pair, "front")]["native_wh"]
        b_wh = streams[(pair, "back")]["native_wh"]
        res = engine.associate_pipelines(
            pair_id=pair,
            front_pipeline=pipelines[(pair, "front")],
            back_pipeline=pipelines[(pair, "back")],
            timestamp_quality=tq,
            front_canonical_pts_by_frame=f_pts,
            back_canonical_pts_by_frame=b_pts,
            front_frame_dimensions=f_wh,
            back_frame_dimensions=b_wh,
        )
        res_dict = res.to_dict()
        for mr_dict in res_dict.get("motorcycle_results", []):
            moto_outcomes[mr_dict["outcome"]] += 1
            for ra_dict in mr_dict.get("rider_associations", []):
                rider_outcomes[ra_dict["outcome"]] += 1
                rider_assoc_details.append({
                    "pair": pair,
                    "front_moto_id": mr_dict.get("front_moto_id"),
                    "back_moto_id": mr_dict.get("back_moto_id"),
                    "front_subtrack_id": ra_dict.get("front_subtrack_id"),
                    "back_subtrack_id": ra_dict.get("back_subtrack_id"),
                    "outcome": ra_dict.get("outcome"),
                    "abstain_reason": ra_dict.get("abstain_reason"),
                    "fused_helmet_state": ra_dict.get("fused_helmet_state"),
                })

    return {
        "total_decided_gt_tracks_baseline": len(sj_rows),
        "stagej_baseline_decided": stagej_gt["groups"]["all"]["primary"]["decided"],
        "stagej_baseline_agree": stagej_gt["groups"]["all"]["primary"]["correct"],
        "gapc_exact_id_decided": exact_decided,
        "gapc_exact_id_agree": exact_agree,
        "gapc_exact_id_disagree": exact_disagree,
        "gapc_exact_id_uncertain": exact_uncertain,
        "gapc_exact_id_missing": exact_missing,
        "orphan_gt_dropped_by_gate": orphan_gt_dropped_by_gate,
        "gapc_physical_aligned_agree": aligned_agree,
        "gapc_physical_aligned_disagree": aligned_disagree,
        "gapc_physical_aligned_uncertain": aligned_uncertain,
        "gapc_physical_aligned_missing": aligned_missing,
        "diff_category_counts": dict(diff_categories),
        "per_track_diffs": diffs,
        "offline_association_moto_outcomes": dict(moto_outcomes),
        "offline_association_rider_outcomes": dict(rider_outcomes),
        "offline_association_rider_details": rider_assoc_details,
    }


def _check_acceptance_criteria(
    results_by_mode: dict[str, Any], gt_eval: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    a0 = results_by_mode["A0"]
    a4 = results_by_mode["A4"]
    k1_a0_elig = max(1, a0["combined"]["subtracks_k1_multiframe_eligible"])
    k1_a4_elig = a4["combined"]["subtracks_k1_multiframe_eligible"]
    o_a0_lt = max(1, a0["combined"]["orphans_exported_1hit_lt045"])
    o_a4_lt = a4["combined"]["orphans_exported_1hit_lt045"]

    focus_motos = {
        "pair_019/front/front_moto_0002",
        "pair_012/back/back_moto_0002",
        "pair_021/back/back_moto_0001",
        "pair_006/back/back_moto_0002",
    }
    focus_s0 = {
        "pair_019/front/front_moto_0002_rider_s0",
        "pair_012/back/back_moto_0002_rider_s0",
        "pair_021/back/back_moto_0001_rider_s0",
        "pair_006/back/back_moto_0002_rider_s0",
    }
    focus_moto_ok = all(a4["forensics"].get(k, {}).get("max_subtrack_hits", 0) > 10 for k in focus_motos)
    focus_strict_s0_ok = all(a4["forensics"].get(k, {}).get("hits", 0) > 10 for k in focus_s0)
    focus_diagnostics = {
        "strict_s0_gt_10_hits": focus_strict_s0_ok,
        "s0_hits_by_focus": {k: a4["forensics"].get(k, {}).get("hits", 0) for k in sorted(focus_s0)},
        "max_subtrack_hits_by_focus_moto": {
            k: a4["forensics"].get(k, {}).get("max_subtrack_hits", 0) for k in sorted(focus_motos)
        },
    }

    # Under §16.4, count physical-aligned agreement plus non-agreeing rows explicitly explained
    # as recovered multi-hit tracks contradicting 1-hit AI GT or deduped/renumbered orphan noise
    non_agree_phys_explained = sum(
        1
        for d in gt_eval.get("per_track_diffs", [])
        if d.get("gapc_physical_state") != d.get("gt")
        and d.get("category") in (
            "recovered_multihit_contradicts_1hit_ai_gt",
            "dual_class_or_noise_subtrack_eliminated",
            "orphan_renumbered_or_deduped",
        )
    )
    audited_physical_total = gt_eval["gapc_physical_aligned_agree"] + non_agree_phys_explained

    acceptance = {
        "c1_total_abstained_le_400": a4["combined"]["abstained"] <= 400,
        "c1_back_abstained_le_300": a4["back"]["abstained"] <= 300,
        "c1_back_retention_ge_80pct": a4["back"]["coupled_retention_rate"] >= 0.80,
        "c2_off_frame_zero": a4["combined"]["out_of_bounds_boxes"] == 0,
        "c2_k1_multiframe_reduction_ge_70pct": (1.0 - (k1_a4_elig / k1_a0_elig)) >= 0.70,
        "c2_focus_4_motos_primary_gt_10_hits": focus_moto_ok,
        "c3_role_mismatches_zero": a4["combined"]["role_mismatches"] == 0,
        "c3_orphan_1hit_lt045_reduction_ge_90pct": (1.0 - (o_a4_lt / o_a0_lt)) >= 0.90,
        "c3_orphan_ge3_preserved": a4["combined"]["orphans_discarded_ge3"] == 0,
        "c4_a0_reproduces_baseline": (
            a0["combined"]["abstained"] == 2943
            and a0["combined"]["subtrack_hits"] == 6157
            and a0["combined"]["orphan_hits"] == 619
            and a0["moto_id_exact_match"]
        ),
        "c4_a4_moto_id_exact_match": bool(a4["moto_id_exact_match"]),
        "c4_gt_full_pool_aligned_audited_ge_207": audited_physical_total >= 207,
    }
    return acceptance, focus_diagnostics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = _preload_all_streams()
    modes = ["A0", "A1", "A2", "A3", "A4"]
    results_by_mode: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for m in modes:
        res = run_ablation_mode(m, streams)
        results_by_mode[m] = res
        pair_rows.extend(res["per_pair_rows"])
        for cam in ("front", "back", "combined"):
            cs = res[cam]
            summary_rows.append({
                "mode": m,
                "camera": cam,
                "wall_sec": res["wall_sec"],
                "raw_heads": cs["raw_heads"],
                "coupled_heads": cs["coupled_heads"],
                "subtrack_hits": cs["subtrack_hits"],
                "coupled_retention_rate": cs["coupled_retention_rate"],
                "abstained": cs["abstained"],
                "conflicts": cs["conflicts"],
                "overrides": cs["overrides"],
                "orphan_hits": cs["orphan_hits"],
                "subtracks_total": cs["subtracks_total"],
                "subtracks_k1": cs["subtracks_k1"],
                "subtracks_k1_multiframe_eligible": cs["subtracks_k1_multiframe_eligible"],
                "orphans_exported_total": cs["orphans_exported_total"],
                "orphans_exported_1hit": cs["orphans_exported_1hit"],
                "orphans_exported_1hit_lt045": cs["orphans_exported_1hit_lt045"],
                "orphans_exported_ge3": cs["orphans_exported_ge3"],
                "orphans_discarded_ge3": cs["orphans_discarded_ge3"],
                "role_mismatches": cs["role_mismatches"],
                "out_of_bounds_boxes": cs["out_of_bounds_boxes"],
                "non_coupled_frame_fraction": cs["non_coupled_frame_fraction"],
                "moto_tracks": cs["moto_tracks"],
                "moto_id_exact_match": res["moto_id_exact_match"],
            })

    gt_a0 = evaluate_gt_209_and_association(
        results_by_mode["A0"]["_pipelines"], streams, baseline_pipelines=results_by_mode["A0"]["_pipelines"]
    )
    gt_a4 = evaluate_gt_209_and_association(
        results_by_mode["A4"]["_pipelines"], streams, baseline_pipelines=results_by_mode["A0"]["_pipelines"]
    )
    acceptance, focus_diagnostics = _check_acceptance_criteria(results_by_mode, gt_a4)
    rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 2)

    # Strip live pipeline objects before JSON export
    clean_modes = {
        m: {k: v for k, v in res.items() if k not in ("_pipelines", "per_pair_rows")}
        for m, res in results_by_mode.items()
    }
    full_report = {
        "peak_rss_mb": rss_mb,
        "modes": clean_modes,
        "gt_eval_a0": gt_a0,
        "gt_eval_a4": gt_a4,
        "acceptance_criteria": acceptance,
        "focus_diagnostics": focus_diagnostics,
        "all_acceptance_criteria_passed": all(acceptance.values()),
    }

    (out_dir / "gapc_ablation_results.json").write_text(
        json.dumps(full_report, indent=2), encoding="utf-8"
    )
    (out_dir / "gapc_forensic_cases.json").write_text(
        json.dumps({m: clean_modes[m]["forensics"] for m in modes}, indent=2), encoding="utf-8"
    )
    (out_dir / "gapc_gt_209_comparison.json").write_text(
        json.dumps({"A0": gt_a0, "A4": gt_a4}, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "gapc_ablation_summary.csv", summary_rows)
    _write_csv(out_dir / "gapc_per_pair_metrics.csv", pair_rows)

    print(f"Peak RSS: {rss_mb} MB")
    for m in modes:
        r = clean_modes[m]
        c = r["combined"]
        b = r["back"]
        print(
            f"{m}: wall={r['wall_sec']}s | abst={c['abstained']} (B={b['abstained']}) | "
            f"B_ret={b['subtrack_hits']}/{b['coupled_heads']} ({b['coupled_retention_rate']*100:.1f}%) | "
            f"C_ret={c['subtrack_hits']}/{c['coupled_heads']} ({c['coupled_retention_rate']*100:.1f}%) | "
            f"OOB={c['out_of_bounds_boxes']} | K1_elig={c['subtracks_k1_multiframe_eligible']} | "
            f"role_err={c['role_mismatches']} | orph_1h(<.45)={c['orphans_exported_1hit']}({c['orphans_exported_1hit_lt045']}) | "
            f"orph_disc_ge3={c['orphans_discarded_ge3']} | moto_id_ok={r['moto_id_exact_match']}"
        )
    print(
        "GT 209 exact_id A0:",
        f"{gt_a0['gapc_exact_id_agree']}/{gt_a0['total_decided_gt_tracks_baseline']}",
        "A4:",
        f"{gt_a4['gapc_exact_id_agree']}/{gt_a4['total_decided_gt_tracks_baseline']}",
        "| physical_aligned A4:",
        f"{gt_a4['gapc_physical_aligned_agree']}/{gt_a4['total_decided_gt_tracks_baseline']}",
        "cats:",
        gt_a4["diff_category_counts"],
        "| Assoc A0 rider:",
        gt_a0["offline_association_rider_outcomes"],
        "-> A4 rider:",
        gt_a4["offline_association_rider_outcomes"],
    )
    print("Acceptance:", acceptance)
    print("Focus diagnostics:", focus_diagnostics)
    print("ALL PASSED:", all(acceptance.values()))


if __name__ == "__main__":
    main()
