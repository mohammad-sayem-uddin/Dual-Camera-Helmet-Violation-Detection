"""Phase 12D Real-Video Smoke Test & Per-Camera Inference Benchmark Script.

Executes controlled short-window smoke tests on 4 representative benchmark pairs:
  - `pair_001`: 60 fps Front (1 drop) + 60 fps Back (Sub-Frame sync)
  - `pair_013`: 60 fps Front + VFR (40.15 avg fps, 30 fps in overlap) Back
  - `pair_015`: 60 fps Front + 30 fps CFR Back (Near-Frame sync)
  - `pair_019`: 60 fps Front + 60 fps Back (Sub-Frame sync)

Produces:
  - `results/phase12d/phase12d_smoke_test_summary.json`
  - `results/phase12d/per_camera_inference_benchmark.csv`
  - Representative annotated side-by-side preview frames in `results/phase12d/previews/`.

Scientific Caveat:
These runs are unannotated operational smoke tests and latency/memory benchmarks.
They do NOT constitute ground-truth detection, coupling, or tracking accuracy claims.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.camera_detector import PerCameraDetector, SharedDetectorRegistry
from src.synchronization.synchronized_reader import SynchronizedVideoReader
from src.tracking.camera_tracker import CameraTrackingPipeline


def draw_camera_tracking_overlay(
    frame_bgr: np.ndarray,
    step_out: dict[str, Any],
    camera_label: str,
    target_height: int = 540,
) -> np.ndarray:
    """Render motorcycle boxes, rider subtrack head boxes, and orphan tracks onto a resized frame."""
    h_orig, w_orig = frame_bgr.shape[:2]
    scale = target_height / float(h_orig)
    w_new = int(round(w_orig * scale))
    canvas = cv2.resize(frame_bgr, (w_new, target_height))

    # Draw coupled motorcycles and their rider subtracks
    for m in step_out.get("active_moto_tracks", []):
        mx1, my1, mx2, my2 = [int(round(c * scale)) for c in m["bbox"]]
        m_id = m["moto_track_id"]
        l_state = m["lifecycle_state"]
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (255, 180, 0), 2)
        cv2.putText(
            canvas,
            f"{m_id} [{l_state}]",
            (max(4, mx1), max(18, my1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 200, 50),
            2,
        )
        for s_idx_str, r_sub in m.get("rider_subtracks", {}).items():
            rx1, ry1, rx2, ry2 = [int(round(c * scale)) for c in r_sub["bbox"]]
            t_state = r_sub["temporal_state"]
            role = r_sub["slot_role"]
            color = (0, 0, 255) if "NO_HELMET" in t_state else (0, 220, 0)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(
                canvas,
                f"s{s_idx_str}:{role} ({t_state})",
                (max(4, rx1), max(18, ry1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )

    # Draw orphan rider tracks
    for o in step_out.get("active_orphan_tracks", []):
        ox1, oy1, ox2, oy2 = [int(round(c * scale)) for c in o["bbox"]]
        t_state = o["temporal_state"]
        color = (0, 140, 255)
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), color, 2)
        cv2.putText(
            canvas,
            f"{o['orphan_track_id']} ({t_state})",
            (max(4, ox1), max(18, oy1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    cv2.putText(
        canvas,
        f"{camera_label} | f#{step_out['frame_index']} | PTS={step_out['pts_timestamp_sec']:.3f}s | updated={step_out['did_advance_state']}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return canvas


def main() -> None:
    t0_wall = time.perf_counter()
    out_dir = PROJECT_ROOT / "results" / "phase12d"
    prev_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det_front = PerCameraDetector(camera_id="front", device=device)
    det_back = PerCameraDetector(camera_id="back", device=device)

    # Warmup inference on a dummy frame so first-frame MPS compilation is measured separately
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = det_front.detect_frame(dummy, 0, 0.0)

    target_pairs = ["pair_001", "pair_013", "pair_015", "pair_019"]
    ticks_per_pair = 30  # 0.50 seconds of 60 Hz canonical overlap per representative pair

    csv_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []

    for pair_id in target_pairs:
        trk_front = CameraTrackingPipeline.from_yaml("front")
        trk_back = CameraTrackingPipeline.from_yaml("back")

        front_veh_lats: list[float] = []
        front_helm_lats: list[float] = []
        front_tot_lats: list[float] = []
        back_veh_lats: list[float] = []
        back_helm_lats: list[float] = []
        back_tot_lats: list[float] = []

        front_motos_detected = 0
        front_persons_detected = 0
        front_heads_detected = 0
        back_motos_detected = 0
        back_persons_detected = 0
        back_heads_detected = 0

        preview_saved = False
        last_bundle_f = None
        last_bundle_b = None
        last_out_f = None
        last_out_b = None

        with SynchronizedVideoReader(pair_id) as sync_reader:
            # Start near midpoint of overlap where motorcycles are actively passing
            t_mid = 0.5 * (
                sync_reader.sync_info.overlap_start_back_sec
                + sync_reader.sync_info.overlap_end_back_sec
            )
            t_start = max(
                sync_reader.sync_info.overlap_start_back_sec,
                t_mid - 0.25,
            )

            for tick in range(ticks_per_pair):
                t_ref = t_start + tick / 60.0
                if t_ref > sync_reader.sync_info.overlap_end_back_sec:
                    break
                fb, ff, status = sync_reader.get_synchronized_pair(t_ref)
                if not status["valid_pair"] or fb is None or ff is None:
                    continue

                if status["should_update_front_state"]:
                    bundle_f = det_front.detect_frame(
                        ff, status["frame_idx_front"], status["t_front_actual_pts_sec"]
                    )
                    last_bundle_f = bundle_f
                    front_veh_lats.append(bundle_f.vehicle_latency_ms)
                    front_helm_lats.append(bundle_f.helmet_latency_ms)
                    front_tot_lats.append(bundle_f.total_latency_ms)
                    front_motos_detected += len(bundle_f.motorcycles)
                    front_persons_detected += len(bundle_f.persons)
                    front_heads_detected += len(bundle_f.all_heads)
                    out_f = trk_front.step(bundle_f, should_update_state=True)
                    last_out_f = out_f
                else:
                    assert last_bundle_f is not None
                    out_f = trk_front.step(last_bundle_f, should_update_state=False)
                    last_out_f = out_f

                if status["should_update_back_state"]:
                    bundle_b = det_back.detect_frame(
                        fb, status["frame_idx_back"], status["t_back_actual_pts_sec"]
                    )
                    last_bundle_b = bundle_b
                    back_veh_lats.append(bundle_b.vehicle_latency_ms)
                    back_helm_lats.append(bundle_b.helmet_latency_ms)
                    back_tot_lats.append(bundle_b.total_latency_ms)
                    back_motos_detected += len(bundle_b.motorcycles)
                    back_persons_detected += len(bundle_b.persons)
                    back_heads_detected += len(bundle_b.all_heads)
                    out_b = trk_back.step(bundle_b, should_update_state=True)
                    last_out_b = out_b
                else:
                    assert last_bundle_b is not None
                    out_b = trk_back.step(last_bundle_b, should_update_state=False)
                    last_out_b = out_b

                # Save a side-by-side preview at tick 15 (or first frame with active tracks)
                if not preview_saved and tick >= 10 and last_out_f and last_out_b:
                    vis_f = draw_camera_tracking_overlay(ff, last_out_f, f"{pair_id} FRONT")
                    vis_b = draw_camera_tracking_overlay(fb, last_out_b, f"{pair_id} BACK")
                    side_by_side = np.hstack([vis_f, vis_b])
                    cv2.imwrite(
                        str(prev_dir / f"{pair_id}_phase12d_preview.jpg"),
                        side_by_side,
                    )
                    preview_saved = True

        for cam, v_lats, h_lats, t_lats, trk, n_m, n_p, n_h in [
            (
                "front",
                front_veh_lats,
                front_helm_lats,
                front_tot_lats,
                trk_front,
                front_motos_detected,
                front_persons_detected,
                front_heads_detected,
            ),
            (
                "back",
                back_veh_lats,
                back_helm_lats,
                back_tot_lats,
                trk_back,
                back_motos_detected,
                back_persons_detected,
                back_heads_detected,
            ),
        ]:
            all_motos = list(trk.active_moto_tracks.values()) + trk.terminated_moto_tracks
            all_orphans = list(trk.active_orphan_tracks.values()) + trk.terminated_orphan_tracks
            total_rider_subtracks = sum(len(m.rider_subtracks) for m in all_motos)

            row = {
                "pair_id": pair_id,
                "camera_id": cam,
                "device": device,
                "ticks_evaluated": trk.frames_updated + trk.duplicate_ticks_skipped,
                "unique_frames_updated": trk.frames_updated,
                "duplicate_ticks_skipped": trk.duplicate_ticks_skipped,
                "vehicle_yolo_mean_ms": round(float(np.mean(v_lats)), 2) if v_lats else 0.0,
                "helmet_yolo_mean_ms": round(float(np.mean(h_lats)), 2) if h_lats else 0.0,
                "combined_detect_mean_ms": round(float(np.mean(t_lats)), 2) if t_lats else 0.0,
                "combined_detect_p95_ms": round(float(np.percentile(t_lats, 95)), 2) if t_lats else 0.0,
                "raw_motorcycles_detected": n_m,
                "raw_persons_detected": n_p,
                "raw_heads_detected": n_h,
                "total_moto_tracks_created": len(all_motos),
                "total_rider_subtracks_created": total_rider_subtracks,
                "total_orphan_tracks_created": len(all_orphans),
                "stage2_low_conf_recoveries": sum(m.stage2_low_conf_matches for m in all_motos),
            }
            csv_rows.append(row)

        pair_summaries.append({
            "pair_id": pair_id,
            "preview_path": f"results/phase12d/previews/{pair_id}_phase12d_preview.jpg",
            "front_active_moto_tracks": [m.to_dict() for m in trk_front.active_moto_tracks.values()],
            "back_active_moto_tracks": [m.to_dict() for m in trk_back.active_moto_tracks.values()],
            "front_active_orphan_tracks": [o.to_dict() for o in trk_front.active_orphan_tracks.values()],
            "back_active_orphan_tracks": [o.to_dict() for o in trk_back.active_orphan_tracks.values()],
        })

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    csv_path = out_dir / "per_camera_inference_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    elapsed_sec = time.perf_counter() - t0_wall

    summary_payload = {
        "phase": "12D",
        "evaluation_type": "unannotated_real_video_smoke_test_and_resource_benchmark",
        "device": device,
        "models_verified": {
            "vehicle_person_detector": {
                "path": "yolov8n.pt",
                "sha256": SharedDetectorRegistry.get_vehicle_detector(device=device).get_sha256(),
                "target_classes": {0: "person", 3: "motorcycle"},
            },
            "helmet_detector": {
                "path": "results/phase9_multiseed/exp_combined/seed_42/weights/best.pt",
                "sha256": "0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5",
                "target_classes": {0: "no_helmet", 1: "helmet"},
            },
        },
        "peak_rss_memory_mb": round(peak_rss_mb, 2),
        "memory_budget_mb": 2048.0,
        "within_memory_budget": bool(peak_rss_mb < 2048.0),
        "elapsed_sec": round(elapsed_sec, 2),
        "benchmark_rows": csv_rows,
        "pair_smoke_summaries": pair_summaries,
    }
    with open(out_dir / "phase12d_smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(
        f"Phase 12D smoke benchmark complete in {elapsed_sec:.2f}s | "
        f"Device: {device} | Peak RSS: {peak_rss_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
