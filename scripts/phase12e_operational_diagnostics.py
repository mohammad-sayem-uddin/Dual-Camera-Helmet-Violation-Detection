"""Phase 12E Controlled Real-Video Operational Diagnostics & Runtime Benchmark.

Executes end-to-end synchronized dual-camera detection (Phase 12D `CameraDetector`),
per-camera hierarchical tracking (`CameraTrackingPipeline`), and cross-camera tracklet
association (`CrossCameraAssociationEngine`) on four representative real video pairs:
1. `pair_001`: Standard 60 fps Front + 60 fps Back baseline.
2. `pair_013`: Confirmed VFR pair (frame interval range 15.0 ms - 31.667 ms).
3. `pair_015`: Mixed frame-rate pair (60 fps Front + 30 fps Back, epsilon_sync_ms = 37.0 ms).
4. `pair_019`: High-density multi-vehicle traffic pair.

Outputs:
- `results/phase12e/phase12e_diagnostics_summary.json`
- `results/phase12e/tracklet_association_audit.csv`
- `results/phase12e/cue_score_distributions.csv`
- `results/phase12e/association_runtime_benchmark.csv`

Scientific Caveat:
This script measures operational behavior, cue availability, outcome distributions,
and runtime/memory overhead on real synchronized video pairs. It does NOT claim
ground-truth cross-camera identity accuracy (IDF1/HOTA), which requires human
cross-camera annotation.
"""
from __future__ import annotations

import gc
from pathlib import Path
import resource
import sys
import time
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.association import (
    AssociationAuditor,
    CrossCameraAssociationEngine,
    TimestampQualityRecord,
)
from src.detection.camera_detector import PerCameraDetector
from src.synchronization.synchronized_reader import SynchronizedVideoReader
from src.tracking.camera_tracker import CameraTrackingPipeline


TARGET_PAIRS = ["pair_001", "pair_013", "pair_015", "pair_019"]
ACTIVE_WINDOW_SEC = 1.60  # 96 canonical 60 Hz ticks covering the cross-camera transit window
MAX_CACHED_FRAMES_PER_CAM = 8  # Cap cached frames per camera to keep Peak RSS low


def _get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def run_diagnostics() -> dict[str, Any]:
    out_dir = PROJECT_ROOT / "results" / "phase12e"
    out_dir.mkdir(parents=True, exist_ok=True)

    front_detector = PerCameraDetector(camera_id="front")
    back_detector = PerCameraDetector(camera_id="back")
    assoc_engine = CrossCameraAssociationEngine.from_yaml(
        PROJECT_ROOT / "configs" / "cross_camera_association_config.yaml"
    )

    all_audit_rows: list[dict[str, Any]] = []
    all_cue_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}

    for pair_id in TARGET_PAIRS:
        t_pair_start = time.perf_counter()
        reader = SynchronizedVideoReader(pair_id=pair_id)
        meta = reader.sync_info

        front_pipe = CameraTrackingPipeline.from_yaml(
            "front", PROJECT_ROOT / "configs" / "tracking_config.yaml"
        )
        back_pipe = CameraTrackingPipeline.from_yaml(
            "back", PROJECT_ROOT / "configs" / "tracking_config.yaml"
        )

        # Store a bounded sample of frames with detections to keep Peak RSS low
        front_frames_cache: dict[int, np.ndarray] = {}
        back_frames_cache: dict[int, np.ndarray] = {}
        front_can_pts: dict[int, float] = {}
        back_can_pts: dict[int, float] = {}
        front_delta_ms: list[float] = []
        back_delta_ms: list[float] = []
        front_delta_by_f: dict[int, float] = {}
        back_delta_by_f: dict[int, float] = {}

        ticks_processed = 0
        front_updates = 0
        back_updates = 0
        back_repeated_skipped = 0

        # Evaluate the active cross-camera transit window inside the overlap interval
        window_start_sec = max(
            meta.overlap_start_back_sec,
            meta.overlap_end_back_sec - ACTIVE_WINDOW_SEC,
        )

        for frame_back, frame_front, status in reader.iter_synchronized_frames(
            step_frames=1, overlap_only=True, tick_hz=60.0
        ):
            if not status.get("valid_pair") or not status.get("is_in_overlap"):
                continue
            t_ref = float(status["t_ref_sec"])
            if t_ref + 1e-6 < window_start_sec:
                continue
            ticks_processed += 1

            f_idx = int(status["frame_idx_front"])
            b_idx = int(status["frame_idx_back"])

            front_delta_ms.append(float(status["delta_pts_front_ms"]))
            back_delta_ms.append(float(status["delta_pts_back_ms"]))

            if status["should_update_front_state"] and frame_front is not None and f_idx >= 0:
                front_updates += 1
                f_bundle = front_detector.detect_frame(
                    frame_bgr=frame_front,
                    frame_index=f_idx,
                    pts_timestamp_sec=float(status["t_front_actual_pts_sec"]),
                )
                front_pipe.step(f_bundle, should_update_state=True)
                front_can_pts[f_idx] = t_ref
                front_delta_by_f[f_idx] = float(status["delta_pts_front_ms"])
                if (
                    (f_bundle.motorcycles or f_bundle.all_heads)
                    and len(front_frames_cache) < MAX_CACHED_FRAMES_PER_CAM
                ):
                    front_frames_cache[f_idx] = frame_front

            if status["should_update_back_state"] and frame_back is not None and b_idx >= 0:
                back_updates += 1
                b_bundle = back_detector.detect_frame(
                    frame_bgr=frame_back,
                    frame_index=b_idx,
                    pts_timestamp_sec=float(status["t_back_actual_pts_sec"]),
                )
                back_pipe.step(b_bundle, should_update_state=True)
                back_can_pts[b_idx] = t_ref
                back_delta_by_f[b_idx] = float(status["delta_pts_back_ms"])
                if (
                    (b_bundle.motorcycles or b_bundle.all_heads)
                    and len(back_frames_cache) < MAX_CACHED_FRAMES_PER_CAM
                ):
                    back_frames_cache[b_idx] = frame_back
            else:
                back_repeated_skipped += 1

        reader.close()

        tq = TimestampQualityRecord(
            pair_id=pair_id,
            reference_clock_anchor="back_stream_pts",
            global_offset_sec=float(meta.global_offset_sec),
            epsilon_sync_ms=float(meta.epsilon_sync_ms),
            mean_delta_pts_front_ms=float(np.mean(front_delta_ms)) if front_delta_ms else 0.0,
            max_delta_pts_front_ms=float(np.max(front_delta_ms)) if front_delta_ms else 0.0,
            mean_delta_pts_back_ms=float(np.mean(back_delta_ms)) if back_delta_ms else 0.0,
            max_delta_pts_back_ms=float(np.max(back_delta_ms)) if back_delta_ms else 0.0,
            is_in_overlap=True,
        )

        batch_res = assoc_engine.associate_pipelines(
            pair_id=pair_id,
            front_pipeline=front_pipe,
            back_pipeline=back_pipe,
            timestamp_quality=tq,
            front_frames_by_index=front_frames_cache,
            back_frames_by_index=back_frames_cache,
            front_canonical_pts_by_frame=front_can_pts,
            back_canonical_pts_by_frame=back_can_pts,
            front_delta_pts_ms_by_frame=front_delta_by_f,
            back_delta_pts_ms_by_frame=back_delta_by_f,
            calibrated_homography_available=False,
        )

        total_pair_wall_sec = time.perf_counter() - t_pair_start
        peak_rss_mb = _get_peak_rss_mb()

        all_audit_rows.extend(AssociationAuditor.build_flat_audit_rows(batch_res))
        all_cue_rows.extend(AssociationAuditor.build_cue_distribution_rows(batch_res))

        rt = batch_res.runtime_ms
        runtime_rows.append(
            {
                "pair_id": pair_id,
                "canonical_ticks_evaluated": ticks_processed,
                "front_physical_updates": front_updates,
                "back_physical_updates": back_updates,
                "back_repeated_ticks_skipped": back_repeated_skipped,
                "front_moto_tracklets": len(batch_res.front_motorcycle_tracklets),
                "back_moto_tracklets": len(batch_res.back_motorcycle_tracklets),
                "front_orphan_tracklets": len(batch_res.front_orphan_tracklets),
                "back_orphan_tracklets": len(batch_res.back_orphan_tracklets),
                "candidate_pairs_evaluated": len(batch_res.candidate_pair_scores),
                "tracklet_build_ms": round(rt["tracklet_build_ms"], 4),
                "candidate_generation_ms": round(rt["candidate_generation_ms"], 4),
                "cue_scoring_ms": round(rt["cue_scoring_ms"], 4),
                "global_assignment_and_rider_ms": round(rt["global_assignment_and_rider_ms"], 4),
                "orphan_association_ms": round(rt["orphan_association_ms"], 4),
                "total_association_ms": round(rt["total_association_ms"], 4),
                "total_pair_wall_sec": round(total_pair_wall_sec, 3),
                "peak_rss_mb": round(peak_rss_mb, 2),
            }
        )

        pair_summaries[pair_id] = {
            "ticks_evaluated": ticks_processed,
            "front_physical_updates": front_updates,
            "back_physical_updates": back_updates,
            "back_repeated_ticks_skipped": back_repeated_skipped,
            "timestamp_quality": tq.to_dict(),
            "front_moto_tracklets_count": len(batch_res.front_motorcycle_tracklets),
            "back_moto_tracklets_count": len(batch_res.back_motorcycle_tracklets),
            "front_orphan_tracklets_count": len(batch_res.front_orphan_tracklets),
            "back_orphan_tracklets_count": len(batch_res.back_orphan_tracklets),
            "candidate_pairs_count": len(batch_res.candidate_pair_scores),
            "motorcycle_outcome_counts": batch_res.outcome_counts,
            "rider_outcome_counts": batch_res.rider_outcome_counts,
            "orphan_outcome_counts": batch_res.orphan_outcome_counts,
            "runtime_ms": {k: round(v, 4) for k, v in rt.items()},
            "peak_rss_mb": round(peak_rss_mb, 2),
            "motorcycle_results": [m.to_dict() for m in batch_res.motorcycle_results],
            "orphan_results": [o.to_dict() for o in batch_res.orphan_results],
        }

        front_frames_cache.clear()
        back_frames_cache.clear()
        gc.collect()

    summary_payload = {
        "schema_version": "1.0.0",
        "phase": "Phase 12E — Cross-Camera Tracklet Association Operational Diagnostics",
        "scientific_notice": (
            "Operational diagnostic results on real synchronized video pairs. "
            "Cross-camera identity accuracy (IDF1, HOTA, precision/recall) remains "
            "unmeasured until human cross-camera identity ground truth is annotated."
        ),
        "pairs_evaluated": TARGET_PAIRS,
        "active_window_sec": ACTIVE_WINDOW_SEC,
        "peak_rss_mb": round(_get_peak_rss_mb(), 2),
        "pair_summaries": pair_summaries,
    }

    AssociationAuditor.write_json(
        summary_payload, out_dir / "phase12e_diagnostics_summary.json"
    )
    AssociationAuditor.write_csv(
        all_audit_rows,
        out_dir / "tracklet_association_audit.csv",
        fieldnames=[
            "pair_id",
            "entity_level",
            "global_id",
            "front_track_id",
            "back_track_id",
            "outcome",
            "association_score",
            "second_best_score",
            "ambiguity_margin",
            "swap_margin_2x2",
            "active_cues_count",
            "available_cues",
            "unavailable_cues",
            "cue_temporal_transit_score",
            "cue_temporal_transit_eff_weight",
            "cue_rider_cardinality_score",
            "cue_rider_cardinality_eff_weight",
            "cue_appearance_hsv_score",
            "cue_appearance_hsv_eff_weight",
            "cue_lane_spatial_score",
            "cue_convoy_order_score",
            "epsilon_sync_ms",
            "delta_pts_front_ms",
            "delta_pts_back_ms",
            "delta_transit_sec",
            "transit_uncertainty_sec",
            "decision_reasons",
        ],
    )
    AssociationAuditor.write_csv(
        all_cue_rows,
        out_dir / "cue_score_distributions.csv",
        fieldnames=[
            "pair_id",
            "front_moto_track_id",
            "back_moto_track_id",
            "passed_candidate_gate",
            "composite_score",
            "cue_name",
            "available",
            "unavailable_reason",
            "quality",
            "normalized_score",
            "nominal_weight",
            "effective_weight",
            "weighted_contribution",
            "epsilon_sync_ms",
            "delta_pts_front_ms",
            "delta_pts_back_ms",
            "delta_transit_sec",
            "transit_uncertainty_sec",
        ],
    )
    AssociationAuditor.write_csv(
        runtime_rows,
        out_dir / "association_runtime_benchmark.csv",
    )

    print(
        f"[Phase 12E Diagnostics Complete] Evaluated {len(TARGET_PAIRS)} pairs, "
        f"Peak RSS = {_get_peak_rss_mb():.1f} MB"
    )
    return summary_payload


if __name__ == "__main__":
    run_diagnostics()
