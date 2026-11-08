"""Phase 12C Forensic Timestamp & PTS Alignment Audit Script.

Audits all 50 video containers (25 front + 25 back) in data/dual_camera_test_v1/:
1. Validates ffprobe rational PTS tables against sequential OpenCV CAP_PROP_POS_MSEC.
2. Quantifies legacy nominal-FPS (frame_idx / avg_fps) and OpenCV CAP_PROP_POS_FRAMES
   random-seek errors across all 25 pairs (highlighting VFR pairs pair_013, pair_014).
3. Evaluates canonical 60 Hz synchronized timeline stepping across the common overlap
   of all 25 pairs, recording strictly separated:
     - epsilon_sync_ms (Phase 11.6 inter-camera synchronization residual uncertainty)
     - delta_pts_front_ms (Front container PTS discretization offset)
     - delta_pts_back_ms (Back container PTS discretization offset)
     - inter_camera_pts_skew_ms
     - unique vs. repeated frame counts (30/60 fps deduplication verification)
4. Saves reproducible CSV and JSON validation artifacts under results/phase12c_pts_sync/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synchronization.pts_index import (
    FramePTSIndex,
    validate_pts_against_opencv_sequential,
)
from src.synchronization.synchronized_reader import SynchronizedVideoReader


def main() -> None:
    t0_wall = time.perf_counter()
    out_dir = PROJECT_ROOT / "results" / "phase12c_pts_sync"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = PROJECT_ROOT / "data" / "dual_camera_test_v1" / "metadata" / "synchronization_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        pairs_meta = json.load(f)

    stream_rows: list[dict] = []
    pair_alignment_rows: list[dict] = []

    for item in pairs_meta:
        pair_id = item["pair_id"]
        front_path = PROJECT_ROOT / "data" / "dual_camera_test_v1" / "front" / f"{pair_id}_front.mp4"
        back_path = PROJECT_ROOT / "data" / "dual_camera_test_v1" / "back" / f"{pair_id}_back.mp4"

        for cam, vpath in [("front", front_path), ("back", back_path)]:
            idx = FramePTSIndex.from_video(vpath)
            summary = idx.get_summary()
            val = validate_pts_against_opencv_sequential(vpath)
            row = {
                "pair_id": pair_id,
                "camera": cam,
                **summary,
                "opencv_decoded_frame_count": val["opencv_decoded_frame_count"],
                "frame_count_exact_match": val["frame_count_exact_match"],
                "opencv_frame0_pos_sec": val["opencv_frame0_pos_sec"],
                "seq_pos_msec_vs_norm_pts_mean_diff_ms": round(val["seq_pos_msec_vs_norm_pts_mean_diff_ms"], 6),
                "seq_pos_msec_vs_norm_pts_max_diff_ms": round(val["seq_pos_msec_vs_norm_pts_max_diff_ms"], 6),
                "seq_pos_msec_vs_raw_pts_max_diff_ms": round(val["seq_pos_msec_vs_raw_pts_max_diff_ms"], 4),
                "pos_frames_seek_error_at_mid_ms": val["pos_frames_seek_error_at_mid_ms"],
            }
            stream_rows.append(row)

        # Evaluate 60 Hz synchronized timeline stepping across overlap
        with SynchronizedVideoReader(pair_id) as sync_reader:
            info = sync_reader.sync_info
            idx_f = sync_reader.reader_front.pts_index
            idx_b = sync_reader.reader_back.pts_index

            t_start = info.overlap_start_back_sec
            t_end = info.overlap_end_back_sec
            dt = 1.0 / 60.0
            n_ticks = int(np.floor((t_end - t_start + 1e-9) / dt)) + 1

            delta_f_list: list[float] = []
            delta_b_list: list[float] = []
            skew_list: list[float] = []
            legacy_err_b_list: list[float] = []
            legacy_err_f_list: list[float] = []

            prev_b = -1
            prev_f = -1
            unique_b_count = 0
            unique_f_count = 0
            dup_b_suppressed = 0
            dup_f_suppressed = 0

            fps_b_nom = sync_reader.reader_back.fps
            fps_f_nom = sync_reader.reader_front.fps

            for k in range(max(0, n_ticks)):
                t_ref = t_start + k * dt
                if t_ref > t_end + 1e-6:
                    break
                t_front_target = t_ref - info.global_offset_sec

                lb = idx_b.nearest_frame(t_ref)
                lf = idx_f.nearest_frame(t_front_target)

                delta_b_list.append(lb.delta_pts_ms)
                delta_f_list.append(lf.delta_pts_ms)
                skew_ms = abs((lb.actual_pts_norm_sec - lf.actual_pts_norm_sec) - info.global_offset_sec) * 1000.0
                skew_list.append(skew_ms)

                if lb.frame_index != prev_b:
                    unique_b_count += 1
                    prev_b = lb.frame_index
                else:
                    dup_b_suppressed += 1

                if lf.frame_index != prev_f:
                    unique_f_count += 1
                    prev_f = lf.frame_index
                else:
                    dup_f_suppressed += 1

                # Legacy nominal frame index error at this tick
                leg_idx_b = min(max(0, int(round(t_ref * fps_b_nom))), idx_b.frame_count - 1)
                leg_idx_f = min(max(0, int(round(t_front_target * fps_f_nom))), idx_f.frame_count - 1)
                legacy_err_b_list.append(abs(float(idx_b.pts_norm_sec[leg_idx_b]) - t_ref) * 1000.0)
                legacy_err_f_list.append(abs(float(idx_f.pts_norm_sec[leg_idx_f]) - t_front_target) * 1000.0)

            pair_row = {
                "pair_id": pair_id,
                "sync_tier": info.accuracy_classification,
                "epsilon_sync_ms": info.epsilon_sync_ms,
                "global_offset_sec": info.global_offset_sec,
                "overlap_duration_sec": round(info.overlap_duration_sec, 4),
                "front_effective_fps": round(idx_f.effective_fps, 2),
                "back_effective_fps": round(idx_b.effective_fps, 2),
                "back_is_vfr": idx_b.is_vfr,
                "canonical_60hz_ticks": len(delta_b_list),
                "unique_front_frames_in_overlap": unique_f_count,
                "unique_back_frames_in_overlap": unique_b_count,
                "suppressed_duplicate_front_ticks": dup_f_suppressed,
                "suppressed_duplicate_back_ticks": dup_b_suppressed,
                "delta_pts_front_mean_ms": round(float(np.mean(delta_f_list)), 4),
                "delta_pts_front_p95_ms": round(float(np.percentile(delta_f_list, 95)), 4),
                "delta_pts_front_max_ms": round(float(np.max(delta_f_list)), 4),
                "delta_pts_back_mean_ms": round(float(np.mean(delta_b_list)), 4),
                "delta_pts_back_p95_ms": round(float(np.percentile(delta_b_list, 95)), 4),
                "delta_pts_back_max_ms": round(float(np.max(delta_b_list)), 4),
                "inter_camera_pts_skew_mean_ms": round(float(np.mean(skew_list)), 4),
                "inter_camera_pts_skew_max_ms": round(float(np.max(skew_list)), 4),
                "legacy_nominal_back_mean_err_ms": round(float(np.mean(legacy_err_b_list)), 4),
                "legacy_nominal_back_max_err_ms": round(float(np.max(legacy_err_b_list)), 4),
                "legacy_nominal_front_max_err_ms": round(float(np.max(legacy_err_f_list)), 4),
            }
            pair_alignment_rows.append(pair_row)

    # Save CSVs
    stream_csv = out_dir / "stream_pts_forensics_50_videos.csv"
    with open(stream_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stream_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stream_rows)

    pair_csv = out_dir / "pair_pts_alignment_validation_25_pairs.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_alignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_alignment_rows)

    # Measure peak RSS memory (macOS ru_maxrss is in bytes)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    elapsed_sec = time.perf_counter() - t0_wall

    summary_json = {
        "phase": "12C",
        "total_streams_audited": len(stream_rows),
        "total_pairs_audited": len(pair_alignment_rows),
        "all_50_frame_counts_exact_match": all(r["frame_count_exact_match"] for r in stream_rows),
        "max_seq_pos_msec_vs_norm_pts_diff_ms": max(r["seq_pos_msec_vs_norm_pts_max_diff_ms"] for r in stream_rows),
        "vfr_streams_detected": [r["filename"] for r in stream_rows if r["is_vfr"]],
        "cfr_30fps_streams_detected": [
            r["filename"] for r in stream_rows if (not r["is_vfr"] and r["effective_fps"] < 35.0)
        ],
        "peak_rss_memory_mb": round(peak_rss_mb, 2),
        "audit_elapsed_sec": round(elapsed_sec, 2),
        "pair_alignment_summary": pair_alignment_rows,
        "stream_forensics_summary": stream_rows,
    }
    with open(out_dir / "phase12c_pts_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(
        f"Phase 12C forensic audit complete in {elapsed_sec:.2f}s | "
        f"Peak RSS: {peak_rss_mb:.2f} MB | "
        f"Max seq POS_MSEC vs norm PTS diff: {summary_json['max_seq_pos_msec_vs_norm_pts_diff_ms']:.6f} ms"
    )


if __name__ == "__main__":
    main()
