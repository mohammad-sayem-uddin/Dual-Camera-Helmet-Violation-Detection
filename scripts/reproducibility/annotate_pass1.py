"""Pass 1 annotation runner (annotation-side code; does NOT modify approved Phase 12 modules).

For one synchronized pair:
  - Sequentially decodes EVERY frame of each video (no seeking), PTS from the
    Phase 12C FramePTSIndex (container PTS, presentation order).
  - Runs the approved Phase 12D PerCameraDetector + CameraTrackingPipeline per frame.
  - Records coverage, raw detections, per-frame tracked state, final track objects.
  - Runs the approved Phase 12E CrossCameraAssociationEngine on the two pipelines.
  - Saves audit contact sheets (sampled frames with tracked overlays).

Outputs go to --out-dir (kept outside the repo dataset folder during processing;
the final package is assembled later). Originals are opened read-only.
"""
from __future__ import annotations

import argparse, csv, gc, json, os, resource, sys, time
from pathlib import Path

REPO = Path(os.environ.get("RESEARCH_HELMET_ROOT", os.environ.get("HELMET_REPO", str(Path(__file__).resolve().parents[2]))))
sys.path.insert(0, str(REPO))
os.chdir(REPO)
DEVICE = os.environ.get("HELMET_DEVICE", "cpu")
try:
    import torch
    torch.set_num_threads(int(os.environ.get("HELMET_TORCH_THREADS", "4")))
except Exception:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.synchronization.pts_index import FramePTSIndex  # noqa: E402
from src.detection.camera_detector import PerCameraDetector  # noqa: E402
from src.tracking.camera_tracker import CameraTrackingPipeline  # noqa: E402
from src.association.cross_camera_association import CrossCameraAssociationEngine  # noqa: E402
from src.association.tracklet_models import TimestampQualityRecord  # noqa: E402


def load_sync(pair_id: str) -> dict:
    data = json.loads((REPO / "data/dual_camera_test_v1/metadata/synchronization_metadata.json").read_text())
    for item in data:
        if item["pair_id"] == pair_id:
            return item
    raise ValueError(pair_id)


def run_camera(pair_id: str, cam: str, video_path: Path, out_dir: Path, detector) -> dict:
    pts_index = FramePTSIndex.from_video(video_path)
    pts_norm = pts_index.pts_norm_sec
    pts_raw = pts_index.pts_raw_sec
    n_expected = int(pts_index.frame_count)
    eff_fps = float(pts_index.effective_fps)

    pipe = CameraTrackingPipeline.from_yaml(cam, REPO / "configs/tracking_config.yaml")

    cov_path = out_dir / f"{pair_id}_{cam}_coverage.csv"
    det_path = out_dir / f"{pair_id}_{cam}_detections.jsonl"
    trk_path = out_dir / f"{pair_id}_{cam}_track_frames.jsonl"

    key_frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    t0 = time.perf_counter()
    with open(cov_path, "w", newline="") as cf, open(det_path, "w") as df, open(trk_path, "w") as tf:
        cw = csv.writer(cf)
        cw.writerow(["pair_id", "camera", "video_file", "frame_index", "pts_raw_sec", "pts_norm_sec",
                     "n_motorcycle_det", "n_person_det", "n_head_det", "detector_ran", "tracked_moto_active",
                     "tracked_riders_active", "orphans_active"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n >= n_expected:
                break
            pts = float(pts_norm[n])
            bundle = detector.detect_frame(frame, n, pts)
            step = pipe.step(bundle, should_update_state=True)
            motos = step.get("active_moto_tracks", [])
            n_riders = sum(len(m.get("rider_subtracks", {})) for m in motos)
            orphans = step.get("active_orphan_tracks", [])
            cw.writerow([pair_id, cam, video_path.name, n, f"{float(pts_raw[n]):.6f}", f"{pts:.6f}",
                         len(bundle.motorcycles), len(bundle.persons), len(bundle.all_heads), 1,
                         len(motos), n_riders, len(orphans)])
            df.write(json.dumps({
                "frame_index": n, "pts_norm_sec": round(pts, 6), "pts_raw_sec": round(float(pts_raw[n]), 6),
                "detections": [d.to_dict() for d in
                               list(bundle.motorcycles) + list(bundle.persons) + list(bundle.all_heads)],
            }) + "\n")
            tf.write(json.dumps({
                "frame_index": n, "pts_norm_sec": round(pts, 6),
                "active_moto_tracks": motos, "active_orphan_tracks": orphans,
                "new_alerts": step.get("new_alerts", []),
            }, default=str) + "\n")
            # cache a bounded set of frames for association appearance features:
            # frames with moto/head detections, spread across the video
            if (bundle.motorcycles or bundle.all_heads) and len(key_frames) < 24 and n % 3 == 0:
                small = cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame.copy()
                key_frames[n] = small
            n += 1
    cap.release()
    wall = time.perf_counter() - t0

    # final track objects (terminated + still active)
    def moto_full(m):
        d = m.to_dict()
        d["rider_pts_history"] = {
            str(s_idx): [[int(f), round(float(p), 6), [round(float(c), 2) for c in bb], str(st), round(float(cf_), 4)]
                         for (f, p, bb, st, cf_) in sub.pts_history]
            for s_idx, sub in sorted(m.rider_subtracks.items())
        }
        return d

    def orphan_full(o):
        d = o.to_dict()
        d["pts_history"] = [[int(f), round(float(p), 6), [round(float(c), 2) for c in bb], str(st), round(float(cf_), 4)]
                            for (f, p, bb, st, cf_) in o.pts_history]
        return d

    tracks = {
        "moto_tracks": [moto_full(m) for m in list(pipe.terminated_moto_tracks) + list(pipe.active_moto_tracks.values())],
        "orphan_tracks": [orphan_full(o) for o in list(pipe.terminated_orphan_tracks) + list(pipe.active_orphan_tracks.values())],
        "frames_updated": pipe.frames_updated,
    }
    (out_dir / f"{pair_id}_{cam}_tracks_final.json").write_text(json.dumps(tracks, indent=1, default=str))

    grid = np.arange(n) / eff_fps if n else np.zeros(0)
    delta_ms = np.abs(pts_norm[:n] - grid) * 1000.0 if n else np.zeros(0)
    summary = {
        "pair_id": pair_id, "camera": cam, "video_file": video_path.name,
        "frames_expected_container": n_expected, "frames_decoded_sequential": n,
        "coverage_complete": bool(n == n_expected),
        "duration_norm_sec": round(float(pts_index.duration_norm_sec), 6),
        "effective_fps": round(eff_fps, 4), "is_vfr": bool(pts_index.is_vfr),
        "mean_delta_pts_ms": round(float(np.mean(delta_ms)), 4) if n else 0.0,
        "max_delta_pts_ms": round(float(np.max(delta_ms)), 4) if n else 0.0,
        "moto_tracks_total": len(tracks["moto_tracks"]),
        "orphan_tracks_total": len(tracks["orphan_tracks"]),
        "wall_sec": round(wall, 2),
        "ms_per_frame": round(wall * 1000.0 / n, 1) if n else 0.0,
    }
    return {"summary": summary, "pipeline": pipe, "key_frames": key_frames,
            "pts_norm": pts_norm, "delta_by_frame": {i: float(delta_ms[i]) for i in range(n)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out-root", default=os.environ.get("HELMET_PASS1_OUT", str(REPO / "results/pass1")))
    args = ap.parse_args()
    pair_id = args.pair
    out_dir = Path(args.out_root) / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_sync(pair_id)
    offset = float(meta["offset_sec"])
    eps = float(meta["residual_error_ms"])

    t_pair = time.perf_counter()
    results = {}
    detectors = {}
    for cam in ("front", "back"):
        vp = REPO / "data/dual_camera_test_v1" / cam / f"{pair_id}_{cam}.mp4"
        if not vp.exists():
            print(f"SKIP {pair_id} {cam}: video missing on VM", flush=True)
            return
        detectors[cam] = PerCameraDetector(camera_id=cam, device=DEVICE)
        print(f"[{pair_id}] {cam}: start", flush=True)
        results[cam] = run_camera(pair_id, cam, vp, out_dir, detectors[cam])
        print(f"[{pair_id}] {cam}: {results[cam]['summary']}", flush=True)
        gc.collect()

    # canonical (back-clock) PTS maps
    can = {}
    for cam in ("front", "back"):
        pts = results[cam]["pts_norm"]
        shift = offset if cam == "front" else 0.0
        can[cam] = {i: float(pts[i]) + shift for i in range(len(pts))}

    tq = TimestampQualityRecord(
        pair_id=pair_id, reference_clock_anchor="back_stream_pts",
        global_offset_sec=offset, epsilon_sync_ms=eps,
        mean_delta_pts_front_ms=results["front"]["summary"]["mean_delta_pts_ms"],
        max_delta_pts_front_ms=results["front"]["summary"]["max_delta_pts_ms"],
        mean_delta_pts_back_ms=results["back"]["summary"]["mean_delta_pts_ms"],
        max_delta_pts_back_ms=results["back"]["summary"]["max_delta_pts_ms"],
        is_in_overlap=True,
    )
    engine = CrossCameraAssociationEngine.from_yaml(REPO / "configs/cross_camera_association_config.yaml")
    batch = engine.associate_pipelines(
        pair_id=pair_id,
        front_pipeline=results["front"]["pipeline"], back_pipeline=results["back"]["pipeline"],
        timestamp_quality=tq,
        front_frames_by_index=results["front"]["key_frames"], back_frames_by_index=results["back"]["key_frames"],
        front_canonical_pts_by_frame=can["front"], back_canonical_pts_by_frame=can["back"],
        front_delta_pts_ms_by_frame=results["front"]["delta_by_frame"],
        back_delta_pts_ms_by_frame=results["back"]["delta_by_frame"],
        calibrated_homography_available=False,
    )
    (out_dir / f"{pair_id}_association.json").write_text(json.dumps(batch.to_dict(), indent=1, default=str))

    summary = {
        "pair_id": pair_id, "global_offset_sec": offset, "epsilon_sync_ms": eps,
        "sync_quality": meta.get("quality"), "drift_status": meta.get("drift_status"),
        "front": results["front"]["summary"], "back": results["back"]["summary"],
        "association_outcome_counts": batch.outcome_counts,
        "rider_outcome_counts": batch.rider_outcome_counts,
        "orphan_outcome_counts": batch.orphan_outcome_counts,
        "front_tracklets": len(batch.front_motorcycle_tracklets),
        "back_tracklets": len(batch.back_motorcycle_tracklets),
        "pair_wall_sec": round(time.perf_counter() - t_pair, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }
    (out_dir / f"{pair_id}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[{pair_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
