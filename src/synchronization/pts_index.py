"""Presentation Timestamp (PTS) Indexing and Forensic Validation Engine.

Builds exact presentation-order timestamp lookup tables from container packet/stream
metadata (via ffprobe rational time_base) and validates them against sequential
OpenCV decoded frame timestamps (CAP_PROP_POS_MSEC).

Metrological Separation Note:
- Container PTS indexing measures and resolves intra-stream variable frame rate (VFR)
  and frame-rate discretization residuals (delta_pts_front_ms, delta_pts_back_ms).
- It does NOT eliminate the inter-camera synchronization residual (epsilon_sync_ms)
  or establish physical CMOS rolling-shutter exposure genlock.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any
import cv2
import numpy as np


@dataclass(frozen=True)
class PTSLookupResult:
    """Result of nearest-frame PTS lookup for a target stream-normalized timestamp."""
    frame_index: int
    target_norm_sec: float
    actual_pts_norm_sec: float
    actual_pts_raw_sec: float
    pts_ticks: int
    delta_pts_ms: float
    signed_delta_pts_ms: float
    local_frame_interval_ms: float


class FramePTSIndex:
    """
    Monotonically sorted Presentation Timestamp (PTS) lookup table for a single video stream.
    """

    def __init__(
        self,
        video_path: str | Path,
        pts_ticks: np.ndarray,
        time_base_num: int,
        time_base_den: int,
        r_frame_rate_str: str = "60/1",
        avg_frame_rate_str: str = "60/1",
    ) -> None:
        self.video_path = Path(video_path)
        if len(pts_ticks) == 0:
            raise ValueError(f"Empty PTS table for video: {self.video_path}")
        if time_base_num <= 0 or time_base_den <= 0:
            raise ValueError(f"Invalid rational time_base: {time_base_num}/{time_base_den}")

        # Ensure presentation order (sorted by PTS ticks)
        sorted_ticks = np.sort(np.asarray(pts_ticks, dtype=np.int64))
        self.pts_ticks = sorted_ticks
        self.time_base_num = int(time_base_num)
        self.time_base_den = int(time_base_den)
        self.time_base_sec = float(time_base_num) / float(time_base_den)
        self.r_frame_rate_str = r_frame_rate_str
        self.avg_frame_rate_str = avg_frame_rate_str

        # Raw container timestamps in seconds (may have non-zero start_pts due to audio lead)
        self.pts_raw_sec = self.pts_ticks.astype(np.float64) * self.time_base_sec
        self.start_pts_ticks = int(self.pts_ticks[0])
        self.start_time_raw_sec = float(self.pts_raw_sec[0])

        # Stream-normalized timestamps anchored to Frame 0 == 0.000000 s
        self.pts_norm_sec = (self.pts_ticks - self.pts_ticks[0]).astype(np.float64) * self.time_base_sec
        self.frame_count = int(len(self.pts_ticks))
        self.duration_norm_sec = float(self.pts_norm_sec[-1]) if self.frame_count > 1 else 0.0

        # Inter-frame intervals in milliseconds
        if self.frame_count > 1:
            self.frame_intervals_ms = np.diff(self.pts_norm_sec) * 1000.0
            self.min_interval_ms = float(np.min(self.frame_intervals_ms))
            self.median_interval_ms = float(np.median(self.frame_intervals_ms))
            self.mean_interval_ms = float(np.mean(self.frame_intervals_ms))
            self.max_interval_ms = float(np.max(self.frame_intervals_ms))
            self.std_interval_ms = float(np.std(self.frame_intervals_ms))
            self.effective_fps = (self.frame_count - 1) / self.duration_norm_sec if self.duration_norm_sec > 0 else 0.0
        else:
            self.frame_intervals_ms = np.array([16.6667], dtype=np.float64)
            self.min_interval_ms = 16.6667
            self.median_interval_ms = 16.6667
            self.mean_interval_ms = 16.6667
            self.max_interval_ms = 16.6667
            self.std_interval_ms = 0.0
            self.effective_fps = 60.0

        # A stream is classified as VFR if inter-frame interval spread exceeds 5.0 ms
        self.interval_spread_ms = self.max_interval_ms - self.min_interval_ms
        self.is_vfr = bool(self.interval_spread_ms > 5.0)

        # Count 60fps (~16.7ms) vs 30fps (~33.3ms) vs other intervals
        self.count_dt_60fps = int(np.sum((self.frame_intervals_ms >= 12.0) & (self.frame_intervals_ms <= 22.0)))
        self.count_dt_30fps = int(np.sum((self.frame_intervals_ms > 22.0) & (self.frame_intervals_ms <= 40.0)))
        self.count_dt_other = int(len(self.frame_intervals_ms) - self.count_dt_60fps - self.count_dt_30fps)

    @classmethod
    def from_video(cls, video_path: str | Path) -> FramePTSIndex:
        """
        Build a FramePTSIndex from a video file using ffprobe rational packet PTS metadata,
        with fallback to sequential OpenCV CAP_PROP_POS_MSEC scan if ffprobe is unavailable.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=time_base,r_frame_rate,avg_frame_rate:packet=pts",
                "-of", "json",
                str(path),
            ]
            out = subprocess.check_output(cmd, timeout=30)
            data = json.loads(out)
            streams = data.get("streams", [])
            packets = data.get("packets", [])
            if streams and packets:
                st = streams[0]
                tb_str = st.get("time_base", "1/90000")
                tb_num, tb_den = map(int, tb_str.split("/"))
                r_fps = st.get("r_frame_rate", "60/1")
                avg_fps = st.get("avg_frame_rate", "60/1")
                pts_list = [int(p["pts"]) for p in packets if "pts" in p and p["pts"] is not None]
                if pts_list:
                    return cls(
                        video_path=path,
                        pts_ticks=np.array(pts_list, dtype=np.int64),
                        time_base_num=tb_num,
                        time_base_den=tb_den,
                        r_frame_rate_str=r_fps,
                        avg_frame_rate_str=avg_fps,
                    )
        except (subprocess.SubprocessError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            pass

        # Fallback: Sequential OpenCV grab scan in microsecond ticks (1/1,000,000)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for PTS indexing: {path}")
        pos_us: list[int] = []
        while True:
            ret = cap.grab()
            if not ret:
                break
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            pos_us.append(int(round(pos_ms * 1000.0)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        return cls(
            video_path=path,
            pts_ticks=np.array(pos_us, dtype=np.int64),
            time_base_num=1,
            time_base_den=1_000_000,
            r_frame_rate_str=f"{fps:.3f}/1",
            avg_frame_rate_str=f"{fps:.3f}/1",
        )

    def nearest_frame(self, target_norm_sec: float) -> PTSLookupResult:
        """
        Locate the frame index whose stream-normalized PTS is closest to target_norm_sec
        using O(log N) binary search (np.searchsorted).
        """
        idx_right = int(np.searchsorted(self.pts_norm_sec, target_norm_sec, side="left"))

        if idx_right <= 0:
            best_idx = 0
        elif idx_right >= self.frame_count:
            best_idx = self.frame_count - 1
        else:
            idx_left = idx_right - 1
            diff_left = abs(float(self.pts_norm_sec[idx_left]) - target_norm_sec)
            diff_right = abs(float(self.pts_norm_sec[idx_right]) - target_norm_sec)
            best_idx = idx_left if diff_left <= diff_right else idx_right

        actual_norm = float(self.pts_norm_sec[best_idx])
        actual_raw = float(self.pts_raw_sec[best_idx])
        ticks = int(self.pts_ticks[best_idx])
        signed_delta_ms = (actual_norm - target_norm_sec) * 1000.0
        delta_ms = abs(signed_delta_ms)

        if self.frame_count > 1:
            interval_idx = min(best_idx, self.frame_count - 2)
            local_dt_ms = float(self.frame_intervals_ms[interval_idx])
        else:
            local_dt_ms = 16.6667

        return PTSLookupResult(
            frame_index=best_idx,
            target_norm_sec=float(target_norm_sec),
            actual_pts_norm_sec=actual_norm,
            actual_pts_raw_sec=actual_raw,
            pts_ticks=ticks,
            delta_pts_ms=delta_ms,
            signed_delta_pts_ms=signed_delta_ms,
            local_frame_interval_ms=local_dt_ms,
        )

    def nominal_fps_timestamp_error_stats(self, nominal_fps: float | None = None) -> dict[str, float]:
        """
        Compute the timestamp error (in ms) that would occur if frame timestamps were
        estimated naively as frame_idx / nominal_fps instead of true container PTS.
        """
        fps_to_use = float(nominal_fps) if nominal_fps and nominal_fps > 0 else self.effective_fps
        indices = np.arange(self.frame_count, dtype=np.float64)
        nominal_sec = indices / fps_to_use
        err_ms = np.abs(nominal_sec - self.pts_norm_sec) * 1000.0
        return {
            "nominal_fps_used": round(fps_to_use, 4),
            "mean_error_ms": float(np.mean(err_ms)),
            "median_error_ms": float(np.median(err_ms)),
            "p95_error_ms": float(np.percentile(err_ms, 95)),
            "max_error_ms": float(np.max(err_ms)),
            "max_error_frame_idx": int(np.argmax(err_ms)),
        }

    def get_summary(self) -> dict[str, Any]:
        """Return a serializable dictionary of stream timing forensics."""
        nom_err = self.nominal_fps_timestamp_error_stats()
        return {
            "filename": self.video_path.name,
            "frame_count": self.frame_count,
            "time_base": f"{self.time_base_num}/{self.time_base_den}",
            "r_frame_rate": self.r_frame_rate_str,
            "avg_frame_rate": self.avg_frame_rate_str,
            "effective_fps": round(self.effective_fps, 4),
            "start_pts_ticks": self.start_pts_ticks,
            "start_time_raw_sec": round(self.start_time_raw_sec, 6),
            "duration_norm_sec": round(self.duration_norm_sec, 6),
            "is_vfr": self.is_vfr,
            "min_interval_ms": round(self.min_interval_ms, 4),
            "median_interval_ms": round(self.median_interval_ms, 4),
            "mean_interval_ms": round(self.mean_interval_ms, 4),
            "max_interval_ms": round(self.max_interval_ms, 4),
            "std_interval_ms": round(self.std_interval_ms, 4),
            "interval_spread_ms": round(self.interval_spread_ms, 4),
            "count_dt_60fps": self.count_dt_60fps,
            "count_dt_30fps": self.count_dt_30fps,
            "count_dt_other": self.count_dt_other,
            "nominal_fps_mean_error_ms": round(nom_err["mean_error_ms"], 4),
            "nominal_fps_p95_error_ms": round(nom_err["p95_error_ms"], 4),
            "nominal_fps_max_error_ms": round(nom_err["max_error_ms"], 4),
            "nominal_fps_max_error_frame_idx": nom_err["max_error_frame_idx"],
        }


def validate_pts_against_opencv_sequential(video_path: str | Path) -> dict[str, Any]:
    """
    Empirically validate the ffprobe rational PTS table against OpenCV's sequential
    decoder timestamps (CAP_PROP_POS_MSEC after sequential cap.grab()).
    Also measures the seek error of OpenCV's random CAP_PROP_POS_FRAMES seek at the midpoint frame.
    """
    pts_index = FramePTSIndex.from_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cv_pos_sec_list: list[float] = []
    while True:
        ret = cap.grab()
        if not ret:
            break
        cv_pos_sec_list.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)

    # Also test random CAP_PROP_POS_FRAMES seek at midpoint frame
    mid_idx = pts_index.frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret_seek = cap.grab()
    seek_pos_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 if ret_seek else -1.0
    cap.release()

    cv_pos_sec = np.array(cv_pos_sec_list, dtype=np.float64)
    cv_norm_sec = cv_pos_sec - cv_pos_sec[0] if len(cv_pos_sec) > 0 else cv_pos_sec

    n_common = min(len(cv_norm_sec), pts_index.frame_count)
    diff_norm_ms = np.abs(cv_norm_sec[:n_common] - pts_index.pts_norm_sec[:n_common]) * 1000.0
    diff_raw_ms = np.abs(cv_pos_sec[:n_common] - pts_index.pts_raw_sec[:n_common]) * 1000.0

    true_mid_sec = float(pts_index.pts_norm_sec[mid_idx])
    pos_frames_seek_err_ms = abs(seek_pos_sec - true_mid_sec) * 1000.0 if seek_pos_sec >= 0 else 0.0

    return {
        "filename": pts_index.video_path.name,
        "ffprobe_frame_count": pts_index.frame_count,
        "opencv_decoded_frame_count": len(cv_pos_sec),
        "frame_count_exact_match": (pts_index.frame_count == len(cv_pos_sec)),
        "ffprobe_start_time_raw_sec": round(pts_index.start_time_raw_sec, 6),
        "opencv_frame0_pos_sec": round(float(cv_pos_sec[0]), 6) if len(cv_pos_sec) > 0 else 0.0,
        "seq_pos_msec_vs_norm_pts_mean_diff_ms": float(np.mean(diff_norm_ms)),
        "seq_pos_msec_vs_norm_pts_max_diff_ms": float(np.max(diff_norm_ms)),
        "seq_pos_msec_vs_raw_pts_max_diff_ms": float(np.max(diff_raw_ms)),
        "mid_frame_idx": mid_idx,
        "mid_frame_true_pts_norm_sec": round(true_mid_sec, 6),
        "mid_frame_pos_frames_seek_sec": round(seek_pos_sec, 6),
        "pos_frames_seek_error_at_mid_ms": round(pos_frames_seek_err_ms, 4),
    }
