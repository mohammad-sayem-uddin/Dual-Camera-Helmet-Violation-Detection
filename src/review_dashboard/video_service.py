"""Memory-bounded HTTP Range video streaming, lazy PTS lookup, and frame extraction."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import subprocess
import threading
from typing import Any
import cv2

from src.review_dashboard.blind_api import WorkqueueCatalog, resolve_video_path_safe
from src.review_dashboard.config import DashboardConfig
from src.review_dashboard.models import t_front_from_ref, t_ref_from_front
from src.synchronization.pts_index import FramePTSIndex


RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
STREAM_CHUNK_BYTES = 256 * 1024


def parse_byte_range(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    """Parse HTTP `Range: bytes=start-end` header into inclusive `(start, end)`."""
    if not range_header or file_size <= 0:
        return None
    m = RANGE_RE.match(range_header.strip())
    if not m:
        return None
    start_s, end_s = m.groups()
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        suffix = int(end_s)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(0, file_size - suffix)
        end = file_size - 1
        return start, end
    start = int(start_s)
    end = int(end_s) if end_s else file_size - 1
    if start >= file_size or end < start:
        raise ValueError(f"Range [{start}, {end}] unsatisfiable for size {file_size}.")
    return start, min(end, file_size - 1)


class BoundedVideoService:
    """Bounded-memory video streaming and PTS-exact single-frame extractor for 8 GB M1 Macs.

    Design guarantees:
    - Never loads all 50 videos at startup or into memory.
    - Keeps at most `max_pts_index_pairs` lightweight PTS tables in LRU cache.
    - Keeps at most ONE active OpenCV VideoCapture pair open at a time and calls
      `.release()` deterministically whenever switching pairs or releasing explicitly.
    """

    def __init__(self, config: DashboardConfig, catalog: WorkqueueCatalog) -> None:
        self.config = config
        self.catalog = catalog
        self._lock = threading.RLock()
        self._proxy_build_locks: dict[tuple[str, str], threading.Lock] = {}
        self._pts_cache: OrderedDict[tuple[str, str], FramePTSIndex] = OrderedDict()
        self._proxy_cache: OrderedDict[tuple[str, str], Path] = OrderedDict()
        self._active_pair_id: str | None = None
        self._active_caps: dict[str, tuple[cv2.VideoCapture, int]] = {}
        self._last_decoded_frames: dict[str, tuple[str, int, Any]] = {}
        self._release_count: int = 0

    def release_active_captures(self) -> int:
        """Explicitly release any open OpenCV VideoCapture instances."""
        with self._lock:
            released = 0
            for _cam, (cap, _cursor) in list(self._active_caps.items()):
                try:
                    cap.release()
                except Exception:
                    pass
                released += 1
            self._active_caps.clear()
            self._last_decoded_frames.clear()
            self._active_pair_id = None
            if released:
                self._release_count += released
            return released

    @staticmethod
    def _grab_to_frame(cap: cv2.VideoCapture, cursor: int, target_idx: int) -> tuple[bool, Any, int]:
        """Advance capture sequentially via grab() until target_idx and retrieve the decoded frame."""
        while cursor < target_idx:
            if not cap.grab():
                return False, None, -1
            cursor += 1
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            return False, None, -1
        return True, frame, target_idx

    def close(self) -> None:
        """Clean up active decoders and PTS caches on shutdown."""
        with self._lock:
            self.release_active_captures()
            self._pts_cache.clear()

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_pair_id": self._active_pair_id,
                "open_captures": len(self._active_caps),
                "cached_pts_streams": len(self._pts_cache),
                "max_pts_streams": self.config.max_pts_index_pairs * 2,
                "cached_proxy_streams": len(self._proxy_cache),
                "max_proxy_streams": self.config.max_proxy_pairs * 2,
                "total_captures_released": self._release_count,
            }

    def get_pts_index(self, pair_id: str, camera: str) -> FramePTSIndex:
        key = (pair_id, camera)
        with self._lock:
            if key in self._pts_cache:
                self._pts_cache.move_to_end(key)
                return self._pts_cache[key]

        path = resolve_video_path_safe(self.config, self.catalog, pair_id, camera)
        idx = FramePTSIndex.from_video(path)
        with self._lock:
            self._pts_cache[key] = idx
            self._pts_cache.move_to_end(key)
            max_streams = max(2, self.config.max_pts_index_pairs * 2)
            while len(self._pts_cache) > max_streams:
                self._pts_cache.popitem(last=False)
        return idx

    def resolve_stream_path(self, pair_id: str, camera: str, mode: str = "original") -> Path:
        """Return original untouched MP4 path or lazily build a bounded H.264 720p fallback proxy."""
        src = resolve_video_path_safe(self.config, self.catalog, pair_id, camera)
        if mode != "proxy_h264" or not self.config.allow_proxy_transcode:
            return src

        key = (pair_id, camera)
        proxy_path = self.config.proxy_cache_dir / f"{pair_id}_{camera}_h264_720p.mp4"
        with self._lock:
            if proxy_path.exists() and proxy_path.stat().st_size > 1024:
                self._proxy_cache[key] = proxy_path
                self._proxy_cache.move_to_end(key)
                return proxy_path
            key_lock = self._proxy_build_locks.setdefault(key, threading.Lock())
            self.config.proxy_cache_dir.mkdir(parents=True, exist_ok=True)

        # Transcode per stream key concurrently without holding the global service lock
        with key_lock:
            if proxy_path.exists() and proxy_path.stat().st_size > 1024:
                return proxy_path
            tmp_proxy = self.config.proxy_cache_dir / f".tmp_{pair_id}_{camera}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                "scale=1280:-2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp_proxy),
            ]
            subprocess.run(cmd, check=True, timeout=60)
            tmp_proxy.replace(proxy_path)
            with self._lock:
                self._proxy_cache[key] = proxy_path
                self._proxy_cache.move_to_end(key)
                max_proxies = max(2, self.config.max_proxy_pairs * 2)
                while len(self._proxy_cache) > max_proxies:
                    _old_key, old_path = self._proxy_cache.popitem(last=False)
                    if old_path.exists():
                        old_path.unlink(missing_ok=True)
            return proxy_path

    def lookup_sync_state(self, pair_id: str, t_ref_sec: float) -> dict[str, Any]:
        """Compute exact PTS lookup diagnostics on both streams at reference timeline t_ref_sec."""
        timing = self.catalog.get_pair_timing(pair_id)
        idx_back = self.get_pts_index(pair_id, "back")
        idx_front = self.get_pts_index(pair_id, "front")

        t_ref = float(t_ref_sec)
        t_front_target = t_front_from_ref(t_ref, timing.global_offset_sec)

        back_in_range = -0.02 <= t_ref <= idx_back.duration_norm_sec + 0.02
        front_in_range = -0.02 <= t_front_target <= idx_front.duration_norm_sec + 0.02
        in_overlap = (
            timing.overlap_start_ref_sec - 1e-4 <= t_ref <= timing.overlap_end_ref_sec + 1e-4
        )

        b_lookup = idx_back.nearest_frame(min(max(0.0, t_ref), idx_back.duration_norm_sec))
        f_lookup = idx_front.nearest_frame(min(max(0.0, t_front_target), idx_front.duration_norm_sec))

        return {
            "pair_id": pair_id,
            "t_ref_sec": round(t_ref, 6),
            "global_offset_sec": timing.global_offset_sec,
            "in_overlap": in_overlap,
            "back": {
                "in_range": back_in_range,
                "target_norm_sec": round(t_ref, 6),
                "frame_index": b_lookup.frame_index,
                "frame_count": idx_back.frame_count,
                "actual_pts_norm_sec": round(b_lookup.actual_pts_norm_sec, 6),
                "delta_pts_ms": round(abs((b_lookup.actual_pts_norm_sec - t_ref) * 1000.0), 3),
                "duration_sec": round(idx_back.duration_norm_sec, 6),
            },
            "front": {
                "in_range": front_in_range,
                "target_norm_sec": round(t_front_target, 6),
                "frame_index": f_lookup.frame_index,
                "frame_count": idx_front.frame_count,
                "actual_pts_norm_sec": round(f_lookup.actual_pts_norm_sec, 6),
                "actual_t_ref_sec": round(t_ref_from_front(f_lookup.actual_pts_norm_sec, timing.global_offset_sec), 6),
                "delta_pts_ms": round(abs((f_lookup.actual_pts_norm_sec - t_front_target) * 1000.0), 3),
                "duration_sec": round(idx_front.duration_norm_sec, 6),
            },
        }

    def step_reference_time(self, pair_id: str, t_ref_sec: float, delta_frames: int, anchor_camera: str = "back") -> float:
        """Compute exact next/previous physical frame timestamp on `anchor_camera` converted to t_ref_sec."""
        timing = self.catalog.get_pair_timing(pair_id)
        idx = self.get_pts_index(pair_id, anchor_camera)
        if anchor_camera == "back":
            target_norm = float(t_ref_sec)
        else:
            target_norm = t_front_from_ref(float(t_ref_sec), timing.global_offset_sec)
        cur = idx.nearest_frame(min(max(0.0, target_norm), idx.duration_norm_sec))
        next_idx = min(max(0, cur.frame_index + int(delta_frames)), idx.frame_count - 1)
        next_norm = float(idx.pts_norm_sec[next_idx])
        if anchor_camera == "back":
            return round(next_norm, 6)
        return round(t_ref_from_front(next_norm, timing.global_offset_sec), 6)

    def _get_pair_capture(self, pair_id: str, camera: str) -> tuple[cv2.VideoCapture, int]:
        """Return bounded active VideoCapture for `(pair_id, camera)`; closes previous pair first."""
        if self._active_pair_id != pair_id:
            self.release_active_captures()
            self._active_pair_id = pair_id

        if camera in self._active_caps:
            cap, cursor = self._active_caps[camera]
            if cap.isOpened():
                return cap, cursor
            try:
                cap.release()
            except Exception:
                pass
            self._release_count += 1
            self._active_caps.pop(camera, None)

        path = resolve_video_path_safe(self.config, self.catalog, pair_id, camera)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video capture: {path}")
        self._active_caps[camera] = (cap, -1)
        return cap, -1

    def extract_frame_data(
        self,
        pair_id: str,
        camera: str,
        *,
        t_ref_sec: float | None = None,
        t_norm_sec: float | None = None,
        frame_index: int | None = None,
        fmt: str = "jpg",
        max_width: int | None = None,
        crop: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Extract one PTS-verified frame (and optional zoom crop) without keeping clips in RAM."""
        timing = self.catalog.get_pair_timing(pair_id)
        pts_index = self.get_pts_index(pair_id, camera)

        if frame_index is not None:
            target_idx = min(max(0, int(frame_index)), pts_index.frame_count - 1)
            pts_norm = float(pts_index.pts_norm_sec[target_idx])
            delta_ms = 0.0
        else:
            if t_norm_sec is not None:
                req_norm = float(t_norm_sec)
            elif t_ref_sec is not None:
                req_norm = float(t_ref_sec) if camera == "back" else t_front_from_ref(float(t_ref_sec), timing.global_offset_sec)
            else:
                req_norm = 0.0
            clamped = min(max(0.0, req_norm), pts_index.duration_norm_sec)
            lookup = pts_index.nearest_frame(clamped)
            target_idx = lookup.frame_index
            pts_norm = lookup.actual_pts_norm_sec
            delta_ms = abs((pts_norm - req_norm) * 1000.0)

        ret = False
        frame = None
        with self._lock:
            cached = self._last_decoded_frames.get(camera)
            if cached is not None and cached[0] == pair_id and cached[1] == target_idx:
                ret, frame = True, cached[2]
            else:
                cap, cursor = self._get_pair_capture(pair_id, camera)
                # Never use cv2.CAP_PROP_POS_FRAMES: VFR H.264 and B-frame HEVC streams seek
                # inaccurately (landing on wrong frames or failing near EOF). Reset and grab sequentially.
                if cursor >= 0 and target_idx < cursor:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self._release_count += 1
                    self._active_caps.pop(camera, None)
                    self._last_decoded_frames.pop(camera, None)
                    cap, cursor = self._get_pair_capture(pair_id, camera)

                started_fresh = cursor < 0
                ret, frame, new_cursor = self._grab_to_frame(cap, cursor, target_idx)
                if (not ret or frame is None) and not started_fresh:
                    # Single clean reopen fallback if advancing an existing capture failed
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self._release_count += 1
                    self._active_caps.pop(camera, None)
                    self._last_decoded_frames.pop(camera, None)
                    cap, cursor = self._get_pair_capture(pair_id, camera)
                    ret, frame, new_cursor = self._grab_to_frame(cap, cursor, target_idx)

                self._active_caps[camera] = (cap, new_cursor)
                if ret and frame is not None:
                    self._last_decoded_frames[camera] = (pair_id, target_idx, frame)
                else:
                    self._last_decoded_frames.pop(camera, None)

        if not ret or frame is None:
            raise RuntimeError(f"Could not decode frame {target_idx} for {pair_id}/{camera}")

        h, w = frame.shape[:2]
        if crop is not None:
            cx, cy, cw, ch = crop
            x0 = min(max(0, cx), max(0, w - 1))
            y0 = min(max(0, cy), max(0, h - 1))
            x1 = min(w, max(x0 + 1, x0 + cw))
            y1 = min(h, max(y0 + 1, y0 + ch))
            frame = frame[y0:y1, x0:x1]

        if max_width and frame.shape[1] > max_width:
            scale = float(max_width) / float(frame.shape[1])
            new_h = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (int(max_width), new_h), interpolation=cv2.INTER_AREA)

        is_png = fmt.lower() == "png"
        ext = ".png" if is_png else ".jpg"
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3] if is_png else [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        ok, buf = cv2.imencode(ext, frame, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode frame.")

        t_ref_actual = pts_norm if camera == "back" else t_ref_from_front(pts_norm, timing.global_offset_sec)
        return {
            "bytes": buf.tobytes(),
            "mime_type": "image/png" if is_png else "image/jpeg",
            "ext": "png" if is_png else "jpg",
            "frame_index": target_idx,
            "pts_norm_sec": round(pts_norm, 6),
            "t_ref_sec": round(t_ref_actual, 6),
            "delta_pts_ms": round(delta_ms, 3),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
