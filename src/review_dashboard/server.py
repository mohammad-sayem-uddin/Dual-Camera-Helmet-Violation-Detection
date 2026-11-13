"""HTTP server and strict role-isolated API handlers for the blind human review dashboard."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from src.review_dashboard import DASHBOARD_VERSION
from src.review_dashboard.blind_api import (
    BlindPermissionError,
    WorkqueueCatalog,
    assert_adjudicator_actor,
    assert_reviewer_actor,
    sanitize_item_for_reviewer,
)
from src.review_dashboard.config import DashboardConfig
from src.review_dashboard.models import (
    CAMERA_EVIDENCE_CHOICES,
    CROSS_CAMERA_LINK_STATUSES,
    HELMET_STATES,
    HELMET_TO_V1_1_LABEL,
    PLATE_VISIBILITY_TIERS,
    REVIEWER_LABELS,
    RIDER_ROLES,
    UNCERTAINTY_REASON_TAGS,
    VALID_ACTORS,
    VIOLATION_EVENT_TYPES,
)
from src.review_dashboard.storage import ConflictError, ReviewStorage, ValidationFailedError
from src.review_dashboard.video_service import BoundedVideoService, STREAM_CHUNK_BYTES, parse_byte_range


STATIC_DIR = Path(__file__).resolve().parent / "static"


class ReviewAppContext:
    """Shared thread-safe application services bound to the HTTP server."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self.catalog = WorkqueueCatalog(config)
        self.storage = ReviewStorage(config, self.catalog)
        self.video = BoundedVideoService(config, self.catalog)

    def close(self) -> None:
        self.video.close()


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing strict blind API isolation and byte-range video streaming."""

    protocol_version = "HTTP/1.1"
    server_version = f"BlindGTReview/{DASHBOARD_VERSION}"

    @property
    def app(self) -> ReviewAppContext:
        return self.server.app_context  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy per-range-chunk console logs unless explicitly debugging
        if getattr(self.server, "quiet_logs", True):
            return
        super().log_message(format, *args)

    def _send_json(
        self,
        status: int,
        payload: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, error: str, details: Any = None) -> None:
        # Close keep-alive sockets after errors so unconsumed POST/PUT bodies never corrupt the next request
        self.close_connection = True
        payload: dict[str, Any] = {"error": error}
        if details is not None:
            payload["details"] = details
        self._send_json(status, payload, extra_headers={"Connection": "close"})

    def _read_json_body(self) -> dict[str, Any]:
        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > 8 * 1024 * 1024:
            raise ValueError("Request body exceeds 8 MiB limit.")
        raw = self.rfile.read(length)

        def _reject_constant(token: str) -> Any:
            raise ValueError(f"Non-standard JSON token not allowed: {token!r}")

        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def _resolve_actor(self) -> str | None:
        hdr = self.headers.get("X-Reviewer-Id") or self.headers.get("X-Review-Actor")
        if hdr is not None and hdr.strip() != "":
            actor = hdr.strip()
            if actor not in VALID_ACTORS:
                raise ValueError(f"Unsupported actor identifier: {actor!r}")
            return actor

        cookie_hdr = self.headers.get("Cookie")
        if cookie_hdr:
            cookie = SimpleCookie()
            cookie.load(cookie_hdr)
            morsel = cookie.get("gt_review_actor")
            if morsel and morsel.value in VALID_ACTORS:
                return morsel.value
        return None

    def _require_any_actor(self) -> str:
        actor = self._resolve_actor()
        if actor is None:
            raise PermissionError("No active session role selected. Select Reviewer A, Reviewer B, or Adjudicator.")
        return actor

    def _require_reviewer(self, query: dict[str, list[str]]) -> str:
        actor = self._require_any_actor()
        assert_reviewer_actor(actor)
        requested_peer = (query.get("reviewer_id") or [""])[0].strip()
        if requested_peer and requested_peer != actor:
            raise BlindPermissionError("Cross-reviewer access forbidden.")
        return actor

    def _require_adjudicator(self) -> str:
        actor = self._require_any_actor()
        return assert_adjudicator_actor(actor)

    def _serve_static(self, rel_path: str) -> None:
        clean = "index.html" if rel_path in ("", "/") else rel_path.lstrip("/")
        if clean not in ("index.html", "styles.css", "app.js"):
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Static asset not found: {clean}")
            return
        file_path = (STATIC_DIR / clean).resolve()
        if STATIC_DIR.resolve() not in file_path.parents or not file_path.exists():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Static asset missing: {clean}")
            return
        ctype, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{ctype or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_video(self, pair_id: str, camera: str, query: dict[str, list[str]]) -> None:
        mode = (query.get("mode") or ["original"])[0]
        video_path = self.app.video.resolve_stream_path(pair_id, camera, mode=mode)
        file_size = video_path.stat().st_size
        range_hdr = self.headers.get("Range")

        try:
            byte_range = parse_byte_range(range_hdr, file_size)
        except ValueError as exc:
            try:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if byte_range is not None:
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT
        else:
            start, end = 0, file_size - 1
            status = HTTPStatus.OK

        content_len = (end - start) + 1
        try:
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_len))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command == "HEAD":
                return

            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_len
                while remaining > 0:
                    chunk = f.read(min(STREAM_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when browser seeks mid-stream
            pass

    def _serve_binary(
        self,
        data: bytes,
        mime_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_request(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        # Static assets
        if method in ("GET", "HEAD") and not path.startswith("/api/"):
            self._serve_static(path)
            return

        # Health endpoint (no AI or peer info)
        if method in ("GET", "HEAD") and path == "/api/health":
            with open(self.app.config.release_metadata_json, "r", encoding="utf-8") as f:
                rel_meta = json.load(f)
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "dashboard_version": DASHBOARD_VERSION,
                    "release_id": rel_meta.get("release_id"),
                    "release_human_verification_status": rel_meta.get("human_verification_status"),
                    "fabricated_labels_present": rel_meta.get("fabricated_labels_present"),
                    "workqueue_count": len(self.app.catalog.raw_items),
                    "pairs_count": len(self.app.catalog.pair_ids),
                    "blindness_mode": "STRICT_ALLOWLIST_NO_AI_NO_PEER_LEAK",
                    "video_diagnostics": self.app.video.diagnostics(),
                },
            )
            return

        # Session discovery / switching
        if method == "GET" and path == "/api/session":
            actor = self._resolve_actor()
            self._send_json(
                HTTPStatus.OK,
                {
                    "actor": actor,
                    "actor_label": REVIEWER_LABELS.get(actor) if actor else None,
                    "valid_actors": [
                        {"id": a, "label": REVIEWER_LABELS[a], "is_adjudicator": a == "adjudicator"}
                        for a in VALID_ACTORS
                    ],
                    "local_research_mode": True,
                    "security_note": "Local workstation role selector for two independent annotators and an adjudicator; not cryptographic multi-user authentication.",
                },
            )
            return

        if method == "POST" and path == "/api/session":
            body = self._read_json_body()
            actor = body.get("actor")
            if actor in (None, ""):
                self._send_json(
                    HTTPStatus.OK,
                    {"actor": None, "actor_label": None},
                    extra_headers={"Set-Cookie": "gt_review_actor=; Path=/; Max-Age=0; SameSite=Lax"},
                )
                return
            actor = str(actor).strip()
            if actor not in VALID_ACTORS:
                self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid actor: {actor!r}")
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "actor": actor,
                    "actor_label": REVIEWER_LABELS[actor],
                    "local_research_mode": True,
                },
                extra_headers={"Set-Cookie": f"gt_review_actor={actor}; Path=/; SameSite=Lax"},
            )
            return

        if method == "GET" and path == "/api/meta/enums":
            self._send_json(
                HTTPStatus.OK,
                {
                    "cross_camera_link_statuses": list(CROSS_CAMERA_LINK_STATUSES),
                    "rider_roles": list(RIDER_ROLES),
                    "helmet_states": list(HELMET_STATES),
                    "helmet_to_v1_1_map": HELMET_TO_V1_1_LABEL,
                    "plate_visibility_tiers": list(PLATE_VISIBILITY_TIERS),
                    "violation_event_types": list(VIOLATION_EVENT_TYPES),
                    "camera_evidence_choices": list(CAMERA_EVIDENCE_CHOICES),
                    "uncertainty_reason_tags": list(UNCERTAINTY_REASON_TAGS),
                },
            )
            return

        # ------------------------------------------------------------------
        # Blind Reviewer APIs (/api/reviewer/*)
        # ------------------------------------------------------------------
        if method == "GET" and path == "/api/reviewer/queue":
            actor = self._require_reviewer(query)
            status_map = self.app.storage.get_reviewer_status_map(actor)
            pair_filter = (query.get("pair_id") or [""])[0].strip()
            status_filter = (query.get("status") or [""])[0].strip().upper()
            scope_filter = (query.get("camera_scope") or [""])[0].strip().upper()
            bucket_filter = (query.get("bucket") or [""])[0].strip().upper()
            search = (query.get("q") or [""])[0].strip().lower()

            all_sanitized = [
                sanitize_item_for_reviewer(raw, my_record=status_map.get(raw["workqueue_id"]))
                for raw in self.app.catalog.raw_items
            ]
            counts = {
                "total": len(all_sanitized),
                "not_started": sum(1 for x in all_sanitized if x["my_status"] == "NOT_STARTED"),
                "draft": sum(1 for x in all_sanitized if x["my_status"] == "DRAFT"),
                "submitted": sum(1 for x in all_sanitized if x["my_status"] == "SUBMITTED"),
            }
            filtered = []
            for item in all_sanitized:
                if pair_filter and item["pair_id"] != pair_filter:
                    continue
                if status_filter and status_filter != "ALL" and item["my_status"] != status_filter:
                    continue
                if scope_filter and scope_filter != "ALL" and item["camera_scope"] != scope_filter:
                    continue
                if bucket_filter and bucket_filter != "ALL" and item["sampling_bucket"] != bucket_filter:
                    continue
                if search and search not in item["workqueue_id"].lower() and search not in item["pair_id"].lower():
                    continue
                filtered.append(item)

            # Determine resume candidate (most recently updated DRAFT, else first NOT_STARTED)
            drafts = [x for x in all_sanitized if x["my_status"] == "DRAFT"]
            drafts.sort(key=lambda x: x["my_updated_at"] or "", reverse=True)
            resume_item = drafts[0]["workqueue_id"] if drafts else next(
                (x["workqueue_id"] for x in all_sanitized if x["my_status"] == "NOT_STARTED"),
                all_sanitized[0]["workqueue_id"] if all_sanitized else None,
            )

            self._send_json(
                HTTPStatus.OK,
                {
                    "reviewer_id": actor,
                    "counts": counts,
                    "resume_workqueue_id": resume_item,
                    "items": filtered,
                },
            )
            return

        if method == "GET" and path == "/api/reviewer/pairs":
            actor = self._require_reviewer(query)
            status_map = self.app.storage.get_reviewer_status_map(actor)
            by_pair: dict[str, list[dict[str, Any]]] = {pid: [] for pid in self.app.catalog.pair_ids}
            for raw in self.app.catalog.raw_items:
                by_pair.setdefault(raw["pair_id"], []).append(raw)

            pairs_out = []
            for pid in self.app.catalog.pair_ids:
                timing = self.app.catalog.get_pair_timing(pid).to_dict()
                pair_items = by_pair.get(pid, [])
                statuses = [
                    (status_map.get(it["workqueue_id"]) or {}).get("status", "NOT_STARTED")
                    for it in pair_items
                ]
                timing["items_total"] = len(pair_items)
                timing["my_not_started"] = sum(1 for s in statuses if s == "NOT_STARTED")
                timing["my_draft"] = sum(1 for s in statuses if s == "DRAFT")
                timing["my_submitted"] = sum(1 for s in statuses if s == "SUBMITTED")
                timing["workqueue_ids"] = [it["workqueue_id"] for it in pair_items]
                pairs_out.append(timing)

            self._send_json(HTTPStatus.OK, {"reviewer_id": actor, "pairs": pairs_out})
            return

        if path.startswith("/api/reviewer/items/"):
            sub = path[len("/api/reviewer/items/") :].strip("/")
            parts = sub.split("/") if sub else []
            if not parts or len(parts) > 2:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Invalid reviewer item route: {path}")
                return
            workqueue_id = parts[0]
            action = parts[1] if len(parts) == 2 else ""
            actor = self._require_reviewer(query)
            raw_item = self.app.catalog.get_raw_item(workqueue_id)

            if method == "GET" and not action:
                my_record = self.app.storage.get_reviewer_item_record(workqueue_id, actor)
                timing = self.app.catalog.get_pair_timing(raw_item["pair_id"]).to_dict()
                idx_in_queue = next(
                    (i for i, r in enumerate(self.app.catalog.raw_items) if r["workqueue_id"] == workqueue_id),
                    0,
                )
                prev_id = self.app.catalog.raw_items[idx_in_queue - 1]["workqueue_id"] if idx_in_queue > 0 else None
                next_id = (
                    self.app.catalog.raw_items[idx_in_queue + 1]["workqueue_id"]
                    if idx_in_queue + 1 < len(self.app.catalog.raw_items)
                    else None
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "reviewer_id": actor,
                        "item": sanitize_item_for_reviewer(raw_item, my_record=my_record),
                        "pair_timing": timing,
                        "annotation": my_record,
                        "prev_workqueue_id": prev_id,
                        "next_workqueue_id": next_id,
                    },
                )
                return

            if method == "PUT" and action == "draft":
                body = self._read_json_body()
                if "reviewer_id" in body and body["reviewer_id"] != actor:
                    raise BlindPermissionError("Cannot save draft for another reviewer.")
                record = self.app.storage.save_draft(workqueue_id, actor, body)
                sanitized = sanitize_item_for_reviewer(raw_item, my_record=record)
                self._send_json(HTTPStatus.OK, {"item": sanitized, "annotation": record})
                return

            if method == "POST" and action == "submit":
                body = self._read_json_body()
                if "reviewer_id" in body and body["reviewer_id"] != actor:
                    raise BlindPermissionError("Cannot submit for another reviewer.")
                record = self.app.storage.submit_annotation(workqueue_id, actor, body)
                sanitized = sanitize_item_for_reviewer(raw_item, my_record=record)
                self._send_json(HTTPStatus.OK, {"item": sanitized, "annotation": record})
                return

        if method == "GET" and path == "/api/reviewer/history":
            actor = self._require_reviewer(query)
            bundle = self.app.storage.get_reviewer_history_bundle(actor)
            self._send_json(HTTPStatus.OK, {"reviewer_id": actor, **bundle})
            return

        if method == "GET" and path == "/api/reviewer/export":
            actor = self._require_reviewer(query)
            export_bundle = self.app.storage.export_single_reviewer_annotations(actor)
            headers = None
            if (query.get("download") or ["0"])[0] == "1":
                headers = {"Content-Disposition": f'attachment; filename="gt_v1_1_{actor}_annotations.json"'}
            self._send_json(HTTPStatus.OK, export_bundle, extra_headers=headers)
            return

        if method == "POST" and path == "/api/reviewer/snapshots":
            actor = self._require_reviewer(query)
            body = self._read_json_body()
            workqueue_id = str(body.get("workqueue_id") or "").strip()
            camera = str(body.get("camera") or "back").strip()
            raw_item = self.app.catalog.get_raw_item(workqueue_id)
            t_ref = body.get("t_ref_sec")
            frame_idx = body.get("frame_index")
            ext = str(body.get("ext") or "jpg").strip().lower()
            frame_data = self.app.video.extract_frame_data(
                raw_item["pair_id"],
                camera,
                t_ref_sec=float(t_ref) if t_ref is not None else None,
                frame_index=int(frame_idx) if frame_idx is not None else None,
                fmt=ext,
            )
            snap = self.app.storage.save_reviewer_snapshot_image(
                reviewer_id=actor,
                workqueue_id=workqueue_id,
                camera=camera,
                frame_index=frame_data["frame_index"],
                pts_norm_sec=frame_data["pts_norm_sec"],
                t_ref_sec=frame_data["t_ref_sec"],
                image_bytes=frame_data["bytes"],
                ext=frame_data["ext"],
            )
            self._send_json(HTTPStatus.OK, {"snapshot": snap})
            return

        if method in ("GET", "HEAD") and path.startswith("/api/reviewer/snapshots/"):
            actor = self._require_any_actor()
            rest = path[len("/api/reviewer/snapshots/") :].strip("/")
            parts = rest.split("/", 1)
            if len(parts) != 2:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Malformed snapshot path.")
                return
            owner, filename = parts
            if owner not in ("reviewer_a", "reviewer_b"):
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid snapshot owner.")
                return
            if actor in ("reviewer_a", "reviewer_b"):
                if actor != owner:
                    raise BlindPermissionError("Cannot access another reviewer's frame snapshots.")
            elif actor == "adjudicator":
                # Verify item prefix has both A & B submitted before adjudicator views reviewer snapshot
                parts_fn = filename.split("_")
                wid = f"{parts_fn[0]}_{parts_fn[1]}" if filename.startswith("WQ_") and len(parts_fn) >= 2 else ""
                if wid in self.app.catalog.items_by_id:
                    rec_a = self.app.storage.get_reviewer_item_record(wid, "reviewer_a")
                    rec_b = self.app.storage.get_reviewer_item_record(wid, "reviewer_b")
                    if not (rec_a and rec_a["status"] == "SUBMITTED" and rec_b and rec_b["status"] == "SUBMITTED"):
                        raise ConflictError("Adjudicator cannot view reviewer snapshots before both submit.")

            snap_root = (self.app.config.snapshots_dir / owner).resolve()
            candidate = (snap_root / filename).resolve()
            if snap_root not in candidate.parents or not candidate.exists():
                self._send_error_json(HTTPStatus.NOT_FOUND, "Snapshot file not found.")
                return
            mime = "image/png" if candidate.suffix.lower() == ".png" else "image/jpeg"
            self._serve_binary(candidate.read_bytes(), mime)
            return

        # ------------------------------------------------------------------
        # Bounded Dual-Camera Video & Frame Endpoints (Blind-neutral)
        # ------------------------------------------------------------------
        if method in ("GET", "HEAD") and path.startswith("/api/video/"):
            _ = self._require_any_actor()
            rest = path[len("/api/video/") :].strip("/")
            parts = rest.split("/")
            if len(parts) != 2:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Expected /api/video/<pair_id>/<camera>")
                return
            self._serve_video(parts[0], parts[1], query)
            return

        if method in ("GET", "HEAD") and path.startswith("/api/frame/"):
            _ = self._require_any_actor()
            rest = path[len("/api/frame/") :].strip("/")
            parts = rest.split("/")
            if len(parts) != 2:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Expected /api/frame/<pair_id>/<camera>")
                return
            pair_id, camera = parts
            t_ref_q = (query.get("t_ref") or [None])[0]
            t_norm_q = (query.get("t_norm") or [None])[0]
            frame_idx_q = (query.get("frame_idx") or [None])[0]
            fmt = (query.get("fmt") or ["jpg"])[0]
            max_w_q = (query.get("max_width") or [None])[0]
            crop_q = (query.get("crop") or [None])[0]
            crop = None
            if crop_q:
                nums = [int(x) for x in crop_q.split(",")]
                if len(nums) != 4:
                    raise ValueError(f"crop must have 4 comma-separated integers (x,y,w,h), got {crop_q!r}")
                crop = (nums[0], nums[1], nums[2], nums[3])

            res = self.app.video.extract_frame_data(
                pair_id,
                camera,
                t_ref_sec=float(t_ref_q) if t_ref_q is not None else None,
                t_norm_sec=float(t_norm_q) if t_norm_q is not None else None,
                frame_index=int(frame_idx_q) if frame_idx_q is not None else None,
                fmt=fmt,
                max_width=int(max_w_q) if max_w_q is not None else None,
                crop=crop,
            )
            self._serve_binary(
                res["bytes"],
                res["mime_type"],
                extra_headers={
                    "X-Frame-Index": str(res["frame_index"]),
                    "X-Pts-Norm-Sec": str(res["pts_norm_sec"]),
                    "X-T-Ref-Sec": str(res["t_ref_sec"]),
                    "X-Delta-Pts-Ms": str(res["delta_pts_ms"]),
                    "X-Frame-Width": str(res["width"]),
                    "X-Frame-Height": str(res["height"]),
                },
            )
            return

        if method == "GET" and path.startswith("/api/frame_info/"):
            _ = self._require_any_actor()
            pair_id = path[len("/api/frame_info/") :].strip("/")
            if not pair_id or "/" in pair_id:
                raise ValueError(f"Invalid pair_id: {pair_id!r}")
            t_ref = float((query.get("t_ref") or ["0.0"])[0])
            if not math.isfinite(t_ref):
                raise ValueError(f"t_ref must be finite, got {t_ref!r}")
            step_q = (query.get("step_frames") or [None])[0]
            anchor = (query.get("anchor") or ["back"])[0]
            if anchor not in ("front", "back"):
                raise ValueError(f"Invalid anchor camera: {anchor!r}")
            if step_q is not None and int(step_q) != 0:
                t_ref = self.app.video.step_reference_time(pair_id, t_ref, int(step_q), anchor_camera=anchor)
            info = self.app.video.lookup_sync_state(pair_id, t_ref)
            self._send_json(HTTPStatus.OK, info)
            return

        if method == "POST" and path == "/api/video/release":
            _ = self._require_any_actor()
            released = self.app.video.release_active_captures()
            self._send_json(HTTPStatus.OK, {"released": released, "diagnostics": self.app.video.diagnostics()})
            return

        # ------------------------------------------------------------------
        # Coordinator / Adjudication Routes (FORBIDDEN for reviewer_a / reviewer_b)
        # ------------------------------------------------------------------
        if method == "GET" and path == "/api/adjudication/overview":
            _ = self._require_adjudicator()
            self._send_json(HTTPStatus.OK, self.app.storage.get_adjudication_overview())
            return

        if path.startswith("/api/adjudication/items/"):
            workqueue_id = path[len("/api/adjudication/items/") :].strip("/")
            actor = self._require_adjudicator()
            if method == "GET":
                detail = self.app.storage.get_adjudication_item(workqueue_id)
                timing = self.app.catalog.get_pair_timing(detail["item"]["pair_id"]).to_dict()
                self._send_json(HTTPStatus.OK, {**detail, "pair_timing": timing})
                return
            if method == "POST":
                body = self._read_json_body()
                updated = self.app.storage.save_adjudication(workqueue_id, body, adjudicator_id=actor)
                self._send_json(HTTPStatus.OK, updated)
                return

        if method == "GET" and path == "/api/adjudication/export":
            _ = self._require_adjudicator()
            manifest = self.app.storage.export_full_v1_1_bundle()
            self._send_json(HTTPStatus.OK, manifest)
            return

        if method == "POST" and path == "/api/system/backup":
            _ = self._require_any_actor()
            body = self._read_json_body()
            label = str(body.get("label") or "manual")
            dest = self.app.storage.create_backup(label=label)
            self._send_json(
                HTTPStatus.OK,
                {
                    "backup_filename": dest.name,
                    "backup_path": str(dest),
                },
            )
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {method} {path}")

    def _safe_dispatch(self, method: str) -> None:
        try:
            self._handle_request(method)
        except BlindPermissionError as exc:
            self._send_error_json(HTTPStatus.FORBIDDEN, str(exc))
        except PermissionError as exc:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, str(exc))
        except ConflictError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, str(exc))
        except ValidationFailedError as exc:
            self._send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                str(exc),
                details={"errors": exc.errors, "warnings": exc.warnings},
            )
        except KeyError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"Internal server error: {exc}")

    def do_GET(self) -> None:
        self._safe_dispatch("GET")

    def do_HEAD(self) -> None:
        self._safe_dispatch("HEAD")

    def do_POST(self) -> None:
        self._safe_dispatch("POST")

    def do_PUT(self) -> None:
        self._safe_dispatch("PUT")


class ReviewDashboardHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying bounded app context and clean shutdown."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app_context: ReviewAppContext,
        *,
        quiet_logs: bool = True,
    ) -> None:
        self.app_context = app_context
        self.quiet_logs = quiet_logs
        super().__init__(server_address, _ReviewRequestHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Suppress expected client socket aborts when browsers seek/switch HTTP Range video streams
        import sys

        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        try:
            self.app_context.close()
        finally:
            super().server_close()


def create_server(
    config: DashboardConfig | None = None,
    *,
    quiet_logs: bool = True,
) -> ReviewDashboardHTTPServer:
    """Instantiate ReviewAppContext and bound ReviewDashboardHTTPServer."""
    cfg = config or DashboardConfig.from_repo()
    ctx = ReviewAppContext(cfg)
    return ReviewDashboardHTTPServer((cfg.host, cfg.port), ctx, quiet_logs=quiet_logs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Blind Dual-Camera Human GT v1.1 Review Dashboard Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Override review_workspace directory (default: data/.../dual_camera_gt_v1.1_blind_human/review_workspace)",
    )
    parser.add_argument("--db-path", type=Path, default=None, help="Override SQLite database path")
    parser.add_argument("--no-proxy", action="store_true", help="Disable on-demand H.264 browser fallback proxies")
    parser.add_argument("--verbose-http", action="store_true", help="Log HTTP requests to stderr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = DashboardConfig.from_repo(
        workspace_dir=args.workspace_dir,
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        allow_proxy_transcode=not args.no_proxy,
    )
    server = create_server(cfg, quiet_logs=not args.verbose_http)
    actual_host, actual_port = server.server_address[:2]
    print("=" * 72, flush=True)
    print("  Blind Dual-Camera Human GT v1.1 Review Dashboard", flush=True)
    print(f"  URL       : http://{actual_host}:{actual_port}/", flush=True)
    print(f"  Workspace : {cfg.workspace_dir}", flush=True)
    print(f"  Database  : {cfg.db_path}", flush=True)
    print("  Blindness : Strict reviewer A/B isolation + zero AI/v1.0 exposure", flush=True)
    print("  Press Ctrl+C to stop cleanly.", flush=True)
    print("=" * 72, flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nShutting down review dashboard...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
