"""End-to-end tests for the blind human GT v1.1 dual-camera review dashboard."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

import cv2
import httpx
import jsonschema
import numpy as np
import pytest

from src.review_dashboard.config import DashboardConfig
from src.review_dashboard.models import t_front_from_ref, t_ref_from_front
from src.review_dashboard.server import create_server


FORBIDDEN_BLIND_KEYS = {
    "ai_prediction",
    "yolo",
    "resnet",
    "ocr",
    "association_score",
    "track_id",
    "review_focus",
    "source_origin",
    "raw_priority_stratum",
    "annotator_a_link_status",
    "annotator_b_link_status",
    "adjudicator_final_event_type",
    "disagreements",
    "cohens_kappa",
    "v1_0",
    "dual_camera_gt_v1.0",
}


def _collect_keys_and_strings(obj: Any, keys_out: set[str], strings_out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys_out.add(str(k).lower())
            _collect_keys_and_strings(v, keys_out, strings_out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys_and_strings(item, keys_out, strings_out)
    elif isinstance(obj, str):
        strings_out.append(obj.lower())


def _sample_payload(
    *,
    link_status: str = "LINKED_BOTH_VIEWS",
    helmet_driver: str = "NO_HELMET",
    event_type: str = "DRIVER_NO_HELMET",
    start_t: float = 0.5,
    end_t: float = 1.2,
    plate_tier: str = "PLATE_UNREADABLE_MOTION_BLUR",
    plate_text: str = "",
) -> dict[str, Any]:
    return {
        "item_link_status": link_status,
        "is_non_motorcycle_transit": False,
        "uncertainty": {"is_uncertain": False, "reasons": []},
        "reviewer_notes": "Clear frame observation.",
        "motorcycles": [
            {
                "moto_id": "M1",
                "cross_camera_link_status": link_status,
                "visible_front": link_status in ("LINKED_BOTH_VIEWS", "FRONT_VIEW_ONLY", "AMBIGUOUS_HUMAN_UNCERTAIN"),
                "visible_back": link_status in ("LINKED_BOTH_VIEWS", "BACK_VIEW_ONLY", "AMBIGUOUS_HUMAN_UNCERTAIN"),
                "start_t_ref_sec": start_t,
                "end_t_ref_sec": end_t,
                "rear_plate": {
                    "visibility_tier": plate_tier,
                    "plate_text_normalized": plate_text,
                    "notes": "",
                },
            }
        ],
        "riders": [
            {
                "rider_id": "R1",
                "moto_id": "M1",
                "role": "DRIVER",
                "helmet_state": helmet_driver,
                "front_helmet_evidence": helmet_driver if link_status != "BACK_VIEW_ONLY" else "NOT_VISIBLE",
                "back_helmet_evidence": helmet_driver if link_status != "FRONT_VIEW_ONLY" else "NOT_VISIBLE",
            }
        ],
        "violation_events": [
            {
                "event_id": "E1",
                "moto_id": "M1",
                "rider_ids": ["R1"],
                "event_type": event_type,
                "start_t_ref_sec": start_t,
                "end_t_ref_sec": end_t,
                "camera_evidence": (
                    "FRONT_ONLY"
                    if link_status == "FRONT_VIEW_ONLY"
                    else ("BACK_ONLY" if link_status == "BACK_VIEW_ONLY" else "BOTH_CAMERAS")
                ),
            }
        ],
    }


@pytest.fixture()
def dashboard_server(tmp_path: Path):
    cfg = DashboardConfig.from_repo(
        workspace_dir=tmp_path / "workspace",
        db_path=tmp_path / "workspace" / "test_review.sqlite3",
        host="127.0.0.1",
        port=0,
        allow_proxy_transcode=False,
    )
    server = create_server(cfg, quiet_logs=True)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", cfg, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_role_separation_and_blindness_no_ai_or_peer_leakage(dashboard_server) -> None:
    base_url, _cfg, _srv = dashboard_server
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # Unauthenticated requests blocked
        assert client.get("/api/reviewer/queue").status_code == 401
        assert client.get("/api/adjudication/overview").status_code == 401

        # Save Reviewer A draft on WQ_001
        resp = client.put(
            "/api/reviewer/items/WQ_001/draft",
            headers={"X-Reviewer-Id": "reviewer_a"},
            json=_sample_payload(link_status="FRONT_VIEW_ONLY", start_t=-0.1, end_t=1.0),
        )
        assert resp.status_code == 200

        # Reviewer B cannot see Reviewer A's draft in queue, item detail, history, export, or cross param
        q_b = client.get("/api/reviewer/queue", headers={"X-Reviewer-Id": "reviewer_b"})
        assert q_b.status_code == 200
        q_b_json = q_b.json()
        assert q_b_json["counts"]["total"] == 92
        assert q_b_json["counts"]["draft"] == 0
        assert q_b_json["counts"]["submitted"] == 0

        item_b = client.get("/api/reviewer/items/WQ_001", headers={"X-Reviewer-Id": "reviewer_b"})
        assert item_b.status_code == 200
        item_b_json = item_b.json()
        assert item_b_json["annotation"] is None

        # Explicit peer spoofing blocked
        assert client.get(
            "/api/reviewer/queue?reviewer_id=reviewer_a",
            headers={"X-Reviewer-Id": "reviewer_b"},
        ).status_code == 403
        assert client.put(
            "/api/reviewer/items/WQ_001/draft",
            headers={"X-Reviewer-Id": "reviewer_b"},
            json={"reviewer_id": "reviewer_a"},
        ).status_code == 403

        # Reviewer cannot access adjudicator endpoints
        assert client.get("/api/adjudication/overview", headers={"X-Reviewer-Id": "reviewer_a"}).status_code == 403
        assert client.get("/api/adjudication/items/WQ_001", headers={"X-Reviewer-Id": "reviewer_b"}).status_code == 403
        assert client.get("/api/adjudication/export", headers={"X-Reviewer-Id": "reviewer_a"}).status_code == 403

        # Recursive leakage scan on reviewer responses
        for payload in (q_b_json, item_b_json, client.get("/api/reviewer/pairs", headers={"X-Reviewer-Id": "reviewer_a"}).json()):
            keys: set[str] = set()
            strings: list[str] = []
            _collect_keys_and_strings(payload, keys, strings)
            assert not (keys & FORBIDDEN_BLIND_KEYS), f"Leak keys found: {keys & FORBIDDEN_BLIND_KEYS}"
            for bad_hint in ("gapc_unlinked", "pillion_presence_and_helmet_blind_check", "dual_camera_gt_v1.0"):
                assert not any(bad_hint in s for s in strings), f"Leak hint {bad_hint} in reviewer strings"


def test_validation_rules_timestamps_point_events_and_relational_integrity(dashboard_server) -> None:
    base_url, _cfg, _srv = dashboard_server
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        hdr = {"X-Reviewer-Id": "reviewer_a"}

        # 1. Negative-offset boundary: WQ_001 has offset -0.205, so negative t_ref=-0.200 is valid,
        # while a 1-frame point event [1.100, 1.100] is also valid.
        ok_point = _sample_payload(link_status="FRONT_VIEW_ONLY", start_t=-0.15, end_t=-0.15)
        r_ok = client.post("/api/reviewer/items/WQ_001/submit", headers=hdr, json=ok_point)
        assert r_ok.status_code == 200
        assert r_ok.json()["annotation"]["payload"]["violation_events"][0]["is_point_event"] is True

        # 2. end < start rejected
        bad_time = _sample_payload(start_t=1.5, end_t=1.2)
        assert client.put("/api/reviewer/items/WQ_002/draft", headers=hdr, json=bad_time).status_code == 422

        # 3. Out-of-bounds timestamp rejected
        oob = _sample_payload(start_t=99.0, end_t=99.5)
        assert client.put("/api/reviewer/items/WQ_002/draft", headers=hdr, json=oob).status_code == 422

        # 4. Orphan rider/event reference rejected
        orphan = _sample_payload()
        orphan["riders"][0]["moto_id"] = "M999"
        assert client.put("/api/reviewer/items/WQ_002/draft", headers=hdr, json=orphan).status_code == 422

        # 5. Consistency check: NO_HELMET rider vs NO_VIOLATION event blocked on submit
        mismatch = _sample_payload(helmet_driver="NO_HELMET", event_type="NO_VIOLATION")
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=mismatch).status_code == 422

        # 6. PLATE_CLEAR without plate text blocked on submit
        clear_no_text = _sample_payload(plate_tier="PLATE_CLEAR", plate_text="")
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=clear_no_text).status_code == 422

        # 7. Non-motorcycle transit valid with empty motorcycles + explanatory note
        non_moto = {
            "item_link_status": "NOT_A_MOTORCYCLE_TRANSIT",
            "is_non_motorcycle_transit": True,
            "reviewer_notes": "Confirmed pedal bicycle without motor.",
            "motorcycles": [],
            "riders": [],
            "violation_events": [],
        }
        r_nm = client.post("/api/reviewer/items/WQ_071/submit", headers=hdr, json=non_moto)
        assert r_nm.status_code == 200
        assert r_nm.json()["annotation"]["summary"]["link_status"] == "NOT_A_MOTORCYCLE_TRANSIT"


def test_draft_autosave_reload_recovery_submission_lock_and_no_duplicates(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    db = ws / "persistent.sqlite3"
    cfg1 = DashboardConfig.from_repo(workspace_dir=ws, db_path=db, host="127.0.0.1", port=0, allow_proxy_transcode=False)
    srv1 = create_server(cfg1, quiet_logs=True)
    port1 = srv1.server_address[1]
    t1 = threading.Thread(target=srv1.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t1.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port1}", timeout=10.0) as client:
            r1 = client.put(
                "/api/reviewer/items/WQ_004/draft",
                headers={"X-Reviewer-Id": "reviewer_a"},
                json=_sample_payload(helmet_driver="HELMET", event_type="NO_VIOLATION"),
            )
            assert r1.status_code == 200
            assert r1.json()["annotation"]["version"] == 1
    finally:
        srv1.shutdown()
        srv1.server_close()
        t1.join(timeout=3)

    # Restart server pointing to the same SQLite DB (simulates full page/server reload)
    cfg2 = DashboardConfig.from_repo(workspace_dir=ws, db_path=db, host="127.0.0.1", port=0, allow_proxy_transcode=False)
    srv2 = create_server(cfg2, quiet_logs=True)
    port2 = srv2.server_address[1]
    t2 = threading.Thread(target=srv2.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t2.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port2}", timeout=10.0) as client:
            rec = client.get("/api/reviewer/items/WQ_004", headers={"X-Reviewer-Id": "reviewer_a"}).json()
            assert rec["annotation"]["status"] == "DRAFT"
            assert rec["annotation"]["version"] == 1

            sub = client.post(
                "/api/reviewer/items/WQ_004/submit",
                headers={"X-Reviewer-Id": "reviewer_a"},
                json=rec["annotation"]["payload"],
            )
            assert sub.status_code == 200
            assert sub.json()["annotation"]["status"] == "SUBMITTED"
            assert (ws / "submissions" / "reviewer_a" / "WQ_004.json").exists()

            # Locked against edits and duplicate submissions
            assert client.put(
                "/api/reviewer/items/WQ_004/draft",
                headers={"X-Reviewer-Id": "reviewer_a"},
                json=rec["annotation"]["payload"],
            ).status_code == 409
            assert client.post(
                "/api/reviewer/items/WQ_004/submit",
                headers={"X-Reviewer-Id": "reviewer_a"},
                json=rec["annotation"]["payload"],
            ).status_code == 409
    finally:
        srv2.shutdown()
        srv2.server_close()
        t2.join(timeout=3)


def test_adjudication_unlock_kappa_export_and_backup(dashboard_server) -> None:
    base_url, cfg, _srv = dashboard_server
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # Reviewer A submits WQ_002; Reviewer B has NOT submitted yet
        assert client.post(
            "/api/reviewer/items/WQ_002/submit",
            headers={"X-Reviewer-Id": "reviewer_a"},
            json=_sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET", start_t=4.0, end_t=5.2),
        ).status_code == 200

        # Adjudicator blocked from opening WQ_002 until Reviewer B also submits
        assert client.get("/api/adjudication/items/WQ_002", headers={"X-Reviewer-Id": "adjudicator"}).status_code == 409

        # Reviewer B submits a disagreeing verdict on WQ_002
        assert client.post(
            "/api/reviewer/items/WQ_002/submit",
            headers={"X-Reviewer-Id": "reviewer_b"},
            json=_sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="NO_VIOLATION", start_t=4.0, end_t=5.2),
        ).status_code == 200

        # Both submit agreeing on WQ_003
        agree_pay = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="NO_VIOLATION", start_t=4.0, end_t=4.8, plate_tier="PLATE_CLEAR", plate_text="dha 1234")
        assert client.post("/api/reviewer/items/WQ_003/submit", headers={"X-Reviewer-Id": "reviewer_a"}, json=agree_pay).status_code == 200
        assert client.post("/api/reviewer/items/WQ_003/submit", headers={"X-Reviewer-Id": "reviewer_b"}, json=agree_pay).status_code == 200

        ov = client.get("/api/adjudication/overview", headers={"X-Reviewer-Id": "adjudicator"}).json()
        assert ov["counts"]["ready_disagreed"] == 1
        assert ov["counts"]["ready_agreed"] == 1
        assert ov["inter_annotator_agreement"]["binary_violation"]["n"] == 2

        det = client.get("/api/adjudication/items/WQ_002", headers={"X-Reviewer-Id": "adjudicator"}).json()
        assert len(det["disagreements"]) >= 2

        adj_res = client.post(
            "/api/adjudication/items/WQ_002",
            headers={"X-Reviewer-Id": "adjudicator"},
            json={
                "status": "ADJUDICATED",
                "final_link_status": "BACK_VIEW_ONLY",
                "final_rider_count": 1,
                "final_driver_helmet": "no_helmet",
                "final_pillion_helmet": "",
                "final_event_type": "DRIVER_NO_HELMET",
                "final_plate_visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR",
                "adjudicator_notes": "Verified frame 250 shows bare head cleanly.",
            },
        )
        assert adj_res.status_code == 200
        assert adj_res.json()["adjudication"]["status"] == "ADJUDICATED"

        # Export bundle + schema validation
        exp = client.get("/api/adjudication/export", headers={"X-Reviewer-Id": "adjudicator"}).json()
        assert exp["executed_human_units"] == 2
        assert exp["human_verification_status"] == "IN_PROGRESS"

        schema_doc = json.loads(cfg.release_schema_json.read_text(encoding="utf-8"))
        snap_meta = json.loads((cfg.exports_dir / "release_metadata_snapshot.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=snap_meta, schema=schema_doc)
        assert snap_meta["fabricated_labels_present"] is False

        with open(cfg.exports_dir / "blind_annotation_workqueue_populated.csv", "r", encoding="utf-8", newline="") as f:
            pop_rows = {r["workqueue_id"]: r for r in csv.DictReader(f)}
        assert pop_rows["WQ_002"]["review_status"] == "ADJUDICATED_COMPLETE"
        assert pop_rows["WQ_003"]["review_status"] == "AB_AGREED"
        assert pop_rows["WQ_003"]["annotator_a_plate_text_normalized"] == "DHA1234"

        # SQLite backup check
        bk = client.post("/api/system/backup", headers={"X-Reviewer-Id": "adjudicator"}, json={"label": "pytest"}).json()
        bk_path = Path(bk["backup_path"])
        assert bk_path.exists()
        conn = sqlite3.connect(str(bk_path))
        cnt = conn.execute("SELECT COUNT(*) FROM reviewer_annotations").fetchone()[0]
        conn.close()
        assert cnt == 4


def test_video_range_streaming_sync_math_and_bounded_cleanup(dashboard_server) -> None:
    base_url, cfg, srv = dashboard_server
    with httpx.Client(base_url=base_url, timeout=15.0) as client:
        hdr = {"X-Reviewer-Id": "reviewer_a"}

        # Range requests on Front H.264 and Back HEVC MP4s
        for cam in ("front", "back"):
            r = client.get(f"/api/video/pair_001/{cam}", headers={**hdr, "Range": "bytes=0-16383"})
            assert r.status_code == 206
            assert r.headers.get("Accept-Ranges") == "bytes"
            assert r.headers.get("Content-Range", "").startswith("bytes 0-16383/")
            assert len(r.content) == 16384
            assert b"ftyp" in r.content[:64]

        assert client.get("/api/video/pair_001/front", headers={**hdr, "Range": "bytes=999999999-"}).status_code == 416

        # Sync math & PTS-exact stepping
        info = client.get("/api/frame_info/pair_001?t_ref=1.0", headers=hdr).json()
        expected_f = t_front_from_ref(1.0, info["global_offset_sec"])
        assert abs(info["front"]["target_norm_sec"] - expected_f) < 1e-6
        assert abs(t_ref_from_front(info["front"]["actual_pts_norm_sec"], info["global_offset_sec"]) - info["front"]["actual_t_ref_sec"]) < 1e-5

        stepped = client.get("/api/frame_info/pair_001?t_ref=1.0&step_frames=1&anchor=back", headers=hdr).json()
        assert stepped["back"]["frame_index"] == info["back"]["frame_index"] + 1

        # Bounded frame extraction across pair switch releases prior captures
        f1 = client.get("/api/frame/pair_001/back?t_ref=1.0&max_width=640", headers=hdr)
        assert f1.status_code == 200
        assert f1.headers["Content-Type"] == "image/jpeg"
        assert int(f1.headers["X-Frame-Width"]) == 640

        f2 = client.get("/api/frame/pair_004/front?t_ref=0.5&max_width=640", headers=hdr)
        assert f2.status_code == 200

        diag = srv.app_context.video.diagnostics()
        assert diag["active_pair_id"] == "pair_004"
        assert diag["open_captures"] <= 2
        assert diag["total_captures_released"] >= 1

        # Snapshot capture saves solely under reviewer_a directory, forbidden to reviewer_b
        snap_res = client.post(
            "/api/reviewer/snapshots",
            headers=hdr,
            json={"workqueue_id": "WQ_001", "camera": "front", "t_ref_sec": 0.5, "ext": "jpg"},
        )
        assert snap_res.status_code == 200
        snap_url = snap_res.json()["snapshot"]["url"]
        assert client.get(snap_url, headers=hdr).status_code == 200
        assert client.get(snap_url, headers={"X-Reviewer-Id": "reviewer_b"}).status_code == 403

        # Explicit capture release
        rel = client.post("/api/video/release", headers=hdr).json()
        assert rel["diagnostics"]["open_captures"] == 0

    # VFR H.264 & B-frame HEVC frame-exact extraction regression check (no CAP_PROP_POS_FRAMES drift)
    from src.review_dashboard.blind_api import resolve_video_path_safe

    video_svc = srv.app_context.video
    seek_cases = [
        ("pair_001", "front", [103, 102]),
        ("pair_013", "back", [304, 607, 607]),
        ("pair_014", "back", [138, 275]),
    ]
    for pair_id, camera, target_indices in seek_cases:
        video_path = resolve_video_path_safe(cfg, srv.app_context.catalog, pair_id, camera)
        gt_frames: dict[int, np.ndarray] = {}
        needed = set(target_indices)
        max_idx = max(needed)
        gt_cap = cv2.VideoCapture(str(video_path))
        assert gt_cap.isOpened()
        try:
            for idx in range(max_idx + 1):
                assert gt_cap.grab(), f"GT sequential grab failed at {pair_id}/{camera}@{idx}"
                if idx in needed:
                    ok, gt_img = gt_cap.retrieve()
                    assert ok and gt_img is not None
                    gt_frames[idx] = gt_img
        finally:
            gt_cap.release()

        for target_idx in target_indices:
            res = video_svc.extract_frame_data(pair_id, camera, frame_index=target_idx, fmt="png")
            assert res["frame_index"] == target_idx
            assert res["width"] == 3840
            assert res["height"] == 2160
            decoded = cv2.imdecode(np.frombuffer(res["bytes"], np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None
            assert np.array_equal(decoded, gt_frames[target_idx]), (
                f"Exact frame mismatch on {pair_id}/{camera}@{target_idx}"
            )

    assert video_svc.diagnostics()["open_captures"] <= 2
    video_svc.release_active_captures()
    assert video_svc.diagnostics()["open_captures"] == 0

    # Verify checked-in root v1.1 manifest files were never modified or populated with synthetic labels
    root_meta = json.loads(cfg.release_metadata_json.read_text(encoding="utf-8"))
    assert root_meta["human_verification_status"] == "PENDING_HUMAN_REVIEW"
    assert root_meta["human_verified"] is False
    assert root_meta["executed_human_units"] == 0
    assert root_meta["fabricated_labels_present"] is False
    assert hashlib.md5(cfg.workqueue_csv.read_bytes()).hexdigest() == "82d9e890f64711b5c303a06e711ddf30"


def test_cookie_sessions_static_assets_history_unresolved_and_proxy_fallback(tmp_path: Path) -> None:
    cfg = DashboardConfig.from_repo(
        workspace_dir=tmp_path / "workspace",
        db_path=tmp_path / "workspace" / "cookie_proxy.sqlite3",
        host="127.0.0.1",
        port=0,
        allow_proxy_transcode=True,
    )
    srv = create_server(cfg, quiet_logs=True)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20.0) as client:
            # Static UI & schema metadata
            assert client.get("/").status_code == 200
            assert "Dual-Camera Helmet Violation Review" in client.get("/").text
            assert client.get("/styles.css").status_code == 200
            assert client.get("/app.js").status_code == 200
            assert "NOT_VISIBLE" in client.get("/api/meta/enums").json()["helmet_states"]

            # Cookie-based Reviewer A login -> save draft & submit WQ_001 -> check history/export
            assert client.post("/api/session", json={"actor": "reviewer_a"}).status_code == 200
            assert client.get("/api/session").json()["actor"] == "reviewer_a"
            pay_a = _sample_payload(link_status="FRONT_VIEW_ONLY", start_t=-0.1, end_t=0.8)
            assert client.post("/api/reviewer/items/WQ_001/submit", json=pay_a).status_code == 200
            hist_a = client.get("/api/reviewer/history").json()
            assert len(hist_a["records"]) == 1
            assert client.get("/api/reviewer/export").json()["submitted_count"] == 1

            # Switch cookie to Reviewer B -> must see zero records in own history
            assert client.post("/api/session", json={"actor": "reviewer_b"}).status_code == 200
            assert len(client.get("/api/reviewer/history").json()["records"]) == 0
            pay_b = _sample_payload(link_status="AMBIGUOUS_HUMAN_UNCERTAIN", helmet_driver="UNCERTAIN", event_type="INSUFFICIENT_VISUAL_EVIDENCE", start_t=0.1, end_t=0.9)
            assert client.post("/api/reviewer/items/WQ_001/submit", json=pay_b).status_code == 200

            # Switch cookie to Adjudicator -> mark UNRESOLVED
            assert client.post("/api/session", json={"actor": "adjudicator"}).status_code == 200
            unres = client.post(
                "/api/adjudication/items/WQ_001",
                json={
                    "status": "UNRESOLVED",
                    "adjudicator_notes": "Heavy glare obscured visor across overlapping window; escalated.",
                },
            )
            assert unres.status_code == 200
            assert unres.json()["adjudication"]["status"] == "UNRESOLVED"
            assert client.get("/api/adjudication/overview").json()["counts"]["unresolved"] == 1

            # Path traversal guard
            assert client.get("/api/video/..%2F..%2Fetc/front").status_code in (400, 403, 404)

            # Bounded 720p H.264 browser fallback proxy streaming
            prx = client.get("/api/video/pair_001/back?mode=proxy_h264", headers={"Range": "bytes=0-16383"})
            assert prx.status_code == 206
            assert b"ftyp" in prx.content[:64]
            assert (cfg.proxy_cache_dir / "pair_001_back_h264_720p.mp4").exists()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=3)


def test_audit_pass1_strict_validation_and_multi_entity_summary(dashboard_server) -> None:
    base_url, _cfg, _srv = dashboard_server
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        hdr = {"X-Reviewer-Id": "reviewer_a"}

        # 1. Draft allows blank incomplete selections without fabricating defaults; submit rejects them
        blank_draft = _sample_payload()
        blank_draft["motorcycles"][0]["rear_plate"]["visibility_tier"] = ""
        blank_draft["riders"][0]["role"] = ""
        blank_draft["riders"][0]["helmet_state"] = ""
        blank_draft["violation_events"][0]["event_type"] = ""
        blank_draft["violation_events"][0]["camera_evidence"] = ""
        r_draft = client.put("/api/reviewer/items/WQ_001/draft", headers=hdr, json=blank_draft)
        assert r_draft.status_code == 200
        saved_pay = r_draft.json()["annotation"]["payload"]
        assert saved_pay["riders"][0]["helmet_state"] == ""
        assert saved_pay["violation_events"][0]["event_type"] == ""
        assert client.post("/api/reviewer/items/WQ_001/submit", headers=hdr, json=blank_draft).status_code == 422

        # 2. Cross-field contradictions blocked on submit
        front_bad_plate = _sample_payload(link_status="FRONT_VIEW_ONLY", plate_tier="PLATE_CLEAR", plate_text="DHA1", start_t=0.2, end_t=0.8)
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=front_bad_plate).status_code == 422

        front_bad_cam = _sample_payload(link_status="FRONT_VIEW_ONLY", start_t=0.2, end_t=0.8)
        front_bad_cam["violation_events"][0]["camera_evidence"] = "BOTH_CAMERAS"
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=front_bad_cam).status_code == 422

        both_one_rider = _sample_payload(helmet_driver="NO_HELMET", event_type="BOTH_NO_HELMET", start_t=0.2, end_t=0.8)
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=both_one_rider).status_code == 422

        insuff_nh = _sample_payload(helmet_driver="NO_HELMET", event_type="INSUFFICIENT_VISUAL_EVIDENCE", start_t=0.2, end_t=0.8)
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=insuff_nh).status_code == 422

        invisible_ambig = _sample_payload(link_status="AMBIGUOUS_HUMAN_UNCERTAIN", start_t=0.2, end_t=0.8)
        invisible_ambig["motorcycles"][0]["visible_front"] = False
        invisible_ambig["motorcycles"][0]["visible_back"] = False
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=invisible_ambig).status_code == 422

        bad_keyframe = _sample_payload(start_t=1.0, end_t=1.5)
        bad_keyframe["violation_events"][0]["keyframe_ref_t_sec"] = 3.5
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=bad_keyframe).status_code == 422

        # Non-finite NaN timestamp blocked (400 due to strict JSON / float check)
        raw_nan_body = '{"item_link_status": "LINKED_BOTH_VIEWS", "motorcycles": [{"moto_id": "M1", "start_t_ref_sec": NaN}]}'
        r_nan = client.put(
            "/api/reviewer/items/WQ_002/draft",
            headers={**hdr, "Content-Type": "application/json"},
            content=raw_nan_body,
        )
        assert r_nan.status_code in (400, 422)

        # 3. Multi-pillion worst-case aggregation & multi-motorcycle alignment without cross-contamination
        multi_payload = {
            "item_link_status": "LINKED_BOTH_VIEWS",
            "is_non_motorcycle_transit": False,
            "reviewer_notes": "Two motorcycles in window; M2 driver unhelmeted, M1 carries 2 pillions (one unhelmeted).",
            "motorcycles": [
                {
                    "moto_id": "M1",
                    "cross_camera_link_status": "LINKED_BOTH_VIEWS",
                    "visible_front": True,
                    "visible_back": True,
                    "start_t_ref_sec": 0.5,
                    "end_t_ref_sec": 1.5,
                    "rear_plate": {"visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR", "plate_text_normalized": ""},
                },
                {
                    "moto_id": "M2",
                    "cross_camera_link_status": "BACK_VIEW_ONLY",
                    "visible_front": False,
                    "visible_back": True,
                    "start_t_ref_sec": 2.0,
                    "end_t_ref_sec": 3.0,
                    "rear_plate": {"visibility_tier": "PLATE_CLEAR", "plate_text_normalized": "DHA-9999"},
                },
            ],
            "riders": [
                {"rider_id": "R1", "moto_id": "M1", "role": "DRIVER", "helmet_state": "HELMET"},
                {"rider_id": "R2", "moto_id": "M1", "role": "PILLION", "helmet_state": "HELMET"},
                {"rider_id": "R3", "moto_id": "M1", "role": "PILLION", "helmet_state": "NO_HELMET"},
                {"rider_id": "R4", "moto_id": "M2", "role": "DRIVER", "helmet_state": "NO_HELMET"},
            ],
            "violation_events": [
                {
                    "event_id": "E1",
                    "moto_id": "M1",
                    "rider_ids": ["R3"],
                    "event_type": "PILLION_NO_HELMET",
                    "start_t_ref_sec": 0.6,
                    "end_t_ref_sec": 1.2,
                    "camera_evidence": "BOTH_CAMERAS",
                },
                {
                    "event_id": "E2",
                    "moto_id": "M2",
                    "rider_ids": ["R4"],
                    "event_type": "DRIVER_NO_HELMET",
                    "start_t_ref_sec": 2.1,
                    "end_t_ref_sec": 2.8,
                    "camera_evidence": "BACK_ONLY",
                },
            ],
        }
        r_multi = client.post("/api/reviewer/items/WQ_002/submit", headers=hdr, json=multi_payload)
        assert r_multi.status_code == 200
        s_multi = r_multi.json()["annotation"]["summary"]
        # Primary motorcycle M2 selected strictly (DRIVER_NO_HELMET outranks PILLION_NO_HELMET)
        assert s_multi["violation_event_type"] == "DRIVER_NO_HELMET"
        assert s_multi["driver_helmet"] == "no_helmet"
        assert s_multi["pillion_helmet"] == ""
        assert s_multi["plate_text_normalized"] == "DHA-9999"

        # Single-moto with 2 pillions [HELMET, NO_HELMET] must yield pillion_helmet == "no_helmet"
        two_pil = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="PILLION_NO_HELMET", start_t=0.5, end_t=1.2)
        two_pil["riders"].extend(
            [
                {"rider_id": "R2", "moto_id": "M1", "role": "PILLION", "helmet_state": "HELMET"},
                {"rider_id": "R3", "moto_id": "M1", "role": "PILLION", "helmet_state": "NO_HELMET"},
            ]
        )
        two_pil["violation_events"][0]["rider_ids"] = ["R3"]
        r_pil = client.post("/api/reviewer/items/WQ_003/submit", headers=hdr, json=two_pil)
        assert r_pil.status_code == 200
        assert r_pil.json()["annotation"]["summary"]["pillion_helmet"] == "no_helmet"


def test_audit_pass1_temporal_disagreement_iaa_export_and_adjudication_guards(dashboard_server) -> None:
    base_url, cfg, _srv = dashboard_server
    hdr_a = {"X-Reviewer-Id": "reviewer_a"}
    hdr_b = {"X-Reviewer-Id": "reviewer_b"}
    hdr_adj = {"X-Reviewer-Id": "adjudicator"}

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        # 1. WQ_002: same DRIVER_NO_HELMET label, disjoint time windows -> MUST flag disagreement!
        w_a = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET", start_t=0.5, end_t=1.2)
        w_b = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET", start_t=4.2, end_t=4.9)
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr_a, json=w_a).status_code == 200
        assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr_b, json=w_b).status_code == 200

        # Export now while ONLY ready_disagreed == 1 (executed_human_units == 0): hv_status MUST be IN_PROGRESS!
        exp1 = client.get("/api/adjudication/export", headers=hdr_adj).json()
        assert exp1["executed_human_units"] == 0
        assert exp1["human_verification_status"] == "IN_PROGRESS"

        # 2. WQ_003: 1-frame point events within 0.05s tolerance -> MUST agree!
        p_a = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET", start_t=1.200, end_t=1.200)
        p_b = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET", start_t=1.250, end_t=1.250)
        assert client.post("/api/reviewer/items/WQ_003/submit", headers=hdr_a, json=p_a).status_code == 200
        assert client.post("/api/reviewer/items/WQ_003/submit", headers=hdr_b, json=p_b).status_code == 200

        # 3. WQ_004: Insufficient vs No_Violation (tests 3-class binary kappa) + Pillion disagreement (driver agreed, pillion disagreed)
        i_a = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="PILLION_NO_HELMET", start_t=1.0, end_t=1.8)
        i_a["riders"].append({"rider_id": "R2", "moto_id": "M1", "role": "PILLION", "helmet_state": "NO_HELMET"})
        i_a["violation_events"][0]["rider_ids"] = ["R2"]
        i_b = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="INSUFFICIENT_VISUAL_EVIDENCE", start_t=1.0, end_t=1.8)
        i_b["riders"].append({"rider_id": "R2", "moto_id": "M1", "role": "PILLION", "helmet_state": "UNCERTAIN"})
        i_b["violation_events"][0]["rider_ids"] = ["R2"]
        assert client.post("/api/reviewer/items/WQ_004/submit", headers=hdr_a, json=i_a).status_code == 200
        assert client.post("/api/reviewer/items/WQ_004/submit", headers=hdr_b, json=i_b).status_code == 200

        # Unsubmitted draft by Reviewer A on WQ_005 & DRAFT adjudication on WQ_004 must never leak into exports
        assert client.put("/api/reviewer/items/WQ_005/draft", headers=hdr_a, json=_sample_payload()).status_code == 200
        assert client.post(
            "/api/adjudication/items/WQ_004",
            headers=hdr_adj,
            json={"status": "DRAFT", "adjudicator_notes": "Pending closer look."},
        ).status_code == 200

        ov = client.get("/api/adjudication/overview", headers=hdr_adj).json()
        assert ov["counts"]["ready_disagreed"] == 2  # WQ_002 (window) and WQ_004
        assert ov["counts"]["ready_agreed"] == 1     # WQ_003
        # binary_violation kappa includes INSUFFICIENT_VISUAL_EVIDENCE item -> n == 3!
        assert ov["inter_annotator_agreement"]["binary_violation"]["n"] == 3
        # rider_helmet includes 3 drivers + 1 pillion pair -> n == 4!
        assert ov["inter_annotator_agreement"]["rider_helmet"]["n"] == 4
        assert ov["inter_annotator_agreement"]["rider_helmet"]["kappa"] < 1.0

        det_2 = client.get("/api/adjudication/items/WQ_002", headers=hdr_adj).json()
        assert [d["field"] for d in det_2["disagreements"]] == ["violation_window_ref_sec"]

        # Adjudication cross-field & time-bounds validation guards
        assert client.post(
            "/api/adjudication/items/WQ_002",
            headers=hdr_adj,
            json={
                "status": "ADJUDICATED",
                "final_link_status": "BACK_VIEW_ONLY",
                "final_rider_count": 1,
                "final_driver_helmet": "helmet",  # contradicts DRIVER_NO_HELMET!
                "final_event_type": "DRIVER_NO_HELMET",
                "final_plate_visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR",
                "adjudicator_notes": "Invalid test",
            },
        ).status_code == 422
        assert client.post(
            "/api/adjudication/items/WQ_002",
            headers=hdr_adj,
            json={
                "status": "ADJUDICATED",
                "final_link_status": "BACK_VIEW_ONLY",
                "final_rider_count": 1,
                "final_driver_helmet": "no_helmet",
                "final_event_type": "DRIVER_NO_HELMET",
                "final_plate_visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR",
                "final_violation_start_t_ref_sec": 999.0,  # OOB!
                "final_violation_end_t_ref_sec": 1000.0,
                "adjudicator_notes": "OOB test",
            },
        ).status_code == 422

        # Valid adjudication of WQ_002 + UNRESOLVED on agreed WQ_003
        assert client.post(
            "/api/adjudication/items/WQ_002",
            headers=hdr_adj,
            json={
                "status": "ADJUDICATED",
                "final_link_status": "BACK_VIEW_ONLY",
                "final_rider_count": 1,
                "final_driver_helmet": "no_helmet",
                "final_event_type": "DRIVER_NO_HELMET",
                "final_plate_visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR",
                "final_violation_start_t_ref_sec": 0.5,
                "final_violation_end_t_ref_sec": 1.2,
                "adjudicator_notes": "Verified first transit window.",
            },
        ).status_code == 200
        assert client.post(
            "/api/adjudication/items/WQ_003",
            headers=hdr_adj,
            json={"status": "UNRESOLVED", "adjudicator_notes": "Needs second adjudicator review."},
        ).status_code == 200

        exp2 = client.get("/api/adjudication/export", headers=hdr_adj).json()
        assert exp2["executed_human_units"] == 1  # WQ_002 only (UNRESOLVED WQ_003 must NOT count!)
        rev_a_lines = [
            json.loads(line)["workqueue_id"]
            for line in (cfg.exports_dir / "annotations_reviewer_a.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert "WQ_005" not in rev_a_lines  # unsubmitted draft must not be exported
        adj_lines = [
            json.loads(line)["workqueue_id"]
            for line in (cfg.exports_dir / "adjudicated_annotations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert adj_lines == ["WQ_002"]  # DRAFT WQ_004 & UNRESOLVED WQ_003 excluded
        with open(cfg.exports_dir / "blind_annotation_workqueue_populated.csv", "r", encoding="utf-8", newline="") as f:
            pop_rows = {r["workqueue_id"]: r for r in csv.DictReader(f)}
        assert pop_rows["WQ_003"]["review_status"] == "PENDING_HUMAN_REVIEW"
        snap_meta = json.loads((cfg.exports_dir / "release_metadata_snapshot.json").read_text(encoding="utf-8"))
        assert snap_meta["inter_annotator_agreement"]["adjudicated_disagreement_count"] == 1
        assert snap_meta["human_verification_status"] == "IN_PROGRESS"
        assert snap_meta["human_verified"] is False


def test_audit_pass1_negative_offset_stepping_snapshots_and_bad_routes(dashboard_server) -> None:
    base_url, cfg, _srv = dashboard_server
    hdr_a = {"X-Reviewer-Id": "reviewer_a"}

    with httpx.Client(base_url=base_url, timeout=15.0) as client:
        # 1. Cross-camera step boundary handoff on negative-offset pair_001
        r_prev = client.get("/api/frame_info/pair_001?t_ref=0.0&step_frames=-1&anchor=back", headers=hdr_a)
        assert r_prev.status_code == 200
        prev_t = r_prev.json()["t_ref_sec"]
        assert prev_t < 0.0, f"Expected step -1 from Back frame 0 on pair_001 to cross onto Front < 0.0, got {prev_t}"

        r_next = client.get(f"/api/frame_info/pair_001?t_ref={prev_t}&step_frames=1&anchor=back", headers=hdr_a)
        assert r_next.status_code == 200
        assert abs(r_next.json()["t_ref_sec"] - 0.0) < 1e-4

        # 2. Same-second rapid snapshots produce unique files
        s1 = client.post(
            "/api/reviewer/snapshots",
            headers=hdr_a,
            json={"workqueue_id": "WQ_001", "camera": "back", "t_ref_sec": 0.0, "ext": "jpg"},
        )
        s2 = client.post(
            "/api/reviewer/snapshots",
            headers=hdr_a,
            json={"workqueue_id": "WQ_001", "camera": "back", "t_ref_sec": 0.0, "ext": "jpg"},
        )
        assert s1.status_code == 200 and s2.status_code == 200
        snap1 = s1.json()["snapshot"]
        snap2 = s2.json()["snapshot"]
        assert snap1["snapshot_id"] != snap2["snapshot_id"]
        assert (cfg.snapshots_dir / snap1["relpath"]).exists()
        assert (cfg.snapshots_dir / snap2["relpath"]).exists()

        # 3. Malformed paths and non-finite query inputs cleanly return 4xx (not 200 or 500)
        assert client.post("/api/reviewer/items/WQ_001/submit/extra", headers=hdr_a, json={}).status_code == 404
        assert client.get("/api/frame_info/pair_001?t_ref=nan", headers=hdr_a).status_code == 400
        assert client.get("/api/frame/pair_001/back?crop=1,2", headers=hdr_a).status_code == 400


def test_audit_pass2_all_complete_transitions_to_complete_adjudicated(tmp_path: Path) -> None:
    """Verify PENDING_HUMAN_REVIEW -> IN_PROGRESS (incl. UNRESOLVED) -> COMPLETE_ADJUDICATED on full completion."""
    base_cfg = DashboardConfig.from_repo(
        workspace_dir=tmp_path / "base_ws",
        db_path=tmp_path / "base_ws" / "base.sqlite3",
    )
    mini_csv = tmp_path / "mini_workqueue.csv"
    with open(base_cfg.workqueue_csv, "r", encoding="utf-8", newline="") as fin:
        rows = list(csv.DictReader(fin))
        fieldnames = list(rows[0].keys())
    with open(mini_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[:2])

    cfg = DashboardConfig.from_repo(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "ws" / "review.sqlite3",
        host="127.0.0.1",
        port=0,
        allow_proxy_transcode=False,
    )
    cfg.workqueue_csv = mini_csv

    srv = create_server(cfg)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{srv.server_address[1]}"
        hdr_a = {"X-Reviewer-Id": "reviewer_a"}
        hdr_b = {"X-Reviewer-Id": "reviewer_b"}
        hdr_adj = {"X-Reviewer-Id": "adjudicator"}
        with httpx.Client(base_url=base_url, timeout=15.0) as client:
            # 0 submissions -> PENDING_HUMAN_REVIEW
            exp0 = client.get("/api/adjudication/export", headers=hdr_adj).json()
            assert exp0["human_verification_status"] == "PENDING_HUMAN_REVIEW"
            snap0 = json.loads((cfg.exports_dir / "release_metadata_snapshot.json").read_text(encoding="utf-8"))
            assert snap0["human_verified"] is False

            # WQ_001 agreed; WQ_002 disagreed
            agree_payload = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET")
            disagree_a = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="HELMET", event_type="NO_VIOLATION")
            disagree_b = _sample_payload(link_status="BACK_VIEW_ONLY", helmet_driver="NO_HELMET", event_type="DRIVER_NO_HELMET")
            assert client.post("/api/reviewer/items/WQ_001/submit", headers=hdr_a, json=agree_payload).status_code == 200
            assert client.post("/api/reviewer/items/WQ_001/submit", headers=hdr_b, json=agree_payload).status_code == 200
            assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr_a, json=disagree_a).status_code == 200
            assert client.post("/api/reviewer/items/WQ_002/submit", headers=hdr_b, json=disagree_b).status_code == 200

            # UNRESOLVED must remain IN_PROGRESS (not COMPLETE_ADJUDICATED)
            assert client.post(
                "/api/adjudication/items/WQ_002",
                headers=hdr_adj,
                json={"status": "UNRESOLVED", "adjudicator_notes": "Escalated."},
            ).status_code == 200
            exp_unres = client.get("/api/adjudication/export", headers=hdr_adj).json()
            assert exp_unres["human_verification_status"] == "IN_PROGRESS"
            snap_unres = json.loads((cfg.exports_dir / "release_metadata_snapshot.json").read_text(encoding="utf-8"))
            assert snap_unres["human_verified"] is False

            # Resolve WQ_002 -> COMPLETE_ADJUDICATED + human_verified=True + schema-valid
            assert client.post(
                "/api/adjudication/items/WQ_002",
                headers=hdr_adj,
                json={
                    "status": "ADJUDICATED",
                    "final_link_status": "BACK_VIEW_ONLY",
                    "final_rider_count": 1,
                    "final_driver_helmet": "no_helmet",
                    "final_event_type": "DRIVER_NO_HELMET",
                    "final_plate_visibility_tier": "PLATE_UNREADABLE_MOTION_BLUR",
                    "final_violation_start_t_ref_sec": 0.5,
                    "final_violation_end_t_ref_sec": 1.2,
                    "adjudicator_notes": "Confirmed violation.",
                },
            ).status_code == 200
            exp_done = client.get("/api/adjudication/export", headers=hdr_adj).json()
            assert exp_done["executed_human_units"] == 2
            assert exp_done["human_verification_status"] == "COMPLETE_ADJUDICATED"
            snap_meta = json.loads((cfg.exports_dir / "release_metadata_snapshot.json").read_text(encoding="utf-8"))
            assert snap_meta["human_verified"] is True
            assert snap_meta["human_verification_status"] == "COMPLETE_ADJUDICATED"
            schema_doc = json.loads(cfg.release_schema_json.read_text(encoding="utf-8"))
            jsonschema.validate(instance=snap_meta, schema=schema_doc)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)

