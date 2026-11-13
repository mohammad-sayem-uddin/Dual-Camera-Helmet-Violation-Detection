"""Transactional SQLite persistence, atomic JSON backups, audit log, and exports."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from typing import Any, Iterator

import jsonschema

from src.review_dashboard.blind_api import (
    WorkqueueCatalog,
    compute_cohens_kappa,
    compute_reviewer_disagreements,
)
from src.review_dashboard.config import DashboardConfig
from src.review_dashboard.validation import validate_adjudication_payload, validate_annotation_payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class ConflictError(Exception):
    """Raised on duplicate submit or locked item edit attempts."""


class ValidationFailedError(Exception):
    """Raised when strict submit/adjudication validation fails."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.warnings = warnings or []


class ReviewStorage:
    """Thread-safe SQLite + atomic JSON snapshot repository for blind human GT v1.1."""

    def __init__(self, config: DashboardConfig, catalog: WorkqueueCatalog) -> None:
        self.config = config
        self.catalog = catalog
        self._lock = threading.RLock()
        self.config.ensure_workspace_dirs()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.config.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reviewer_annotations (
                    workqueue_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL CHECK (reviewer_id IN ('reviewer_a', 'reviewer_b')),
                    status TEXT NOT NULL CHECK (status IN ('DRAFT', 'SUBMITTED')),
                    version INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    annotation_started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    PRIMARY KEY (workqueue_id, reviewer_id)
                );

                CREATE TABLE IF NOT EXISTS submission_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workqueue_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL CHECK (reviewer_id IN ('reviewer_a', 'reviewer_b')),
                    action TEXT NOT NULL CHECK (action IN ('SAVE_DRAFT', 'SUBMIT', 'SNAPSHOT')),
                    version INTEGER NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS adjudications (
                    workqueue_id TEXT PRIMARY KEY,
                    adjudicator_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('DRAFT', 'ADJUDICATED', 'UNRESOLVED')),
                    decision_json TEXT NOT NULL,
                    disagreements_json TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    finalized_at TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def _row_to_record(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "workqueue_id": row["workqueue_id"],
            "reviewer_id": row["reviewer_id"],
            "status": row["status"],
            "version": int(row["version"]),
            "payload": json.loads(row["payload_json"]),
            "summary": json.loads(row["summary_json"]),
            "warnings": json.loads(row["warnings_json"]),
            "annotation_started_at": row["annotation_started_at"],
            "updated_at": row["updated_at"],
            "submitted_at": row["submitted_at"],
        }

    def get_reviewer_item_record(self, workqueue_id: str, reviewer_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviewer_annotations WHERE workqueue_id = ? AND reviewer_id = ?",
                (workqueue_id, reviewer_id),
            ).fetchone()
            return self._row_to_record(row)

    def get_reviewer_status_map(self, reviewer_id: str) -> dict[str, dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT workqueue_id, status, version, updated_at, submitted_at "
                "FROM reviewer_annotations WHERE reviewer_id = ?",
                (reviewer_id,),
            ).fetchall()
            return {
                r["workqueue_id"]: {
                    "status": r["status"],
                    "version": int(r["version"]),
                    "updated_at": r["updated_at"],
                    "submitted_at": r["submitted_at"],
                }
                for r in rows
            }

    def get_reviewer_history_bundle(self, reviewer_id: str, limit: int = 200) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            ann_rows = conn.execute(
                "SELECT * FROM reviewer_annotations WHERE reviewer_id = ? ORDER BY updated_at DESC",
                (reviewer_id,),
            ).fetchall()
            items = []
            for row in ann_rows:
                rec = self._row_to_record(row)
                assert rec is not None
                raw = self.catalog.items_by_id.get(rec["workqueue_id"], {})
                items.append(
                    {
                        "workqueue_id": rec["workqueue_id"],
                        "pair_id": raw.get("pair_id", rec["payload"].get("pair_id")),
                        "camera_scope": raw.get("camera_scope", ""),
                        "status": rec["status"],
                        "version": rec["version"],
                        "summary": rec["summary"],
                        "updated_at": rec["updated_at"],
                        "submitted_at": rec["submitted_at"],
                        "motorcycles_count": len(rec["payload"].get("motorcycles", [])),
                        "riders_count": len(rec["payload"].get("riders", [])),
                        "events_count": len(rec["payload"].get("violation_events", [])),
                        "reviewer_notes": rec["payload"].get("reviewer_notes", ""),
                    }
                )
            hist_rows = conn.execute(
                "SELECT history_id, workqueue_id, action, version, payload_sha256, created_at "
                "FROM submission_history WHERE reviewer_id = ? ORDER BY history_id DESC LIMIT ?",
                (reviewer_id, limit),
            ).fetchall()
            return {"records": items, "timeline": [dict(r) for r in hist_rows]}

    def save_draft(
        self,
        workqueue_id: str,
        reviewer_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Save or autosave a blind reviewer draft; rejects edits to already submitted items."""
        raw_item = self.catalog.get_raw_item(workqueue_id)
        val = validate_annotation_payload(
            payload,
            workqueue_id=workqueue_id,
            pair_id=raw_item["pair_id"],
            min_t_ref_sec=raw_item["min_t_ref_sec"],
            max_t_ref_sec=raw_item["max_t_ref_sec"],
            global_offset_sec=raw_item["global_offset_sec"],
            mode="draft",
        )
        if not val.ok:
            raise ValidationFailedError(val.errors, val.warnings)

        now = utc_now_iso()
        payload_text = json.dumps(val.normalized_payload, indent=2, sort_keys=True)
        summary_text = json.dumps(val.summary, sort_keys=True)
        warnings_text = json.dumps(val.warnings)
        payload_sha = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                existing = conn.execute(
                    "SELECT status, version, annotation_started_at FROM reviewer_annotations "
                    "WHERE workqueue_id = ? AND reviewer_id = ?",
                    (workqueue_id, reviewer_id),
                ).fetchone()

                if existing and existing["status"] == "SUBMITTED":
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Item {workqueue_id} is already SUBMITTED by {reviewer_id} and locked against edits."
                    )

                next_version = (int(existing["version"]) + 1) if existing else 1
                started_at = existing["annotation_started_at"] if existing else now

                conn.execute(
                    """
                    INSERT INTO reviewer_annotations (
                        workqueue_id, reviewer_id, status, version,
                        payload_json, summary_json, warnings_json,
                        annotation_started_at, updated_at, submitted_at
                    ) VALUES (?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(workqueue_id, reviewer_id) DO UPDATE SET
                        status = 'DRAFT',
                        version = excluded.version,
                        payload_json = excluded.payload_json,
                        summary_json = excluded.summary_json,
                        warnings_json = excluded.warnings_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        workqueue_id,
                        reviewer_id,
                        next_version,
                        payload_text,
                        summary_text,
                        warnings_text,
                        started_at,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO submission_history (
                        workqueue_id, reviewer_id, action, version,
                        payload_sha256, payload_json, created_at
                    ) VALUES (?, ?, 'SAVE_DRAFT', ?, ?, ?, ?)
                    """,
                    (workqueue_id, reviewer_id, next_version, payload_sha, payload_text, now),
                )
                conn.execute(
                    "INSERT INTO audit_log (created_at, actor, action, resource_id, details_json) VALUES (?, ?, ?, ?, ?)",
                    (now, reviewer_id, "SAVE_DRAFT", workqueue_id, json.dumps({"version": next_version})),
                )
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise

        record = self.get_reviewer_item_record(workqueue_id, reviewer_id)
        assert record is not None
        return record

    def submit_annotation(
        self,
        workqueue_id: str,
        reviewer_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate strictly and lock an immutable reviewer submission + atomic JSON snapshot."""
        raw_item = self.catalog.get_raw_item(workqueue_id)
        val = validate_annotation_payload(
            payload,
            workqueue_id=workqueue_id,
            pair_id=raw_item["pair_id"],
            min_t_ref_sec=raw_item["min_t_ref_sec"],
            max_t_ref_sec=raw_item["max_t_ref_sec"],
            global_offset_sec=raw_item["global_offset_sec"],
            mode="submit",
        )
        if not val.ok:
            raise ValidationFailedError(val.errors, val.warnings)

        now = utc_now_iso()
        payload_text = json.dumps(val.normalized_payload, indent=2, sort_keys=True)
        summary_text = json.dumps(val.summary, sort_keys=True)
        warnings_text = json.dumps(val.warnings)
        payload_sha = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                existing = conn.execute(
                    "SELECT status, version, annotation_started_at FROM reviewer_annotations "
                    "WHERE workqueue_id = ? AND reviewer_id = ?",
                    (workqueue_id, reviewer_id),
                ).fetchone()

                if existing and existing["status"] == "SUBMITTED":
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Duplicate submission rejected: {workqueue_id} was already submitted by {reviewer_id}."
                    )

                next_version = (int(existing["version"]) + 1) if existing else 1
                started_at = existing["annotation_started_at"] if existing else now

                conn.execute(
                    """
                    INSERT INTO reviewer_annotations (
                        workqueue_id, reviewer_id, status, version,
                        payload_json, summary_json, warnings_json,
                        annotation_started_at, updated_at, submitted_at
                    ) VALUES (?, ?, 'SUBMITTED', ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workqueue_id, reviewer_id) DO UPDATE SET
                        status = 'SUBMITTED',
                        version = excluded.version,
                        payload_json = excluded.payload_json,
                        summary_json = excluded.summary_json,
                        warnings_json = excluded.warnings_json,
                        updated_at = excluded.updated_at,
                        submitted_at = excluded.submitted_at
                    """,
                    (
                        workqueue_id,
                        reviewer_id,
                        next_version,
                        payload_text,
                        summary_text,
                        warnings_text,
                        started_at,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO submission_history (
                        workqueue_id, reviewer_id, action, version,
                        payload_sha256, payload_json, created_at
                    ) VALUES (?, ?, 'SUBMIT', ?, ?, ?, ?)
                    """,
                    (workqueue_id, reviewer_id, next_version, payload_sha, payload_text, now),
                )
                conn.execute(
                    "INSERT INTO audit_log (created_at, actor, action, resource_id, details_json) VALUES (?, ?, ?, ?, ?)",
                    (now, reviewer_id, "SUBMIT", workqueue_id, json.dumps({"version": next_version, "sha256": payload_sha})),
                )
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise

        record = self.get_reviewer_item_record(workqueue_id, reviewer_id)
        assert record is not None
        snap_doc = {
            **record,
            "payload_sha256": payload_sha,
            "release_id": "dual_camera_gt_v1.1_blind_human",
            "blindness_verified": True,
        }
        atomic_write_text(
            self.config.submissions_dir / reviewer_id / f"{workqueue_id}.json",
            json.dumps(snap_doc, indent=2, sort_keys=True) + "\n",
        )
        return record

    def export_single_reviewer_annotations(self, reviewer_id: str) -> dict[str, Any]:
        """Export ONLY this reviewer's annotations without revealing peer answers."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reviewer_annotations WHERE reviewer_id = ? ORDER BY workqueue_id",
                (reviewer_id,),
            ).fetchall()
            records = [self._row_to_record(r) for r in rows]
        return {
            "release_id": "dual_camera_gt_v1.1_blind_human",
            "exported_at": utc_now_iso(),
            "reviewer_id": reviewer_id,
            "total_records": len(records),
            "submitted_count": sum(1 for r in records if r and r["status"] == "SUBMITTED"),
            "draft_count": sum(1 for r in records if r and r["status"] == "DRAFT"),
            "records": records,
        }

    # ------------------------------------------------------------------
    # Coordinator / Adjudication methods (FORBIDDEN for reviewer_a / reviewer_b)
    # ------------------------------------------------------------------

    def get_adjudication_overview(self) -> dict[str, Any]:
        """Build coordinator overview across all 92 items; computes agreement only where A&B submitted."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM reviewer_annotations").fetchall()
            adj_rows = conn.execute("SELECT * FROM adjudications").fetchall()

        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            rec = self._row_to_record(r)
            if rec:
                by_key[(rec["workqueue_id"], rec["reviewer_id"])] = rec

        adj_by_id = {
            r["workqueue_id"]: {
                "workqueue_id": r["workqueue_id"],
                "status": r["status"],
                "decision": json.loads(r["decision_json"]),
                "disagreements": json.loads(r["disagreements_json"]),
                "version": int(r["version"]),
                "updated_at": r["updated_at"],
                "finalized_at": r["finalized_at"],
            }
            for r in adj_rows
        }

        items: list[dict[str, Any]] = []
        a_viol: list[str] = []
        b_viol: list[str] = []
        a_helmet: list[str] = []
        b_helmet: list[str] = []
        a_link: list[str] = []
        b_link: list[str] = []
        counts = {
            "pending": 0,
            "a_only": 0,
            "b_only": 0,
            "ready_agreed": 0,
            "ready_disagreed": 0,
            "adjudicated": 0,
            "unresolved": 0,
        }

        for raw in self.catalog.raw_items:
            wid = raw["workqueue_id"]
            rec_a = by_key.get((wid, "reviewer_a"))
            rec_b = by_key.get((wid, "reviewer_b"))
            a_sub = bool(rec_a and rec_a["status"] == "SUBMITTED")
            b_sub = bool(rec_b and rec_b["status"] == "SUBMITTED")
            both_sub = a_sub and b_sub
            adj = adj_by_id.get(wid)

            disagreements: list[dict[str, Any]] = []
            if both_sub:
                assert rec_a is not None and rec_b is not None
                sum_a = rec_a["summary"]
                sum_b = rec_b["summary"]
                disagreements = compute_reviewer_disagreements(sum_a, sum_b)
                if "binary_violation" in sum_a and "binary_violation" in sum_b:
                    def _bin_viol_label(val: Any) -> str:
                        if val is True:
                            return "VIOLATION"
                        if val is False:
                            return "NON_VIOLATION"
                        return "INSUFFICIENT_EVIDENCE"

                    a_viol.append(_bin_viol_label(sum_a.get("binary_violation")))
                    b_viol.append(_bin_viol_label(sum_b.get("binary_violation")))
                dh_a = str(sum_a.get("driver_helmet") or "")
                dh_b = str(sum_b.get("driver_helmet") or "")
                if dh_a or dh_b:
                    a_helmet.append(dh_a or "NONE_REPORTED")
                    b_helmet.append(dh_b or "NONE_REPORTED")
                ph_a = str(sum_a.get("pillion_helmet") or "")
                ph_b = str(sum_b.get("pillion_helmet") or "")
                if ph_a or ph_b:
                    a_helmet.append(ph_a or "NO_PILLION")
                    b_helmet.append(ph_b or "NO_PILLION")
                if sum_a.get("link_status") and sum_b.get("link_status"):
                    a_link.append(str(sum_a["link_status"]))
                    b_link.append(str(sum_b["link_status"]))

            if adj and adj["status"] == "ADJUDICATED":
                stage = "ADJUDICATED"
                counts["adjudicated"] += 1
            elif adj and adj["status"] == "UNRESOLVED":
                stage = "UNRESOLVED"
                counts["unresolved"] += 1
            elif both_sub:
                if disagreements:
                    stage = "DISAGREEMENT_NEEDS_ADJUDICATION"
                    counts["ready_disagreed"] += 1
                else:
                    stage = "AB_AGREED_READY"
                    counts["ready_agreed"] += 1
            elif a_sub:
                stage = "A_ONLY_COMPLETE"
                counts["a_only"] += 1
            elif b_sub:
                stage = "B_ONLY_COMPLETE"
                counts["b_only"] += 1
            else:
                stage = "PENDING_HUMAN_REVIEW"
                counts["pending"] += 1

            items.append(
                {
                    "workqueue_id": wid,
                    "pair_id": raw["pair_id"],
                    "transit_index_in_pair": raw["transit_index_in_pair"],
                    "raw_priority_stratum": raw["raw_priority_stratum"],
                    "source_origin": raw["source_origin"],
                    "camera_scope": raw["camera_scope"],
                    "reviewer_a_submitted": a_sub,
                    "reviewer_b_submitted": b_sub,
                    "both_submitted": both_sub,
                    "disagreement_count": len(disagreements),
                    "disagreement_fields": [d["field"] for d in disagreements],
                    "adjudication_status": adj["status"] if adj else None,
                    "workflow_stage": stage,
                }
            )

        return {
            "total_items": len(items),
            "counts": counts,
            "inter_annotator_agreement": {
                "binary_violation": compute_cohens_kappa(a_viol, b_viol),
                "rider_helmet": compute_cohens_kappa(a_helmet, b_helmet),
                "cross_camera_link": compute_cohens_kappa(a_link, b_link),
            },
            "items": items,
        }

    def get_adjudication_item(self, workqueue_id: str) -> dict[str, Any]:
        """Return both submissions & disagreement diff ONLY after both A and B submitted."""
        raw = self.catalog.get_raw_item(workqueue_id)
        rec_a = self.get_reviewer_item_record(workqueue_id, "reviewer_a")
        rec_b = self.get_reviewer_item_record(workqueue_id, "reviewer_b")

        if not (rec_a and rec_a["status"] == "SUBMITTED") or not (rec_b and rec_b["status"] == "SUBMITTED"):
            raise ConflictError(
                f"Item {workqueue_id} cannot be opened for adjudication until BOTH Reviewer A and Reviewer B have submitted."
            )

        disagreements = compute_reviewer_disagreements(rec_a["summary"], rec_b["summary"])
        with self._lock, self._connect() as conn:
            adj_row = conn.execute(
                "SELECT * FROM adjudications WHERE workqueue_id = ?",
                (workqueue_id,),
            ).fetchone()

        adj = None
        if adj_row:
            adj = {
                "workqueue_id": adj_row["workqueue_id"],
                "adjudicator_id": adj_row["adjudicator_id"],
                "status": adj_row["status"],
                "decision": json.loads(adj_row["decision_json"]),
                "disagreements": json.loads(adj_row["disagreements_json"]),
                "version": int(adj_row["version"]),
                "updated_at": adj_row["updated_at"],
                "finalized_at": adj_row["finalized_at"],
            }

        return {
            "item": raw,
            "submission_a": rec_a,
            "submission_b": rec_b,
            "disagreements": disagreements,
            "adjudication": adj,
        }

    def save_adjudication(
        self,
        workqueue_id: str,
        decision_payload: dict[str, Any],
        *,
        adjudicator_id: str = "adjudicator",
    ) -> dict[str, Any]:
        """Save or finalize an adjudication decision after both reviewers have submitted."""
        detail = self.get_adjudication_item(workqueue_id)
        raw_item = detail["item"]
        ok, errors, norm = validate_adjudication_payload(
            decision_payload,
            min_t_ref_sec=raw_item["min_t_ref_sec"],
            max_t_ref_sec=raw_item["max_t_ref_sec"],
        )
        if not ok:
            raise ValidationFailedError(errors)

        now = utc_now_iso()
        status = norm["status"]
        finalized_at = now if status in ("ADJUDICATED", "UNRESOLVED") else None
        decision_text = json.dumps(norm, indent=2, sort_keys=True)
        diffs_text = json.dumps(detail["disagreements"], sort_keys=True)

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                existing = conn.execute(
                    "SELECT version FROM adjudications WHERE workqueue_id = ?",
                    (workqueue_id,),
                ).fetchone()
                next_ver = (int(existing["version"]) + 1) if existing else 1
                conn.execute(
                    """
                    INSERT INTO adjudications (
                        workqueue_id, adjudicator_id, status, decision_json,
                        disagreements_json, version, updated_at, finalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workqueue_id) DO UPDATE SET
                        adjudicator_id = excluded.adjudicator_id,
                        status = excluded.status,
                        decision_json = excluded.decision_json,
                        disagreements_json = excluded.disagreements_json,
                        version = excluded.version,
                        updated_at = excluded.updated_at,
                        finalized_at = excluded.finalized_at
                    """,
                    (
                        workqueue_id,
                        adjudicator_id,
                        status,
                        decision_text,
                        diffs_text,
                        next_ver,
                        now,
                        finalized_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO audit_log (created_at, actor, action, resource_id, details_json) VALUES (?, ?, ?, ?, ?)",
                    (now, adjudicator_id, f"ADJUDICATION_{status}", workqueue_id, json.dumps({"version": next_ver})),
                )
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise

        snap_doc = {
            "workqueue_id": workqueue_id,
            "pair_id": detail["item"]["pair_id"],
            "status": status,
            "version": next_ver,
            "disagreements": detail["disagreements"],
            "decision": norm,
            "updated_at": now,
            "finalized_at": finalized_at,
        }
        atomic_write_text(
            self.config.adjudications_dir / f"{workqueue_id}.json",
            json.dumps(snap_doc, indent=2, sort_keys=True) + "\n",
        )
        return self.get_adjudication_item(workqueue_id)

    def create_backup(self, label: str = "manual") -> Path:
        """Create a consistent SQLite backup snapshot inside review_workspace/backups/."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        dest = self.config.backups_dir / f"review_dashboard_{stamp}_{safe_label}.sqlite3"
        with self._lock:
            src = sqlite3.connect(str(self.config.db_path), timeout=30.0)
            dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
        return dest

    def export_full_v1_1_bundle(self) -> dict[str, Any]:
        """Write complete blind-verified & adjudicated export files to review_workspace/exports/.

        Never overwrites root pending workqueue files automatically and preserves
        v1.1 schema compliance cleanly.
        """
        overview = self.get_adjudication_overview()
        with self._lock, self._connect() as conn:
            ann_rows = conn.execute("SELECT * FROM reviewer_annotations").fetchall()
            adj_rows = conn.execute("SELECT * FROM adjudications").fetchall()

        by_key = {}
        for r in ann_rows:
            rec = self._row_to_record(r)
            if rec:
                by_key[(rec["workqueue_id"], rec["reviewer_id"])] = rec
        adj_by_id = {r["workqueue_id"]: dict(r) for r in adj_rows}

        exports_dir = self.config.exports_dir
        exports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Per-reviewer JSONL (export SUBMITTED records only; never leak unsubmitted drafts)
        for rev in ("reviewer_a", "reviewer_b"):
            lines = [
                json.dumps(by_key[(raw["workqueue_id"], rev)], sort_keys=True)
                for raw in self.catalog.raw_items
                if (raw["workqueue_id"], rev) in by_key
                and by_key[(raw["workqueue_id"], rev)]["status"] == "SUBMITTED"
            ]
            atomic_write_text(exports_dir / f"annotations_{rev}.jsonl", ("\n".join(lines) + "\n") if lines else "")

        # 2. Adjudicated JSONL (export finalized ADJUDICATED rows only)
        adj_lines = []
        for raw in self.catalog.raw_items:
            wid = raw["workqueue_id"]
            if wid in adj_by_id and adj_by_id[wid]["status"] == "ADJUDICATED":
                row = adj_by_id[wid]
                adj_lines.append(
                    json.dumps(
                        {
                            "workqueue_id": wid,
                            "pair_id": raw["pair_id"],
                            "status": row["status"],
                            "decision": json.loads(row["decision_json"]),
                            "disagreements": json.loads(row["disagreements_json"]),
                            "version": int(row["version"]),
                            "finalized_at": row["finalized_at"],
                        },
                        sort_keys=True,
                    )
                )
        atomic_write_text(
            exports_dir / "adjudicated_annotations.jsonl",
            ("\n".join(adj_lines) + "\n") if adj_lines else "",
        )

        # 3. Populated workqueue CSV matching checked-in CSV columns verbatim
        with open(self.config.workqueue_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            base_rows = list(reader)

        executed_units = 0
        adjudicated_disagreement_count = 0
        for row in base_rows:
            wid = row["workqueue_id"]
            rec_a = by_key.get((wid, "reviewer_a"))
            rec_b = by_key.get((wid, "reviewer_b"))
            a_sub = bool(rec_a and rec_a["status"] == "SUBMITTED")
            b_sub = bool(rec_b and rec_b["status"] == "SUBMITTED")
            if a_sub:
                assert rec_a is not None
                sa = rec_a["summary"]
                row["annotator_a_link_status"] = str(sa["link_status"])
                row["annotator_a_rider_count"] = str(sa["rider_count"])
                row["annotator_a_driver_helmet"] = str(sa["driver_helmet"])
                row["annotator_a_pillion_helmet"] = str(sa["pillion_helmet"])
                row["annotator_a_violation_window_ref_sec"] = str(sa["violation_window_ref_sec"])
                row["annotator_a_plate_visibility_tier"] = str(sa["plate_visibility_tier"])
                row["annotator_a_plate_text_normalized"] = str(sa["plate_text_normalized"])
            if b_sub:
                assert rec_b is not None
                sb = rec_b["summary"]
                row["annotator_b_link_status"] = str(sb["link_status"])
                row["annotator_b_rider_count"] = str(sb["rider_count"])
                row["annotator_b_driver_helmet"] = str(sb["driver_helmet"])
                row["annotator_b_pillion_helmet"] = str(sb["pillion_helmet"])
                row["annotator_b_violation_window_ref_sec"] = str(sb["violation_window_ref_sec"])
                row["annotator_b_plate_visibility_tier"] = str(sb["plate_visibility_tier"])
                row["annotator_b_plate_text_normalized"] = str(sb["plate_text_normalized"])

            adj_raw = adj_by_id.get(wid)
            if adj_raw and adj_raw["status"] == "ADJUDICATED":
                dec = json.loads(adj_raw["decision_json"])
                row["adjudicator_final_event_type"] = str(dec["final_event_type"])
                row["adjudicator_plate_visibility_tier"] = str(dec["final_plate_visibility_tier"])
                row["adjudicator_plate_text_normalized"] = str(dec["final_plate_text_normalized"])
                row["review_status"] = "ADJUDICATED_COMPLETE"
                executed_units += 1
                diffs_here = (
                    compute_reviewer_disagreements(rec_a["summary"], rec_b["summary"])  # type: ignore[index]
                    if (a_sub and b_sub)
                    else []
                )
                stored_diffs = json.loads(adj_raw["disagreements_json"])
                if diffs_here or stored_diffs:
                    adjudicated_disagreement_count += 1
            elif adj_raw and adj_raw["status"] == "UNRESOLVED":
                row["review_status"] = "PENDING_HUMAN_REVIEW"
            elif a_sub and b_sub:
                diffs = compute_reviewer_disagreements(rec_a["summary"], rec_b["summary"])  # type: ignore[index]
                row["review_status"] = "AB_AGREED" if not diffs else "PENDING_HUMAN_REVIEW"
                if not diffs:
                    executed_units += 1
            elif a_sub:
                row["review_status"] = "A_ONLY_COMPLETE"
            elif b_sub:
                row["review_status"] = "B_ONLY_COMPLETE"
            else:
                row["review_status"] = "PENDING_HUMAN_REVIEW"

        csv_tmp_fd, csv_tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(exports_dir))
        try:
            with os.fdopen(csv_tmp_fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(base_rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(csv_tmp_path, exports_dir / "blind_annotation_workqueue_populated.csv")
        finally:
            if os.path.exists(csv_tmp_path):
                os.unlink(csv_tmp_path)

        # 4. Schema-validated release_metadata_snapshot.json & agreement_report.json
        with open(self.config.release_metadata_json, "r", encoding="utf-8") as f:
            base_meta = json.load(f)
        with open(self.config.release_schema_json, "r", encoding="utf-8") as f:
            schema_doc = json.load(f)

        total_items = len(self.catalog.raw_items)
        pending_units = max(0, total_items - executed_units)
        counts = overview["counts"]
        has_any_submission_or_adj = any(
            counts[k] > 0
            for k in ("a_only", "b_only", "ready_agreed", "ready_disagreed", "adjudicated", "unresolved")
        )
        all_complete = (
            executed_units == total_items
            and counts["ready_disagreed"] == 0
            and counts["unresolved"] == 0
            and counts["a_only"] == 0
            and counts["b_only"] == 0
            and counts["pending"] == 0
        )
        if not has_any_submission_or_adj:
            hv_status = "PENDING_HUMAN_REVIEW"
        elif all_complete:
            hv_status = "COMPLETE_ADJUDICATED"
        else:
            hv_status = "IN_PROGRESS"

        iaa = overview["inter_annotator_agreement"]
        meta_snapshot = dict(base_meta)
        meta_snapshot["human_verification_status"] = hv_status
        meta_snapshot["human_verified"] = bool(all_complete)
        meta_snapshot["executed_human_units"] = executed_units
        meta_snapshot["pending_human_units"] = pending_units
        meta_snapshot["fabricated_labels_present"] = False
        meta_snapshot["export_generated_at"] = utc_now_iso()
        meta_snapshot["inter_annotator_agreement"] = {
            k: v for k, v in {
                "cohens_kappa_binary_violation": iaa["binary_violation"]["kappa"],
                "cohens_kappa_rider_helmet": iaa["rider_helmet"]["kappa"],
                "cohens_kappa_cross_camera_link": iaa["cross_camera_link"]["kappa"],
                "adjudicated_disagreement_count": adjudicated_disagreement_count,
            }.items() if v is not None
        } or None

        jsonschema.validate(instance=meta_snapshot, schema=schema_doc)
        atomic_write_text(
            exports_dir / "release_metadata_snapshot.json",
            json.dumps(meta_snapshot, indent=2) + "\n",
        )
        atomic_write_text(
            exports_dir / "agreement_report.json",
            json.dumps(
                {
                    "exported_at": utc_now_iso(),
                    "counts": overview["counts"],
                    "inter_annotator_agreement": iaa,
                },
                indent=2,
            )
            + "\n",
        )

        return {
            "exports_dir": str(exports_dir),
            "files": [
                "annotations_reviewer_a.jsonl",
                "annotations_reviewer_b.jsonl",
                "adjudicated_annotations.jsonl",
                "blind_annotation_workqueue_populated.csv",
                "release_metadata_snapshot.json",
                "agreement_report.json",
            ],
            "executed_human_units": executed_units,
            "pending_human_units": pending_units,
            "human_verification_status": hv_status,
        }

    def save_reviewer_snapshot_image(
        self,
        reviewer_id: str,
        workqueue_id: str,
        camera: str,
        frame_index: int,
        pts_norm_sec: float,
        t_ref_sec: float,
        image_bytes: bytes,
        ext: str = "jpg",
    ) -> dict[str, Any]:
        """Store a reviewer-captured frame snapshot exclusively inside that reviewer's directory."""
        import uuid

        stamp = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')}_{uuid.uuid4().hex[:4]}Z"
        safe_ext = "png" if ext.lower() == "png" else "jpg"
        snap_id = f"{workqueue_id}_{camera}_f{frame_index}_{stamp}"
        filename = f"{snap_id}.{safe_ext}"
        dest = self.config.snapshots_dir / reviewer_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=f".{safe_ext}", dir=str(dest.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(image_bytes)
                handle.flush()
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "INSERT INTO audit_log (created_at, actor, action, resource_id, details_json) VALUES (?, ?, 'SNAPSHOT', ?, ?)",
                (now, reviewer_id, workqueue_id, json.dumps({"filename": filename, "frame_index": frame_index})),
            )
            conn.execute("COMMIT;")

        return {
            "snapshot_id": snap_id,
            "camera": camera,
            "frame_index": frame_index,
            "pts_norm_sec": round(pts_norm_sec, 6),
            "t_ref_sec": round(t_ref_sec, 6),
            "relpath": f"{reviewer_id}/{filename}",
            "url": f"/api/reviewer/snapshots/{reviewer_id}/{filename}",
            "created_at": now,
        }
