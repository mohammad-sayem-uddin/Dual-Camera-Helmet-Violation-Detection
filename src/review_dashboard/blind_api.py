"""Blind workqueue catalog, strict allowlist sanitization, and agreement math."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.review_dashboard.config import DashboardConfig
from src.review_dashboard.models import (
    PairTimingMeta,
    STRATUM_NEUTRAL_BUCKET,
    compute_pair_timeline_bounds,
    parse_interval_hint,
)


# Reviewer-visible keys are strictly allowlisted; no AI predictions, v1.0 GT,
# conflict hints (review_focus / source_origin), or peer annotations ever pass.
REVIEWER_ALLOWED_ITEM_KEYS: tuple[str, ...] = (
    "workqueue_id",
    "pair_id",
    "transit_index_in_pair",
    "sampling_bucket",
    "camera_scope",
    "sync_quality",
    "global_offset_sec",
    "sync_uncertainty_ms",
    "common_overlap_ref_sec",
    "overlap_start_ref_sec",
    "overlap_end_ref_sec",
    "front_duration_sec",
    "back_duration_sec",
    "min_t_ref_sec",
    "max_t_ref_sec",
    "front_transit_ref_sec_hint",
    "back_transit_ref_sec_hint",
    "front_hint_interval",
    "back_hint_interval",
    "drift_ms",
)


class BlindPermissionError(PermissionError):
    """Raised when an actor attempts to access restricted or peer data."""


class WorkqueueCatalog:
    """Read-only catalog loaded strictly from v1.1 blind queue + sync metadata."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self.pair_timings: dict[str, PairTimingMeta] = self._load_pair_timings()
        self.raw_items: list[dict[str, Any]] = self._load_workqueue()
        self.items_by_id: dict[str, dict[str, Any]] = {
            item["workqueue_id"]: item for item in self.raw_items
        }
        self.pair_ids: list[str] = sorted(self.pair_timings.keys())

    def _load_pair_timings(self) -> dict[str, PairTimingMeta]:
        with open(self.config.sync_metadata_json, "r", encoding="utf-8") as f:
            sync_rows = json.load(f)

        manifest_by_pair: dict[str, dict[str, str]] = {}
        with open(self.config.pair_manifest_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                manifest_by_pair[row["pair_id"]] = row

        result: dict[str, PairTimingMeta] = {}
        for item in sync_rows:
            pid = item["pair_id"]
            offset = float(item["offset_sec"])
            back_dur = float(item["back_duration"])
            front_dur = float(item["front_duration"])
            overlap_dur = float(item["overlap_duration"])
            overlap_start = max(0.0, offset)
            overlap_end = min(back_dur, offset + front_dur)
            min_ref, max_ref = compute_pair_timeline_bounds(offset, back_dur, front_dur)
            m_row = manifest_by_pair.get(pid, {})
            result[pid] = PairTimingMeta(
                pair_id=pid,
                front_file=m_row.get("front_file") or f"{pid}_front.mp4",
                back_file=m_row.get("back_file") or f"{pid}_back.mp4",
                global_offset_sec=offset,
                front_duration_sec=front_dur,
                back_duration_sec=back_dur,
                overlap_start_ref_sec=round(overlap_start, 6),
                overlap_end_ref_sec=round(overlap_end, 6),
                overlap_duration_sec=round(overlap_dur, 6),
                drift_ms=float(item.get("drift_ms", 0.0)),
                residual_error_ms=float(item.get("residual_error_ms", 3.0)),
                sync_quality=str(item.get("quality", "HIGH")),
                min_t_ref_sec=min_ref,
                max_t_ref_sec=max_ref,
            )
        return result

    def _load_workqueue(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(self.config.workqueue_csv, "r", encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                pid = raw["pair_id"].strip()
                timing = self.pair_timings[pid]
                raw_stratum = raw["priority_stratum"].strip()
                neutral_bucket = STRATUM_NEUTRAL_BUCKET.get(raw_stratum, "STRATUM_OTHER")
                front_hint = raw.get("front_transit_ref_sec_hint", "").strip()
                back_hint = raw.get("back_transit_ref_sec_hint", "").strip()
                rows.append(
                    {
                        "workqueue_id": raw["workqueue_id"].strip(),
                        "pair_id": pid,
                        "transit_index_in_pair": int(raw["transit_index_in_pair"]),
                        "raw_priority_stratum": raw_stratum,
                        "source_origin": raw.get("source_origin", "").strip(),
                        "review_focus": raw.get("review_focus", "").strip(),
                        "sampling_bucket": neutral_bucket,
                        "camera_scope": raw.get("camera_scope", "").strip(),
                        "sync_quality": raw.get("sync_quality", timing.sync_quality).strip(),
                        "global_offset_sec": float(raw.get("global_offset_sec") or timing.global_offset_sec),
                        "sync_uncertainty_ms": float(raw.get("sync_uncertainty_ms") or timing.residual_error_ms),
                        "common_overlap_ref_sec": raw.get("common_overlap_ref_sec", "").strip(),
                        "overlap_start_ref_sec": timing.overlap_start_ref_sec,
                        "overlap_end_ref_sec": timing.overlap_end_ref_sec,
                        "front_duration_sec": timing.front_duration_sec,
                        "back_duration_sec": timing.back_duration_sec,
                        "min_t_ref_sec": timing.min_t_ref_sec,
                        "max_t_ref_sec": timing.max_t_ref_sec,
                        "front_transit_ref_sec_hint": front_hint,
                        "back_transit_ref_sec_hint": back_hint,
                        "front_hint_interval": parse_interval_hint(front_hint),
                        "back_hint_interval": parse_interval_hint(back_hint),
                        "drift_ms": timing.drift_ms,
                    }
                )
        return rows

    def get_raw_item(self, workqueue_id: str) -> dict[str, Any]:
        if workqueue_id not in self.items_by_id:
            raise KeyError(f"Unknown workqueue_id: {workqueue_id}")
        return self.items_by_id[workqueue_id]

    def get_pair_timing(self, pair_id: str) -> PairTimingMeta:
        if pair_id not in self.pair_timings:
            raise KeyError(f"Unknown pair_id: {pair_id}")
        return self.pair_timings[pair_id]


def sanitize_item_for_reviewer(
    raw_item: dict[str, Any],
    *,
    my_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a strictly blind, allowlisted item view for a single reviewer."""
    sanitized = {key: raw_item[key] for key in REVIEWER_ALLOWED_ITEM_KEYS}
    sanitized["my_status"] = my_record["status"] if my_record else "NOT_STARTED"
    sanitized["my_version"] = my_record["version"] if my_record else 0
    sanitized["my_updated_at"] = my_record["updated_at"] if my_record else None
    sanitized["my_submitted_at"] = my_record["submitted_at"] if my_record else None
    return sanitized


def assert_reviewer_actor(actor: str) -> str:
    if actor not in ("reviewer_a", "reviewer_b"):
        raise BlindPermissionError(
            f"Reviewer endpoint requires reviewer_a or reviewer_b session, got {actor!r}."
        )
    return actor


def assert_adjudicator_actor(actor: str) -> str:
    if actor != "adjudicator":
        raise BlindPermissionError(
            f"Adjudication/coordinator endpoint is forbidden for blind reviewer {actor!r}."
        )
    return actor


def compute_reviewer_disagreements(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare Reviewer A and Reviewer B canonical summaries field-by-field for adjudication."""
    fields = (
        "link_status",
        "rider_count",
        "driver_helmet",
        "pillion_helmet",
        "violation_event_type",
        "binary_violation",
        "plate_visibility_tier",
        "plate_text_normalized",
    )
    diffs: list[dict[str, Any]] = []
    for field in fields:
        val_a = summary_a.get(field)
        val_b = summary_b.get(field)
        if val_a != val_b:
            diffs.append(
                {
                    "field": field,
                    "reviewer_a": val_a,
                    "reviewer_b": val_b,
                }
            )
    return diffs


def compute_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    """Compute unweighted Cohen's kappa without external dependencies."""
    pairs = [(str(a), str(b)) for a, b in zip(labels_a, labels_b) if a != "" and b != ""]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "observed_agreement": None, "kappa": None}

    categories = sorted({x for pair in pairs for x in pair})
    po = sum(1 for a, b in pairs if a == b) / n
    counts_a = {c: sum(1 for a, _ in pairs if a == c) for c in categories}
    counts_b = {c: sum(1 for _, b in pairs if b == c) for c in categories}
    pe = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)

    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(1.0 - po) < 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "n": n,
        "observed_agreement": round(po, 6),
        "kappa": round(kappa, 6),
    }


def resolve_video_path_safe(config: DashboardConfig, catalog: WorkqueueCatalog, pair_id: str, camera: str) -> Path:
    """Safely resolve a front/back video path inside data_root preventing directory traversal."""
    if camera not in ("front", "back"):
        raise ValueError(f"Invalid camera {camera!r}")
    timing = catalog.get_pair_timing(pair_id)
    filename = timing.front_file if camera == "front" else timing.back_file
    cam_dir = (config.data_root / camera).resolve()
    candidate = (cam_dir / filename).resolve()
    if cam_dir not in candidate.parents:
        raise PermissionError("Path traversal blocked.")
    if not candidate.exists():
        raise FileNotFoundError(f"Video file missing: {candidate}")
    return candidate
