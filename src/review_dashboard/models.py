"""Canonical domain constants, enum mappings, and dual-camera timeline math."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REVIEWER_ACTORS: tuple[str, ...] = ("reviewer_a", "reviewer_b")
ADJUDICATOR_ACTOR = "adjudicator"
VALID_ACTORS: tuple[str, ...] = ("reviewer_a", "reviewer_b", "adjudicator")

REVIEWER_LABELS: dict[str, str] = {
    "reviewer_a": "Reviewer A (Independent Blind Annotator)",
    "reviewer_b": "Reviewer B (Independent Blind Annotator)",
    "adjudicator": "Adjudication Coordinator (Restricted — Post-Submission Only)",
}

CROSS_CAMERA_LINK_STATUSES: tuple[str, ...] = (
    "LINKED_BOTH_VIEWS",
    "FRONT_VIEW_ONLY",
    "BACK_VIEW_ONLY",
    "AMBIGUOUS_HUMAN_UNCERTAIN",
    "NOT_A_MOTORCYCLE_TRANSIT",
)

RIDER_ROLES: tuple[str, ...] = ("DRIVER", "PILLION", "UNCERTAIN")

HELMET_STATES: tuple[str, ...] = ("HELMET", "NO_HELMET", "UNCERTAIN", "NOT_VISIBLE")

# Deterministic mapping from fine-grained dashboard UI choices to v1.1 schema.json labels.
HELMET_TO_V1_1_LABEL: dict[str, str] = {
    "HELMET": "helmet",
    "NO_HELMET": "no_helmet",
    "UNCERTAIN": "occluded_unknown",
    "NOT_VISIBLE": "occluded_unknown",
}

V1_1_TO_UI_HELMET: dict[str, str] = {
    "helmet": "HELMET",
    "no_helmet": "NO_HELMET",
    "occluded_unknown": "UNCERTAIN",
}

PLATE_VISIBILITY_TIERS: tuple[str, ...] = (
    "PLATE_CLEAR",
    "PLATE_PARTIAL",
    "PLATE_UNREADABLE_MOTION_BLUR",
    "PLATE_NOT_VISIBLE_FRONT_ONLY",
)

VIOLATION_EVENT_TYPES: tuple[str, ...] = (
    "DRIVER_NO_HELMET",
    "PILLION_NO_HELMET",
    "BOTH_NO_HELMET",
    "NO_VIOLATION",
    "INSUFFICIENT_VISUAL_EVIDENCE",
)

CAMERA_EVIDENCE_CHOICES: tuple[str, ...] = (
    "BOTH_CAMERAS",
    "FRONT_ONLY",
    "BACK_ONLY",
    "UNCERTAIN",
)

UNCERTAINTY_REASON_TAGS: tuple[str, ...] = (
    "MOTION_BLUR",
    "LOW_LIGHT_GLARE",
    "PARTIAL_OCCLUSION",
    "HEAVY_OCCLUSION",
    "ROLE_AMBIGUOUS",
    "CROSS_CAMERA_ID_UNCERTAIN",
    "SYNC_JITTER_CONCERN",
    "MOTORCYCLE_VS_BICYCLE_AMBIGUOUS",
    "DISTANT_SMALL_SCALE",
)

STRATUM_NEUTRAL_BUCKET: dict[str, str] = {
    "STRATUM_1_GAPC_PILLION_OR_CONFLICT": "STRATUM_1",
    "STRATUM_2_UNCERTAIN_AMBIGUOUS_OR_DELTA": "STRATUM_2",
    "STRATUM_3_BASELINE_AND_PLATE_AUDIT": "STRATUM_3",
}


def t_front_from_ref(t_ref_sec: float, global_offset_sec: float) -> float:
    """Convert reference (Back PTS) time to Front stream-normalized PTS time.

    Synchronization equation: t_ref = front_pts_norm + global_offset_sec.
    """
    return float(t_ref_sec) - float(global_offset_sec)


def t_ref_from_front(front_pts_norm_sec: float, global_offset_sec: float) -> float:
    """Convert Front stream-normalized PTS time to Back-anchored reference time."""
    return float(front_pts_norm_sec) + float(global_offset_sec)


def compute_pair_timeline_bounds(
    global_offset_sec: float,
    back_duration_sec: float,
    front_duration_sec: float,
) -> tuple[float, float]:
    """Return inclusive [min_t_ref_sec, max_t_ref_sec] spanning both cameras on reference timeline."""
    min_ref = min(0.0, float(global_offset_sec))
    max_ref = max(float(back_duration_sec), float(global_offset_sec) + float(front_duration_sec))
    return round(min_ref, 6), round(max_ref, 6)


def parse_interval_hint(raw: str | None) -> list[float] | None:
    """Parse `'a..b'` interval strings from blind_annotation_workqueue.csv."""
    if not raw:
        return None
    text = str(raw).strip()
    if ".." not in text:
        return None
    parts = text.split("..", 1)
    try:
        start = float(parts[0].strip())
        end = float(parts[1].strip())
    except ValueError:
        return None
    return [round(start, 6), round(end, 6)]


@dataclass(slots=True)
class PairTimingMeta:
    """Synchronization & stream timing metadata for one front/back pair."""

    pair_id: str
    front_file: str
    back_file: str
    global_offset_sec: float
    front_duration_sec: float
    back_duration_sec: float
    overlap_start_ref_sec: float
    overlap_end_ref_sec: float
    overlap_duration_sec: float
    drift_ms: float
    residual_error_ms: float
    sync_quality: str
    min_t_ref_sec: float
    max_t_ref_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "front_file": self.front_file,
            "back_file": self.back_file,
            "global_offset_sec": self.global_offset_sec,
            "front_duration_sec": self.front_duration_sec,
            "back_duration_sec": self.back_duration_sec,
            "overlap_start_ref_sec": self.overlap_start_ref_sec,
            "overlap_end_ref_sec": self.overlap_end_ref_sec,
            "overlap_duration_sec": self.overlap_duration_sec,
            "drift_ms": self.drift_ms,
            "residual_error_ms": self.residual_error_ms,
            "sync_quality": self.sync_quality,
            "min_t_ref_sec": self.min_t_ref_sec,
            "max_t_ref_sec": self.max_t_ref_sec,
        }
