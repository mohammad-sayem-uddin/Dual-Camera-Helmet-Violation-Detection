"""Payload normalization, validation, and canonical summary extraction."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from src.review_dashboard.models import (
    CAMERA_EVIDENCE_CHOICES,
    CROSS_CAMERA_LINK_STATUSES,
    HELMET_STATES,
    HELMET_TO_V1_1_LABEL,
    PLATE_VISIBILITY_TIERS,
    RIDER_ROLES,
    VIOLATION_EVENT_TYPES,
    t_front_from_ref,
)


TIMELINE_TOLERANCE_SEC = 0.08
PLATE_NORMALIZE_RE = re.compile(r"[^A-Z0-9-]+")
EVENT_SEVERITY_ORDER: dict[str, int] = {
    "BOTH_NO_HELMET": 5,
    "DRIVER_NO_HELMET": 4,
    "PILLION_NO_HELMET": 3,
    "INSUFFICIENT_VISUAL_EVIDENCE": 2,
    "NO_VIOLATION": 1,
}


@dataclass(slots=True)
class ValidationResult:
    """Result of validating and normalizing a reviewer annotation payload."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    normalized_payload: dict[str, Any]
    summary: dict[str, Any]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, 6)


def _parse_optional_float(value: Any, field_label: str, errors: list[str]) -> float | None:
    if value is None or value == "":
        return None
    parsed = _as_float(value)
    if parsed is None:
        errors.append(f"{field_label} must be a finite number (got {value!r}).")
    return parsed


def _aggregate_helmet_labels(labels: list[str]) -> str:
    """Deterministically combine multiple rider helmet labels (no_helmet > occluded_unknown > helmet)."""
    cleaned = [lbl for lbl in labels if lbl]
    if not cleaned:
        return ""
    if "no_helmet" in cleaned:
        return "no_helmet"
    if "occluded_unknown" in cleaned:
        return "occluded_unknown"
    return "helmet"


def normalize_plate_text(raw: str | None) -> str | None:
    """Normalize license plate text into uppercase alphanumeric/hyphen string."""
    if raw is None:
        return None
    cleaned = PLATE_NORMALIZE_RE.sub("", str(raw).strip().upper())
    return cleaned or None


def _check_time_in_bounds(
    value: float,
    min_ref: float,
    max_ref: float,
    field_label: str,
    errors: list[str],
) -> None:
    if value < min_ref - TIMELINE_TOLERANCE_SEC or value > max_ref + TIMELINE_TOLERANCE_SEC:
        errors.append(
            f"{field_label} ({value:.3f}s) is outside pair reference bounds "
            f"[{min_ref:.3f}s, {max_ref:.3f}s]."
        )


def validate_annotation_payload(
    payload: dict[str, Any],
    *,
    workqueue_id: str,
    pair_id: str,
    min_t_ref_sec: float,
    max_t_ref_sec: float,
    global_offset_sec: float,
    mode: str = "draft",
) -> ValidationResult:
    """Validate and canonicalize reviewer annotation JSON for draft save or final submission."""
    if mode not in ("draft", "submit"):
        raise ValueError(f"Unsupported validation mode: {mode}")

    errors: list[str] = []
    warnings: list[str] = []

    raw_link = str(payload.get("item_link_status") or "").strip()
    if raw_link and raw_link not in CROSS_CAMERA_LINK_STATUSES:
        errors.append(f"Invalid item_link_status: {raw_link!r}")
        raw_link = ""

    is_non_moto = bool(payload.get("is_non_motorcycle_transit", False)) or (
        raw_link == "NOT_A_MOTORCYCLE_TRANSIT"
    )
    if is_non_moto:
        raw_link = "NOT_A_MOTORCYCLE_TRANSIT"
    elif mode == "submit" and not raw_link:
        errors.append("item_link_status must be explicitly selected before submission.")

    unc_obj = payload.get("uncertainty") or {}
    unc_reasons = [str(r).strip() for r in (unc_obj.get("reasons") or []) if str(r).strip()]
    is_uncertain = bool(unc_obj.get("is_uncertain", False)) or bool(unc_reasons)
    reviewer_notes = str(payload.get("reviewer_notes") or "").strip()

    motorcycles_in = payload.get("motorcycles") or []
    riders_in = payload.get("riders") or []
    events_in = payload.get("violation_events") or []
    snapshots_in = payload.get("snapshots") or []

    norm_motos: list[dict[str, Any]] = []
    seen_moto_ids: set[str] = set()
    moto_by_id: dict[str, dict[str, Any]] = {}
    for idx, m in enumerate(motorcycles_in):
        if not isinstance(m, dict):
            errors.append(f"motorcycles[{idx}] must be an object.")
            continue
        moto_id = str(m.get("moto_id") or f"M{idx + 1}").strip()
        if not moto_id:
            errors.append(f"motorcycles[{idx}] missing moto_id.")
            continue
        if moto_id in seen_moto_ids:
            errors.append(f"Duplicate moto_id {moto_id!r}.")
            continue
        seen_moto_ids.add(moto_id)

        link_status = str(m.get("cross_camera_link_status") or "").strip()
        if link_status == "NOT_A_MOTORCYCLE_TRANSIT":
            errors.append(f"Motorcycle {moto_id}: cannot use NOT_A_MOTORCYCLE_TRANSIT on a motorcycle entry.")
            link_status = ""
        elif link_status and link_status not in CROSS_CAMERA_LINK_STATUSES:
            errors.append(f"Motorcycle {moto_id}: invalid cross_camera_link_status {link_status!r}.")
            link_status = ""
        elif mode == "submit" and not link_status:
            errors.append(f"Motorcycle {moto_id}: cross_camera_link_status must be selected before submission.")

        visible_front = bool(
            m.get("visible_front", link_status in ("LINKED_BOTH_VIEWS", "FRONT_VIEW_ONLY", "AMBIGUOUS_HUMAN_UNCERTAIN"))
        )
        visible_back = bool(
            m.get("visible_back", link_status in ("LINKED_BOTH_VIEWS", "BACK_VIEW_ONLY", "AMBIGUOUS_HUMAN_UNCERTAIN"))
        )
        if mode == "submit":
            if not visible_front and not visible_back:
                errors.append(
                    f"Motorcycle {moto_id}: must check Front View, Back View, or both (otherwise mark NOT_A_MOTORCYCLE_TRANSIT)."
                )
            elif link_status == "LINKED_BOTH_VIEWS" and not (visible_front and visible_back):
                errors.append(
                    f"Motorcycle {moto_id}: LINKED_BOTH_VIEWS requires both Front View and Back View checked."
                )
            elif link_status == "FRONT_VIEW_ONLY" and (not visible_front or visible_back):
                errors.append(
                    f"Motorcycle {moto_id}: FRONT_VIEW_ONLY requires Front View checked and Back View unchecked."
                )
            elif link_status == "BACK_VIEW_ONLY" and (visible_front or not visible_back):
                errors.append(
                    f"Motorcycle {moto_id}: BACK_VIEW_ONLY requires Back View checked and Front View unchecked."
                )

        t_start = _parse_optional_float(m.get("start_t_ref_sec"), f"Motorcycle {moto_id} start_t_ref_sec", errors)
        t_end = _parse_optional_float(m.get("end_t_ref_sec"), f"Motorcycle {moto_id} end_t_ref_sec", errors)
        if t_start is not None:
            _check_time_in_bounds(t_start, min_t_ref_sec, max_t_ref_sec, f"Motorcycle {moto_id} start_t_ref_sec", errors)
        if t_end is not None:
            _check_time_in_bounds(t_end, min_t_ref_sec, max_t_ref_sec, f"Motorcycle {moto_id} end_t_ref_sec", errors)
        if t_start is not None and t_end is not None and t_end < t_start:
            errors.append(f"Motorcycle {moto_id}: end_t_ref_sec ({t_end}) < start_t_ref_sec ({t_start}).")
        elif mode == "submit" and (t_start is None or t_end is None) and not is_non_moto:
            warnings.append(f"Motorcycle {moto_id} does not specify a presence window.")

        plate_raw = m.get("rear_plate") or {}
        plate_tier = str(plate_raw.get("visibility_tier") or "").strip()
        if plate_tier and plate_tier not in PLATE_VISIBILITY_TIERS:
            errors.append(f"Motorcycle {moto_id}: invalid plate visibility_tier {plate_tier!r}.")
            plate_tier = ""
        elif mode == "submit" and not plate_tier:
            errors.append(f"Motorcycle {moto_id}: rear_plate visibility_tier must be selected before submission.")

        plate_text = normalize_plate_text(plate_raw.get("plate_text_normalized"))
        if mode == "submit":
            if plate_tier == "PLATE_CLEAR" and not plate_text:
                errors.append(f"Motorcycle {moto_id}: PLATE_CLEAR requires non-empty normalized rear plate text.")
            if not visible_back:
                if plate_tier in ("PLATE_CLEAR", "PLATE_PARTIAL"):
                    errors.append(
                        f"Motorcycle {moto_id}: {plate_tier} is invalid when Back View is not visible."
                    )
                elif plate_tier == "PLATE_UNREADABLE_MOTION_BLUR":
                    warnings.append(
                        f"Motorcycle {moto_id}: Front-only motorcycle typically uses PLATE_NOT_VISIBLE_FRONT_ONLY."
                    )

        norm_m = {
            "moto_id": moto_id,
            "cross_camera_link_status": link_status,
            "visible_front": visible_front,
            "visible_back": visible_back,
            "start_t_ref_sec": t_start,
            "end_t_ref_sec": t_end,
            "rear_plate": {
                "visibility_tier": plate_tier,
                "plate_text_normalized": plate_text,
                "notes": str(plate_raw.get("notes") or "").strip(),
            },
            "notes": str(m.get("notes") or "").strip(),
        }
        norm_motos.append(norm_m)
        moto_by_id[moto_id] = norm_m

    norm_riders: list[dict[str, Any]] = []
    seen_rider_ids: set[str] = set()
    riders_by_moto: dict[str, list[dict[str, Any]]] = {mid: [] for mid in seen_moto_ids}
    rider_by_id: dict[str, dict[str, Any]] = {}
    rider_to_moto: dict[str, str] = {}

    for idx, r in enumerate(riders_in):
        if not isinstance(r, dict):
            errors.append(f"riders[{idx}] must be an object.")
            continue
        rider_id = str(r.get("rider_id") or f"R{idx + 1}").strip()
        moto_id = str(r.get("moto_id") or "").strip()
        if not rider_id:
            errors.append(f"riders[{idx}] missing rider_id.")
            continue
        if rider_id in seen_rider_ids:
            errors.append(f"Duplicate rider_id {rider_id!r}.")
            continue
        seen_rider_ids.add(rider_id)

        if moto_id not in seen_moto_ids:
            errors.append(f"Rider {rider_id} references unknown motorcycle {moto_id!r}.")
        role = str(r.get("role") or "").strip().upper()
        if role and role not in RIDER_ROLES:
            errors.append(f"Rider {rider_id}: invalid role {role!r}.")
            role = ""
        elif mode == "submit" and not role:
            errors.append(f"Rider {rider_id}: role must be selected before submission.")

        helmet_state = str(r.get("helmet_state") or "").strip().upper()
        if helmet_state and helmet_state not in HELMET_STATES:
            errors.append(f"Rider {rider_id}: invalid helmet_state {helmet_state!r}.")
            helmet_state = ""
        elif mode == "submit" and not helmet_state:
            errors.append(f"Rider {rider_id}: helmet_state must be selected before submission.")

        front_ev = str(r.get("front_helmet_evidence") or "NOT_VISIBLE").strip().upper()
        back_ev = str(r.get("back_helmet_evidence") or "NOT_VISIBLE").strip().upper()
        if front_ev not in HELMET_STATES:
            if front_ev:
                errors.append(f"Rider {rider_id}: invalid front_helmet_evidence {front_ev!r}.")
            front_ev = "NOT_VISIBLE"
        if back_ev not in HELMET_STATES:
            if back_ev:
                errors.append(f"Rider {rider_id}: invalid back_helmet_evidence {back_ev!r}.")
            back_ev = "NOT_VISIBLE"

        r_start = _parse_optional_float(r.get("start_t_ref_sec"), f"Rider {rider_id} start_t_ref_sec", errors)
        r_end = _parse_optional_float(r.get("end_t_ref_sec"), f"Rider {rider_id} end_t_ref_sec", errors)
        if r_start is not None:
            _check_time_in_bounds(r_start, min_t_ref_sec, max_t_ref_sec, f"Rider {rider_id} start_t_ref_sec", errors)
        if r_end is not None:
            _check_time_in_bounds(r_end, min_t_ref_sec, max_t_ref_sec, f"Rider {rider_id} end_t_ref_sec", errors)
        if r_start is not None and r_end is not None and r_end < r_start:
            errors.append(f"Rider {rider_id}: end_t_ref_sec ({r_end}) < start_t_ref_sec ({r_start}).")

        norm_r = {
            "rider_id": rider_id,
            "moto_id": moto_id,
            "role": role,
            "helmet_state": helmet_state,
            "v1_1_helmet_label": HELMET_TO_V1_1_LABEL.get(helmet_state, ""),
            "front_helmet_evidence": front_ev,
            "back_helmet_evidence": back_ev,
            "is_uncertain": bool(r.get("is_uncertain", helmet_state in ("UNCERTAIN", "NOT_VISIBLE"))),
            "start_t_ref_sec": r_start,
            "end_t_ref_sec": r_end,
            "notes": str(r.get("notes") or "").strip(),
        }
        norm_riders.append(norm_r)
        if moto_id in riders_by_moto:
            riders_by_moto[moto_id].append(norm_r)
        rider_by_id[rider_id] = norm_r
        rider_to_moto[rider_id] = moto_id

    norm_events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    events_by_moto: dict[str, list[dict[str, Any]]] = {mid: [] for mid in seen_moto_ids}

    for idx, ev in enumerate(events_in):
        if not isinstance(ev, dict):
            errors.append(f"violation_events[{idx}] must be an object.")
            continue
        event_id = str(ev.get("event_id") or f"E{idx + 1}").strip()
        if not event_id:
            errors.append(f"violation_events[{idx}] missing event_id.")
            continue
        if event_id in seen_event_ids:
            errors.append(f"Duplicate event_id {event_id!r}.")
            continue
        seen_event_ids.add(event_id)

        moto_id = str(ev.get("moto_id") or "").strip()
        if moto_id not in seen_moto_ids:
            errors.append(f"Event {event_id} references unknown motorcycle {moto_id!r}.")

        rider_ids = [str(rid).strip() for rid in (ev.get("rider_ids") or []) if str(rid).strip()]
        for rid in rider_ids:
            if rid not in seen_rider_ids:
                errors.append(f"Event {event_id} references unknown rider {rid!r}.")
            elif moto_id and rider_to_moto.get(rid) != moto_id:
                errors.append(
                    f"Event {event_id} links rider {rid!r} (belongs to {rider_to_moto.get(rid)!r}) "
                    f"to motorcycle {moto_id!r}."
                )

        ev_type = str(ev.get("event_type") or "").strip().upper()
        if ev_type and ev_type not in VIOLATION_EVENT_TYPES:
            errors.append(f"Event {event_id}: invalid event_type {ev_type!r}.")
            ev_type = ""
        elif mode == "submit" and not ev_type:
            errors.append(f"Event {event_id}: event_type must be selected before submission.")

        s_t = _parse_optional_float(ev.get("start_t_ref_sec"), f"Event {event_id} start_t_ref_sec", errors)
        e_t = _parse_optional_float(ev.get("end_t_ref_sec"), f"Event {event_id} end_t_ref_sec", errors)
        if s_t is None or e_t is None:
            if mode == "submit" or (s_t is not None or e_t is not None):
                errors.append(f"Event {event_id} requires numeric start_t_ref_sec and end_t_ref_sec.")
        else:
            _check_time_in_bounds(s_t, min_t_ref_sec, max_t_ref_sec, f"Event {event_id} start_t_ref_sec", errors)
            _check_time_in_bounds(e_t, min_t_ref_sec, max_t_ref_sec, f"Event {event_id} end_t_ref_sec", errors)
            if e_t < s_t:
                errors.append(f"Event {event_id}: end_t_ref_sec ({e_t}) < start_t_ref_sec ({s_t}).")

        cam_ev = str(ev.get("camera_evidence") or "").strip().upper()
        if cam_ev and cam_ev not in CAMERA_EVIDENCE_CHOICES:
            errors.append(f"Event {event_id}: invalid camera_evidence {cam_ev!r}.")
            cam_ev = ""
        elif mode == "submit" and not cam_ev:
            errors.append(f"Event {event_id}: camera_evidence must be selected before submission.")

        key_t = _parse_optional_float(ev.get("keyframe_ref_t_sec"), f"Event {event_id} keyframe_ref_t_sec", errors)
        if key_t is not None:
            _check_time_in_bounds(key_t, min_t_ref_sec, max_t_ref_sec, f"Event {event_id} keyframe_ref_t_sec", errors)
            if s_t is not None and e_t is not None and (key_t < s_t - TIMELINE_TOLERANCE_SEC or key_t > e_t + TIMELINE_TOLERANCE_SEC):
                errors.append(
                    f"Event {event_id}: keyframe_ref_t_sec ({key_t:.3f}s) must lie within event window [{s_t:.3f}s, {e_t:.3f}s]."
                )

        norm_ev = {
            "event_id": event_id,
            "moto_id": moto_id,
            "rider_ids": rider_ids,
            "event_type": ev_type,
            "start_t_ref_sec": s_t,
            "end_t_ref_sec": e_t,
            "is_point_event": bool(s_t is not None and e_t is not None and abs(e_t - s_t) < 1e-6),
            "front_start_pts_sec": round(t_front_from_ref(s_t, global_offset_sec), 6) if s_t is not None else None,
            "front_end_pts_sec": round(t_front_from_ref(e_t, global_offset_sec), 6) if e_t is not None else None,
            "camera_evidence": cam_ev,
            "keyframe_ref_t_sec": key_t,
            "is_uncertain": bool(ev.get("is_uncertain", ev_type == "INSUFFICIENT_VISUAL_EVIDENCE")),
            "notes": str(ev.get("notes") or "").strip(),
        }
        norm_events.append(norm_ev)
        if moto_id in events_by_moto:
            events_by_moto[moto_id].append(norm_ev)

    # Submit-time structural and cross-entity consistency checks
    if mode == "submit":
        if is_non_moto:
            if len(norm_motos) > 0:
                errors.append("Item marked NOT_A_MOTORCYCLE_TRANSIT cannot also contain motorcycles.")
            if not reviewer_notes:
                errors.append("NOT_A_MOTORCYCLE_TRANSIT requires reviewer_notes explaining the scene object (e.g. bicycle/pedestrian).")
        else:
            if len(norm_motos) == 0:
                errors.append("At least one motorcycle must be annotated before submission (or mark NOT_A_MOTORCYCLE_TRANSIT).")
            if len(norm_motos) == 1 and raw_link and norm_motos[0]["cross_camera_link_status"] and raw_link != norm_motos[0]["cross_camera_link_status"]:
                errors.append(
                    f"Item Cross-Camera Link Status ({raw_link}) disagrees with Motorcycle {norm_motos[0]['moto_id']} Link Status ({norm_motos[0]['cross_camera_link_status']})."
                )
            for m in norm_motos:
                mid = m["moto_id"]
                m_riders = riders_by_moto.get(mid, [])
                m_events = events_by_moto.get(mid, [])
                if len(m_riders) == 0:
                    errors.append(f"Motorcycle {mid} has 0 riders. Add at least one rider.")
                else:
                    driver_count = sum(1 for r in m_riders if r["role"] == "DRIVER")
                    if driver_count > 1:
                        errors.append(f"Motorcycle {mid} has {driver_count} DRIVER riders (at most 1 expected).")
                    if len(m_events) == 0:
                        errors.append(f"Motorcycle {mid} has 0 events. Record a temporal window or assessment event.")

                    driver_no_helmet = any(r["role"] == "DRIVER" and r["helmet_state"] == "NO_HELMET" for r in m_riders)
                    pillion_no_helmet = any(r["role"] == "PILLION" and r["helmet_state"] == "NO_HELMET" for r in m_riders)
                    no_helmet_count = sum(1 for r in m_riders if r["helmet_state"] == "NO_HELMET")

                    for ev in m_events:
                        etype = ev["event_type"]
                        ev_cam = ev["camera_evidence"]
                        if not m["visible_front"] and ev_cam in ("FRONT_ONLY", "BOTH_CAMERAS"):
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} camera_evidence ({ev_cam}) conflicts with Front View unchecked."
                            )
                        if not m["visible_back"] and ev_cam in ("BACK_ONLY", "BOTH_CAMERAS"):
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} camera_evidence ({ev_cam}) conflicts with Back View unchecked."
                            )
                        if len(ev["rider_ids"]) == 0:
                            errors.append(f"Motorcycle {mid}: Event {ev['event_id']} must link at least one rider.")
                            continue

                        ev_riders = [rider_by_id[rid] for rid in ev["rider_ids"] if rid in rider_by_id]
                        ev_driver_nh = any(r["role"] in ("DRIVER", "UNCERTAIN") and r["helmet_state"] == "NO_HELMET" for r in ev_riders)
                        ev_pillion_nh = any(r["role"] in ("PILLION", "UNCERTAIN") and r["helmet_state"] == "NO_HELMET" for r in ev_riders)
                        ev_nh_count = sum(1 for r in ev_riders if r["helmet_state"] == "NO_HELMET")

                        if no_helmet_count > 0 and etype == "NO_VIOLATION":
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} is NO_VIOLATION while a rider has helmet_state NO_HELMET."
                            )
                        if etype == "DRIVER_NO_HELMET" and (not ev_driver_nh or (not driver_no_helmet and not any(r["role"] == "UNCERTAIN" and r["helmet_state"] == "NO_HELMET" for r in m_riders))):
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} (DRIVER_NO_HELMET) must link a DRIVER (or UNCERTAIN role) rider marked NO_HELMET."
                            )
                        if etype == "PILLION_NO_HELMET" and (not ev_pillion_nh or (not pillion_no_helmet and not any(r["role"] == "UNCERTAIN" and r["helmet_state"] == "NO_HELMET" for r in m_riders))):
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} (PILLION_NO_HELMET) must link a PILLION (or UNCERTAIN role) rider marked NO_HELMET."
                            )
                        if etype == "BOTH_NO_HELMET":
                            if len(m_riders) < 2 or no_helmet_count < 2 or ev_nh_count < 2:
                                errors.append(
                                    f"Motorcycle {mid}: Event {ev['event_id']} (BOTH_NO_HELMET) requires >=2 linked NO_HELMET riders (driver + pillion)."
                                )
                            elif not (
                                (driver_no_helmet and pillion_no_helmet)
                                or (any(r["role"] == "UNCERTAIN" and r["helmet_state"] == "NO_HELMET" for r in ev_riders) and ev_nh_count >= 2)
                            ):
                                errors.append(
                                    f"Motorcycle {mid}: Event {ev['event_id']} (BOTH_NO_HELMET) requires DRIVER + PILLION (or UNCERTAIN) riders marked NO_HELMET."
                                )
                        if etype == "INSUFFICIENT_VISUAL_EVIDENCE" and ev_nh_count > 0:
                            errors.append(
                                f"Motorcycle {mid}: Event {ev['event_id']} (INSUFFICIENT_VISUAL_EVIDENCE) cannot link a rider already marked NO_HELMET."
                            )
    else:
        if is_non_moto and not reviewer_notes:
            warnings.append("Add a short note before submitting NOT_A_MOTORCYCLE_TRANSIT.")

    norm_snapshots: list[dict[str, Any]] = []
    for snap in snapshots_in:
        if isinstance(snap, dict) and snap.get("snapshot_id"):
            norm_snapshots.append(
                {
                    "snapshot_id": str(snap["snapshot_id"]),
                    "camera": str(snap.get("camera") or "back"),
                    "t_ref_sec": _as_float(snap.get("t_ref_sec")),
                    "pts_norm_sec": _as_float(snap.get("pts_norm_sec")),
                    "frame_index": int(snap["frame_index"]) if snap.get("frame_index") is not None else None,
                    "moto_id": snap.get("moto_id"),
                    "note": str(snap.get("note") or "").strip(),
                    "relpath": snap.get("relpath"),
                    "created_at": snap.get("created_at"),
                }
            )

    summary = derive_item_summary(
        item_link_status=raw_link,
        is_non_moto=is_non_moto,
        motorcycles=norm_motos,
        riders=norm_riders,
        events=norm_events,
    )

    normalized_payload = {
        "workqueue_id": workqueue_id,
        "pair_id": pair_id,
        "item_link_status": raw_link,
        "is_non_motorcycle_transit": is_non_moto,
        "uncertainty": {
            "is_uncertain": is_uncertain,
            "reasons": unc_reasons,
        },
        "reviewer_notes": reviewer_notes,
        "motorcycles": norm_motos,
        "riders": norm_riders,
        "violation_events": norm_events,
        "snapshots": norm_snapshots,
        "annotation_elapsed_sec": _as_float(payload.get("annotation_elapsed_sec")) or 0.0,
    }

    return ValidationResult(
        ok=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        normalized_payload=normalized_payload,
        summary=summary,
    )


def derive_item_summary(
    *,
    item_link_status: str,
    is_non_moto: bool,
    motorcycles: list[dict[str, Any]],
    riders: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute canonical v1.1 summary columns compatible with blind_annotation_workqueue.csv."""
    if is_non_moto:
        return {
            "link_status": "NOT_A_MOTORCYCLE_TRANSIT",
            "rider_count": 0,
            "driver_helmet": "",
            "pillion_helmet": "",
            "violation_event_type": "NO_VIOLATION",
            "binary_violation": False,
            "violation_window_ref_sec": "",
            "plate_visibility_tier": "PLATE_NOT_VISIBLE_FRONT_ONLY",
            "plate_text_normalized": "",
        }

    # Select primary motorcycle aligned with highest-severity event so summary fields always agree
    primary_moto = motorcycles[0] if motorcycles else None
    if len(motorcycles) > 1 and events:
        ranked_motos: dict[str, int] = {}
        for ev in events:
            mid = ev.get("moto_id", "")
            ranked_motos[mid] = max(ranked_motos.get(mid, 0), EVENT_SEVERITY_ORDER.get(ev.get("event_type", ""), 0))
        if ranked_motos:
            best_mid = max(motorcycles, key=lambda m: ranked_motos.get(m["moto_id"], 0))
            primary_moto = best_mid

    link_status = (primary_moto["cross_camera_link_status"] if primary_moto else "") or item_link_status

    moto_riders = (
        [r for r in riders if r["moto_id"] == primary_moto["moto_id"]]
        if primary_moto
        else riders
    )
    moto_events = (
        [ev for ev in events if ev["moto_id"] == primary_moto["moto_id"]]
        if primary_moto
        else events
    )
    drivers = [r for r in moto_riders if r["role"] == "DRIVER"]
    pillions = [r for r in moto_riders if r["role"] == "PILLION"]

    driver_helmet = _aggregate_helmet_labels([r.get("v1_1_helmet_label", "") for r in drivers])
    if not driver_helmet and moto_riders:
        # When rider role is UNCERTAIN and no DRIVER row exists, use deterministic worst-case aggregation
        driver_helmet = _aggregate_helmet_labels([r.get("v1_1_helmet_label", "") for r in moto_riders if r["role"] == "UNCERTAIN"])
    pillion_helmet = _aggregate_helmet_labels([r.get("v1_1_helmet_label", "") for r in pillions])

    # Aggregate violation event precedence strictly across the selected primary motorcycle
    violation_types = {ev["event_type"] for ev in moto_events if ev.get("event_type")}
    if "BOTH_NO_HELMET" in violation_types or (
        "DRIVER_NO_HELMET" in violation_types and "PILLION_NO_HELMET" in violation_types
    ):
        primary_ev_type = "BOTH_NO_HELMET"
    elif "DRIVER_NO_HELMET" in violation_types:
        primary_ev_type = "DRIVER_NO_HELMET"
    elif "PILLION_NO_HELMET" in violation_types:
        primary_ev_type = "PILLION_NO_HELMET"
    elif "INSUFFICIENT_VISUAL_EVIDENCE" in violation_types and "NO_VIOLATION" not in violation_types:
        primary_ev_type = "INSUFFICIENT_VISUAL_EVIDENCE"
    elif "NO_VIOLATION" in violation_types:
        primary_ev_type = "NO_VIOLATION"
    else:
        primary_ev_type = ""

    binary_violation: bool | None
    if primary_ev_type in ("DRIVER_NO_HELMET", "PILLION_NO_HELMET", "BOTH_NO_HELMET"):
        binary_violation = True
    elif primary_ev_type == "NO_VIOLATION":
        binary_violation = False
    else:
        binary_violation = None

    win_str = ""
    viol_events = [
        ev
        for ev in moto_events
        if ev["event_type"] in ("DRIVER_NO_HELMET", "PILLION_NO_HELMET", "BOTH_NO_HELMET")
        and ev.get("start_t_ref_sec") is not None
        and ev.get("end_t_ref_sec") is not None
    ]
    target_ev = viol_events[0] if viol_events else (moto_events[0] if moto_events else None)
    if target_ev and target_ev.get("start_t_ref_sec") is not None and target_ev.get("end_t_ref_sec") is not None:
        win_str = f"{target_ev['start_t_ref_sec']:.3f}..{target_ev['end_t_ref_sec']:.3f}"

    plate_tier = (primary_moto["rear_plate"]["visibility_tier"] if primary_moto else "") or ""
    plate_text = (primary_moto["rear_plate"].get("plate_text_normalized") or "") if primary_moto else ""

    return {
        "link_status": link_status,
        "rider_count": len(moto_riders),
        "driver_helmet": driver_helmet,
        "pillion_helmet": pillion_helmet,
        "violation_event_type": primary_ev_type,
        "binary_violation": binary_violation,
        "violation_window_ref_sec": win_str,
        "plate_visibility_tier": plate_tier,
        "plate_text_normalized": plate_text,
    }


def validate_adjudication_payload(
    payload: dict[str, Any],
    *,
    min_t_ref_sec: float | None = None,
    max_t_ref_sec: float | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate adjudicator resolution payload cleanly."""
    errors: list[str] = []
    status = str(payload.get("status") or "DRAFT").strip().upper()
    if status not in ("DRAFT", "ADJUDICATED", "UNRESOLVED"):
        errors.append(f"Invalid adjudication status: {status!r}")
        status = "DRAFT"

    final_link = str(payload.get("final_link_status") or "").strip()
    if final_link and final_link not in CROSS_CAMERA_LINK_STATUSES:
        errors.append(f"final_link_status must be one of {CROSS_CAMERA_LINK_STATUSES}.")
    elif status == "ADJUDICATED" and not final_link:
        errors.append(f"final_link_status must be one of {CROSS_CAMERA_LINK_STATUSES}.")

    final_event = str(payload.get("final_event_type") or "").strip()
    if final_event and final_event not in VIOLATION_EVENT_TYPES:
        errors.append(f"final_event_type must be one of {VIOLATION_EVENT_TYPES}.")
    elif status == "ADJUDICATED" and not final_event:
        errors.append(f"final_event_type must be one of {VIOLATION_EVENT_TYPES}.")

    rider_count = payload.get("final_rider_count")
    if rider_count not in (None, ""):
        try:
            rider_count = int(rider_count)
            if rider_count < 0:
                errors.append("final_rider_count must be >= 0.")
        except (TypeError, ValueError):
            errors.append("final_rider_count must be an integer.")
            rider_count = None
    else:
        rider_count = None

    driver_helmet = str(payload.get("final_driver_helmet") or "").strip()
    if driver_helmet and driver_helmet not in ("helmet", "no_helmet", "occluded_unknown"):
        if driver_helmet in HELMET_TO_V1_1_LABEL:
            driver_helmet = HELMET_TO_V1_1_LABEL[driver_helmet]
        else:
            errors.append(f"Invalid final_driver_helmet: {driver_helmet!r}")

    pillion_helmet = str(payload.get("final_pillion_helmet") or "").strip()
    if pillion_helmet and pillion_helmet not in ("helmet", "no_helmet", "occluded_unknown"):
        if pillion_helmet in HELMET_TO_V1_1_LABEL:
            pillion_helmet = HELMET_TO_V1_1_LABEL[pillion_helmet]
        else:
            errors.append(f"Invalid final_pillion_helmet: {pillion_helmet!r}")

    plate_tier = str(payload.get("final_plate_visibility_tier") or "").strip()
    if plate_tier and plate_tier not in PLATE_VISIBILITY_TIERS:
        errors.append(f"Invalid final_plate_visibility_tier: {plate_tier!r}")
    elif status == "ADJUDICATED" and not plate_tier:
        errors.append("ADJUDICATED requires final_plate_visibility_tier.")
    plate_text = normalize_plate_text(payload.get("final_plate_text_normalized"))
    if status == "ADJUDICATED" and plate_tier == "PLATE_CLEAR" and not plate_text:
        errors.append("PLATE_CLEAR requires non-empty final_plate_text_normalized.")

    win_start = _parse_optional_float(payload.get("final_violation_start_t_ref_sec"), "final_violation_start_t_ref_sec", errors)
    win_end = _parse_optional_float(payload.get("final_violation_end_t_ref_sec"), "final_violation_end_t_ref_sec", errors)
    if (win_start is None) != (win_end is None):
        errors.append("Provide both final_violation_start_t_ref_sec and final_violation_end_t_ref_sec (or leave both blank).")
    if win_start is not None and min_t_ref_sec is not None and max_t_ref_sec is not None:
        _check_time_in_bounds(win_start, min_t_ref_sec, max_t_ref_sec, "final_violation_start_t_ref_sec", errors)
    if win_end is not None and min_t_ref_sec is not None and max_t_ref_sec is not None:
        _check_time_in_bounds(win_end, min_t_ref_sec, max_t_ref_sec, "final_violation_end_t_ref_sec", errors)
    if win_start is not None and win_end is not None and win_end < win_start:
        errors.append("final_violation_end_t_ref_sec cannot precede final_violation_start_t_ref_sec.")

    notes = str(payload.get("adjudicator_notes") or "").strip()
    reason = str(payload.get("disagreement_resolution_reason") or "").strip()
    if status in ("ADJUDICATED", "UNRESOLVED") and not (notes or reason):
        errors.append("Adjudication requires adjudicator_notes or disagreement_resolution_reason.")

    if status == "ADJUDICATED":
        if final_link == "NOT_A_MOTORCYCLE_TRANSIT":
            if rider_count not in (None, 0):
                errors.append("NOT_A_MOTORCYCLE_TRANSIT requires final_rider_count == 0 or blank.")
            if final_event != "NO_VIOLATION":
                errors.append("NOT_A_MOTORCYCLE_TRANSIT requires final_event_type == NO_VIOLATION.")
            if driver_helmet or pillion_helmet:
                errors.append("NOT_A_MOTORCYCLE_TRANSIT requires empty driver/pillion helmet labels.")
        else:
            if rider_count is None or rider_count < 1:
                errors.append("ADJUDICATED motorcycle item requires final_rider_count >= 1.")
            if final_link == "FRONT_VIEW_ONLY" and plate_tier in ("PLATE_CLEAR", "PLATE_PARTIAL"):
                errors.append(f"FRONT_VIEW_ONLY cannot have final_plate_visibility_tier={plate_tier}.")
            if final_event == "DRIVER_NO_HELMET" and driver_helmet != "no_helmet":
                errors.append("DRIVER_NO_HELMET requires final_driver_helmet == no_helmet.")
            if final_event == "PILLION_NO_HELMET" and pillion_helmet != "no_helmet":
                errors.append("PILLION_NO_HELMET requires final_pillion_helmet == no_helmet.")
            if final_event == "BOTH_NO_HELMET":
                if (rider_count or 0) < 2 or driver_helmet != "no_helmet" or pillion_helmet != "no_helmet":
                    errors.append("BOTH_NO_HELMET requires final_rider_count >= 2 and both driver & pillion == no_helmet.")
            if final_event == "NO_VIOLATION" and (driver_helmet == "no_helmet" or pillion_helmet == "no_helmet"):
                errors.append("NO_VIOLATION conflicts with a rider helmet label of no_helmet.")
            if final_event == "INSUFFICIENT_VISUAL_EVIDENCE" and (driver_helmet == "no_helmet" or pillion_helmet == "no_helmet"):
                errors.append("INSUFFICIENT_VISUAL_EVIDENCE conflicts with an already confirmed no_helmet label.")

    norm = {
        "status": status,
        "final_link_status": final_link,
        "final_rider_count": rider_count,
        "final_driver_helmet": driver_helmet,
        "final_pillion_helmet": pillion_helmet,
        "final_event_type": final_event,
        "final_violation_start_t_ref_sec": win_start,
        "final_violation_end_t_ref_sec": win_end,
        "final_plate_visibility_tier": plate_tier,
        "final_plate_text_normalized": plate_text or "",
        "adjudicator_notes": notes,
        "disagreement_resolution_reason": reason,
    }
    return (len(errors) == 0), errors, norm
