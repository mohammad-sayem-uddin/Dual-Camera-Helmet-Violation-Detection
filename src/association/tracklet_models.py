"""Structured, Versioned, JSON-Serializable Data Models for Phase 12E Cross-Camera Tracklet Association.

Separates:
1. Per-camera tracklet representation (`MotorcycleTracklet`, `RiderTracklet`, `OrphanRiderTracklet`)
   with explicit separation of observed frames (`ObservedFrameRecord`) from Kalman-predicted
   intervals (`occlusion_intervals`).
2. Synchronization and transit uncertainty provenance (`TimestampQualityRecord`) keeping
   `epsilon_sync_ms`, `delta_pts_front_ms`, `delta_pts_back_ms`, `delta_transit_sec`,
   and `transit_uncertainty_sec` strictly distinct.
3. Multi-cue scoring diagnostics (`CueScoreDetail`, `CandidatePairScore`) with explicit
   availability, quality, raw value, normalized score, and renormalized effective weight.
4. Conservative decision outcomes (`AssociationOutcome`: `LINKED`, `UNMATCHED`,
   `AMBIGUOUS_ABSTAIN`, `INSUFFICIENT_EVIDENCE`) at motorcycle, rider, and orphan levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0.0"


class AssociationOutcome(str, Enum):
    """Explicit cross-camera decision outcomes."""
    LINKED = "LINKED"
    UNMATCHED = "UNMATCHED"
    AMBIGUOUS_ABSTAIN = "AMBIGUOUS_ABSTAIN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TrackletCompleteness(str, Enum):
    """Completeness classification of a single-camera tracklet."""
    COMPLETE = "COMPLETE"
    SHORT_OBSERVATION = "SHORT_OBSERVATION"
    FRAGMENTED_TENTATIVE = "FRAGMENTED_TENTATIVE"
    HEAVILY_OCCLUDED = "HEAVILY_OCCLUDED"
    MID_FRAME_TRUNCATED = "MID_FRAME_TRUNCATED"


@dataclass
class TimestampQualityRecord:
    """
    Preserves Phase 11.6 / Phase 12C synchronization uncertainty separately from
    physical cross-camera transit time uncertainty.
    """
    pair_id: str = "unknown_pair"
    reference_clock_anchor: str = "back_stream_pts"
    global_offset_sec: float = 0.0
    epsilon_sync_ms: float = 0.0
    mean_delta_pts_front_ms: float = 0.0
    max_delta_pts_front_ms: float = 0.0
    mean_delta_pts_back_ms: float = 0.0
    max_delta_pts_back_ms: float = 0.0
    is_in_overlap: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "reference_clock_anchor": self.reference_clock_anchor,
            "global_offset_sec": round(float(self.global_offset_sec), 6),
            "epsilon_sync_ms": round(float(self.epsilon_sync_ms), 4),
            "mean_delta_pts_front_ms": round(float(self.mean_delta_pts_front_ms), 4),
            "max_delta_pts_front_ms": round(float(self.max_delta_pts_front_ms), 4),
            "mean_delta_pts_back_ms": round(float(self.mean_delta_pts_back_ms), 4),
            "max_delta_pts_back_ms": round(float(self.max_delta_pts_back_ms), 4),
            "is_in_overlap": bool(self.is_in_overlap),
        }


@dataclass
class ObservedFrameRecord:
    """
    Single observed (non-predicted) physical frame record within a tracklet.
    Purely Kalman-predicted frames are never stored here.
    """
    frame_index: int
    camera_pts_sec: float
    canonical_pts_sec: float
    bbox_xyxy: tuple[float, float, float, float]
    kalman_bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    matched_in_stage2: bool = False
    observed_rider_count: int = 0
    person_bbox_xyxy: tuple[float, float, float, float] | None = None
    person_confidence: float | None = None
    head_class_name: str | None = None
    slot_role: str | None = None
    ordering_confident: bool = True
    delta_pts_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "camera_pts_sec": round(float(self.camera_pts_sec), 6),
            "canonical_pts_sec": round(float(self.canonical_pts_sec), 6),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "kalman_bbox_xyxy": [round(float(c), 2) for c in self.kalman_bbox_xyxy],
            "confidence": round(float(self.confidence), 4),
            "velocity_xy": [round(float(v), 4) for v in self.velocity_xy],
            "matched_in_stage2": bool(self.matched_in_stage2),
            "observed_rider_count": int(self.observed_rider_count),
            "person_bbox_xyxy": (
                [round(float(c), 2) for c in self.person_bbox_xyxy]
                if self.person_bbox_xyxy is not None
                else None
            ),
            "person_confidence": (
                round(float(self.person_confidence), 4)
                if self.person_confidence is not None
                else None
            ),
            "head_class_name": self.head_class_name,
            "slot_role": self.slot_role,
            "ordering_confident": bool(self.ordering_confident),
            "delta_pts_ms": round(float(self.delta_pts_ms), 4),
        }


@dataclass
class AppearanceDescriptorRecord:
    """
    Optional viewpoint-robust appearance descriptor extracted strictly from observed
    frames passing minimum crop resolution and sharpness thresholds.
    Never encodes helmet vs no-helmet class predictions.
    """
    available: bool = False
    unavailable_reason: str | None = "NO_CROPS_PROVIDED"
    valid_observed_crops_count: int = 0
    mean_crop_quality: float = 0.0
    cranial_cap_histogram: list[float] = field(default_factory=list)
    upper_torso_histogram: list[float] = field(default_factory=list)
    moto_chassis_histogram: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "valid_observed_crops_count": int(self.valid_observed_crops_count),
            "mean_crop_quality": round(float(self.mean_crop_quality), 4),
            "cranial_cap_bins": len(self.cranial_cap_histogram),
            "upper_torso_bins": len(self.upper_torso_histogram),
            "moto_chassis_bins": len(self.moto_chassis_histogram),
        }


@dataclass
class RiderTracklet:
    """
    Structured tracklet summary for an individual `RiderSubTrack` bound to a parent
    `MotorcycleTracklet`. Preserves per-camera Phase 10 verification state read-only.
    """
    schema_version: str
    camera_id: str
    subtrack_id: str
    parent_moto_track_id: str
    slot_index: int
    slot_role: str
    ordering_confident: bool
    t_start_sec: float
    t_end_sec: float
    t_centroid_sec: float
    camera_pts_start_sec: float
    camera_pts_end_sec: float
    observed_frame_count: int
    total_span_frames: int
    missed_frame_count: int
    observation_density: float
    occlusion_intervals: list[tuple[int, int]]
    entry_head_bbox: tuple[float, float, float, float]
    exit_head_bbox: tuple[float, float, float, float]
    mean_head_confidence: float
    mean_coupling_score: float
    abstained_updates_count: int
    # Read-only snapshot of per-camera Phase 10 state (never used as an identity cue
    # and never merged or mutated by Phase 12E):
    phase10_temporal_state: str
    phase10_aggregated_confidence: dict[str, float]
    phase10_consecutive_no_helmet_frames: int
    phase10_consecutive_helmet_frames: int
    phase10_alerts_count: int
    completeness: TrackletCompleteness
    observed_frames: list[ObservedFrameRecord] = field(default_factory=list)
    appearance_descriptor: AppearanceDescriptorRecord = field(
        default_factory=AppearanceDescriptorRecord
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "subtrack_id": self.subtrack_id,
            "parent_moto_track_id": self.parent_moto_track_id,
            "slot_index": int(self.slot_index),
            "slot_role": self.slot_role,
            "ordering_confident": bool(self.ordering_confident),
            "t_start_sec": round(float(self.t_start_sec), 6),
            "t_end_sec": round(float(self.t_end_sec), 6),
            "t_centroid_sec": round(float(self.t_centroid_sec), 6),
            "camera_pts_start_sec": round(float(self.camera_pts_start_sec), 6),
            "camera_pts_end_sec": round(float(self.camera_pts_end_sec), 6),
            "observed_frame_count": int(self.observed_frame_count),
            "total_span_frames": int(self.total_span_frames),
            "missed_frame_count": int(self.missed_frame_count),
            "observation_density": round(float(self.observation_density), 4),
            "occlusion_intervals": [[int(a), int(b)] for a, b in self.occlusion_intervals],
            "entry_head_bbox": [round(float(c), 2) for c in self.entry_head_bbox],
            "exit_head_bbox": [round(float(c), 2) for c in self.exit_head_bbox],
            "mean_head_confidence": round(float(self.mean_head_confidence), 4),
            "mean_coupling_score": round(float(self.mean_coupling_score), 4),
            "abstained_updates_count": int(self.abstained_updates_count),
            "phase10_snapshot_readonly": {
                "temporal_state": self.phase10_temporal_state,
                "aggregated_confidence": {
                    k: round(float(v), 4) for k, v in self.phase10_aggregated_confidence.items()
                },
                "consecutive_no_helmet_frames": int(self.phase10_consecutive_no_helmet_frames),
                "consecutive_helmet_frames": int(self.phase10_consecutive_helmet_frames),
                "alerts_count": int(self.phase10_alerts_count),
            },
            "completeness": self.completeness.value,
            "appearance_descriptor": self.appearance_descriptor.to_dict(),
        }


@dataclass
class OrphanRiderTracklet:
    """
    First-class structured tracklet summary for an `OrphanRiderTrack` whose parent
    motorcycle was not reliably detected or coupled in that camera.
    """
    schema_version: str
    camera_id: str
    orphan_track_id: str
    lifecycle_state: str
    last_rejection_reason: str
    t_start_sec: float
    t_end_sec: float
    t_centroid_sec: float
    camera_pts_start_sec: float
    camera_pts_end_sec: float
    observed_frame_count: int
    total_span_frames: int
    missed_frame_count: int
    observation_density: float
    occlusion_intervals: list[tuple[int, int]]
    entry_head_bbox: tuple[float, float, float, float]
    exit_head_bbox: tuple[float, float, float, float]
    mean_velocity_xy: tuple[float, float]
    mean_head_confidence: float
    phase10_temporal_state: str
    phase10_aggregated_confidence: dict[str, float]
    phase10_alerts_count: int
    completeness: TrackletCompleteness
    timestamp_quality: TimestampQualityRecord = field(default_factory=TimestampQualityRecord)
    observed_frames: list[ObservedFrameRecord] = field(default_factory=list)
    appearance_descriptor: AppearanceDescriptorRecord = field(
        default_factory=AppearanceDescriptorRecord
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "orphan_track_id": self.orphan_track_id,
            "lifecycle_state": self.lifecycle_state,
            "last_rejection_reason": self.last_rejection_reason,
            "t_start_sec": round(float(self.t_start_sec), 6),
            "t_end_sec": round(float(self.t_end_sec), 6),
            "t_centroid_sec": round(float(self.t_centroid_sec), 6),
            "camera_pts_start_sec": round(float(self.camera_pts_start_sec), 6),
            "camera_pts_end_sec": round(float(self.camera_pts_end_sec), 6),
            "observed_frame_count": int(self.observed_frame_count),
            "total_span_frames": int(self.total_span_frames),
            "missed_frame_count": int(self.missed_frame_count),
            "observation_density": round(float(self.observation_density), 4),
            "occlusion_intervals": [[int(a), int(b)] for a, b in self.occlusion_intervals],
            "entry_head_bbox": [round(float(c), 2) for c in self.entry_head_bbox],
            "exit_head_bbox": [round(float(c), 2) for c in self.exit_head_bbox],
            "mean_velocity_xy": [round(float(v), 4) for v in self.mean_velocity_xy],
            "mean_head_confidence": round(float(self.mean_head_confidence), 4),
            "phase10_snapshot_readonly": {
                "temporal_state": self.phase10_temporal_state,
                "aggregated_confidence": {
                    k: round(float(v), 4) for k, v in self.phase10_aggregated_confidence.items()
                },
                "alerts_count": int(self.phase10_alerts_count),
            },
            "completeness": self.completeness.value,
            "timestamp_quality": self.timestamp_quality.to_dict(),
            "appearance_descriptor": self.appearance_descriptor.to_dict(),
        }


@dataclass
class MotorcycleTracklet:
    """
    Complete per-camera motorcycle tracklet (`MotorcycleTracklet`) with observed vs.
    Kalman-predicted separation, rider cardinality distribution, entry/exit kinematics,
    and child `RiderTracklet`s.
    """
    schema_version: str
    camera_id: str
    moto_track_id: str
    lifecycle_state: str
    ever_confirmed: bool
    t_start_sec: float
    t_end_sec: float
    t_peak_sec: float
    t_centroid_sec: float
    camera_pts_start_sec: float
    camera_pts_end_sec: float
    first_frame_index: int
    last_frame_index: int
    observed_frame_count: int
    predicted_only_frame_count: int
    total_span_frames: int
    observation_density: float
    stage2_low_conf_matches: int
    recovery_count: int
    occlusion_intervals: list[tuple[int, int]]
    entry_bbox: tuple[float, float, float, float]
    exit_bbox: tuple[float, float, float, float]
    peak_area_bbox: tuple[float, float, float, float]
    entry_velocity_xy: tuple[float, float]
    exit_velocity_xy: tuple[float, float]
    mean_velocity_xy: tuple[float, float]
    motion_direction_unit_xy: tuple[float, float]
    mean_detection_confidence: float
    entry_near_border: bool
    exit_near_border: bool
    rider_count_distribution: dict[int, int]
    modal_rider_count: int
    max_rider_count: int
    rider_subtrack_count: int
    slot_spatial_conflicts_detected: int
    rider_abstained_updates: int
    spatial_identity_overrides: int
    completeness: TrackletCompleteness
    completeness_reasons: list[str]
    timestamp_quality: TimestampQualityRecord = field(default_factory=TimestampQualityRecord)
    rider_tracklets: dict[int, RiderTracklet] = field(default_factory=dict)
    observed_frames: list[ObservedFrameRecord] = field(default_factory=list)
    appearance_descriptor: AppearanceDescriptorRecord = field(
        default_factory=AppearanceDescriptorRecord
    )
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "moto_track_id": self.moto_track_id,
            "lifecycle_state": self.lifecycle_state,
            "ever_confirmed": bool(self.ever_confirmed),
            "t_start_sec": round(float(self.t_start_sec), 6),
            "t_end_sec": round(float(self.t_end_sec), 6),
            "t_peak_sec": round(float(self.t_peak_sec), 6),
            "t_centroid_sec": round(float(self.t_centroid_sec), 6),
            "camera_pts_start_sec": round(float(self.camera_pts_start_sec), 6),
            "camera_pts_end_sec": round(float(self.camera_pts_end_sec), 6),
            "first_frame_index": int(self.first_frame_index),
            "last_frame_index": int(self.last_frame_index),
            "observed_frame_count": int(self.observed_frame_count),
            "predicted_only_frame_count": int(self.predicted_only_frame_count),
            "total_span_frames": int(self.total_span_frames),
            "observation_density": round(float(self.observation_density), 4),
            "stage2_low_conf_matches": int(self.stage2_low_conf_matches),
            "recovery_count": int(self.recovery_count),
            "occlusion_intervals": [[int(a), int(b)] for a, b in self.occlusion_intervals],
            "entry_bbox": [round(float(c), 2) for c in self.entry_bbox],
            "exit_bbox": [round(float(c), 2) for c in self.exit_bbox],
            "peak_area_bbox": [round(float(c), 2) for c in self.peak_area_bbox],
            "entry_velocity_xy": [round(float(v), 4) for v in self.entry_velocity_xy],
            "exit_velocity_xy": [round(float(v), 4) for v in self.exit_velocity_xy],
            "mean_velocity_xy": [round(float(v), 4) for v in self.mean_velocity_xy],
            "motion_direction_unit_xy": [round(float(v), 4) for v in self.motion_direction_unit_xy],
            "mean_detection_confidence": round(float(self.mean_detection_confidence), 4),
            "entry_near_border": bool(self.entry_near_border),
            "exit_near_border": bool(self.exit_near_border),
            "rider_count_distribution": {int(k): int(v) for k, v in self.rider_count_distribution.items()},
            "modal_rider_count": int(self.modal_rider_count),
            "max_rider_count": int(self.max_rider_count),
            "rider_subtrack_count": int(self.rider_subtrack_count),
            "slot_spatial_conflicts_detected": int(self.slot_spatial_conflicts_detected),
            "rider_abstained_updates": int(self.rider_abstained_updates),
            "spatial_identity_overrides": int(self.spatial_identity_overrides),
            "completeness": self.completeness.value,
            "completeness_reasons": list(self.completeness_reasons),
            "timestamp_quality": self.timestamp_quality.to_dict(),
            "rider_tracklets": {
                int(s): rt.to_dict() for s, rt in sorted(self.rider_tracklets.items())
            },
            "appearance_descriptor": self.appearance_descriptor.to_dict(),
            "source_metadata": dict(self.source_metadata),
        }


@dataclass
class CueScoreDetail:
    """
    Detailed diagnostic record for a single association cue (Cue A..E).
    Never turns an unavailable cue into a silent zero penalty.
    """
    cue_name: str
    available: bool
    unavailable_reason: str | None
    quality: float
    raw_value: Any
    normalized_score: float | None
    nominal_weight: float
    effective_weight: float
    weighted_contribution: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_name": self.cue_name,
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "quality": round(float(self.quality), 4),
            "raw_value": self.raw_value,
            "normalized_score": (
                round(float(self.normalized_score), 4)
                if self.normalized_score is not None
                else None
            ),
            "nominal_weight": round(float(self.nominal_weight), 4),
            "effective_weight": round(float(self.effective_weight), 4),
            "weighted_contribution": round(float(self.weighted_contribution), 4),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class CandidatePairScore:
    """
    Complete multi-cue evaluation of a candidate `(front_tracklet, back_tracklet)` pair.
    """
    front_moto_track_id: str
    back_moto_track_id: str
    passed_candidate_gate: bool
    gate_rejection_reasons: list[str]
    composite_score: float
    active_cues_count: int
    available_cue_names: list[str]
    unavailable_cue_names: list[str]
    cue_details: dict[str, CueScoreDetail]
    # Explicit uncertainty separation required by Section 4 & Section 12:
    epsilon_sync_ms: float
    delta_pts_front_ms: float
    delta_pts_back_ms: float
    delta_transit_sec: float
    transit_uncertainty_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_moto_track_id": self.front_moto_track_id,
            "back_moto_track_id": self.back_moto_track_id,
            "passed_candidate_gate": bool(self.passed_candidate_gate),
            "gate_rejection_reasons": list(self.gate_rejection_reasons),
            "composite_score": round(float(self.composite_score), 4),
            "active_cues_count": int(self.active_cues_count),
            "available_cue_names": list(self.available_cue_names),
            "unavailable_cue_names": list(self.unavailable_cue_names),
            "cue_details": {k: v.to_dict() for k, v in self.cue_details.items()},
            "uncertainty_separation": {
                "epsilon_sync_ms": round(float(self.epsilon_sync_ms), 4),
                "delta_pts_front_ms": round(float(self.delta_pts_front_ms), 4),
                "delta_pts_back_ms": round(float(self.delta_pts_back_ms), 4),
                "delta_transit_sec": round(float(self.delta_transit_sec), 6),
                "transit_uncertainty_sec": round(float(self.transit_uncertainty_sec), 6),
            },
        }


@dataclass
class RiderAssociationResult:
    """
    Rider-level cross-camera decision within a `LINKED` motorcycle pair.
    Preserves Front and Back `RiderSubTrack` IDs, `slot_role`s, and Phase 10 states
    independently without merging or overwriting them.
    """
    schema_version: str
    parent_global_motorcycle_id: str
    front_moto_track_id: str
    back_moto_track_id: str
    front_subtrack_id: str | None
    back_subtrack_id: str | None
    front_slot_role: str | None
    back_slot_role: str | None
    outcome: AssociationOutcome
    association_score: float
    second_best_score: float
    ambiguity_margin: float
    decision_reasons: list[str]
    cue_breakdown: dict[str, float]
    front_phase10_readonly: dict[str, Any] | None = None
    back_phase10_readonly: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_global_motorcycle_id": self.parent_global_motorcycle_id,
            "front_moto_track_id": self.front_moto_track_id,
            "back_moto_track_id": self.back_moto_track_id,
            "front_subtrack_id": self.front_subtrack_id,
            "back_subtrack_id": self.back_subtrack_id,
            "front_slot_role": self.front_slot_role,
            "back_slot_role": self.back_slot_role,
            "outcome": self.outcome.value,
            "association_score": round(float(self.association_score), 4),
            "second_best_score": round(float(self.second_best_score), 4),
            "ambiguity_margin": round(float(self.ambiguity_margin), 4),
            "decision_reasons": list(self.decision_reasons),
            "cue_breakdown": {k: round(float(v), 4) for k, v in self.cue_breakdown.items()},
            "front_phase10_readonly": self.front_phase10_readonly,
            "back_phase10_readonly": self.back_phase10_readonly,
        }


@dataclass
class MotorcycleAssociationResult:
    """
    Motorcycle-level cross-camera decision (`LINKED`, `UNMATCHED`, `AMBIGUOUS_ABSTAIN`,
    or `INSUFFICIENT_EVIDENCE`).
    """
    schema_version: str
    global_motorcycle_id: str | None
    front_moto_track_id: str | None
    back_moto_track_id: str | None
    outcome: AssociationOutcome
    association_score: float
    second_best_score: float
    ambiguity_margin: float
    swap_margin_2x2: float | None
    decision_reasons: list[str]
    pair_score_detail: CandidatePairScore | None = None
    rider_associations: list[RiderAssociationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_motorcycle_id": self.global_motorcycle_id,
            "front_moto_track_id": self.front_moto_track_id,
            "back_moto_track_id": self.back_moto_track_id,
            "outcome": self.outcome.value,
            "association_score": round(float(self.association_score), 4),
            "second_best_score": round(float(self.second_best_score), 4),
            "ambiguity_margin": round(float(self.ambiguity_margin), 4),
            "swap_margin_2x2": (
                round(float(self.swap_margin_2x2), 4)
                if self.swap_margin_2x2 is not None
                else None
            ),
            "decision_reasons": list(self.decision_reasons),
            "pair_score_detail": (
                self.pair_score_detail.to_dict()
                if self.pair_score_detail is not None
                else None
            ),
            "rider_associations": [r.to_dict() for r in self.rider_associations],
        }


@dataclass
class OrphanAssociationResult:
    """
    First-class cross-camera decision for an `OrphanRiderTracklet`.
    Explicitly labeled `association_level = "ORPHAN_LEVEL_LINK"` so it never implies
    a confirmed motorcycle association.
    """
    schema_version: str
    global_orphan_link_id: str | None
    association_level: str
    front_track_id: str | None
    back_track_id: str | None
    front_track_kind: str | None
    back_track_kind: str | None
    outcome: AssociationOutcome
    association_score: float
    second_best_score: float
    ambiguity_margin: float
    decision_reasons: list[str]
    cue_breakdown: dict[str, float]
    implies_confirmed_motorcycle_link: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_orphan_link_id": self.global_orphan_link_id,
            "association_level": self.association_level,
            "front_track_id": self.front_track_id,
            "back_track_id": self.back_track_id,
            "front_track_kind": self.front_track_kind,
            "back_track_kind": self.back_track_kind,
            "outcome": self.outcome.value,
            "association_score": round(float(self.association_score), 4),
            "second_best_score": round(float(self.second_best_score), 4),
            "ambiguity_margin": round(float(self.ambiguity_margin), 4),
            "decision_reasons": list(self.decision_reasons),
            "cue_breakdown": {k: round(float(v), 4) for k, v in self.cue_breakdown.items()},
            "implies_confirmed_motorcycle_link": bool(self.implies_confirmed_motorcycle_link),
        }


@dataclass
class CrossCameraAssociationBatchResult:
    """
    Complete output of a cross-camera association run over a pair of camera streams.
    """
    schema_version: str
    pair_id: str
    timestamp_quality: TimestampQualityRecord
    front_motorcycle_tracklets: list[MotorcycleTracklet]
    back_motorcycle_tracklets: list[MotorcycleTracklet]
    front_orphan_tracklets: list[OrphanRiderTracklet]
    back_orphan_tracklets: list[OrphanRiderTracklet]
    candidate_pair_scores: list[CandidatePairScore]
    motorcycle_results: list[MotorcycleAssociationResult]
    orphan_results: list[OrphanAssociationResult]
    outcome_counts: dict[str, int]
    rider_outcome_counts: dict[str, int]
    orphan_outcome_counts: dict[str, int]
    runtime_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "timestamp_quality": self.timestamp_quality.to_dict(),
            "front_motorcycle_tracklets": [t.to_dict() for t in self.front_motorcycle_tracklets],
            "back_motorcycle_tracklets": [t.to_dict() for t in self.back_motorcycle_tracklets],
            "front_orphan_tracklets": [t.to_dict() for t in self.front_orphan_tracklets],
            "back_orphan_tracklets": [t.to_dict() for t in self.back_orphan_tracklets],
            "candidate_pair_scores": [c.to_dict() for c in self.candidate_pair_scores],
            "motorcycle_results": [m.to_dict() for m in self.motorcycle_results],
            "orphan_results": [o.to_dict() for o in self.orphan_results],
            "outcome_counts": dict(self.outcome_counts),
            "rider_outcome_counts": dict(self.rider_outcome_counts),
            "orphan_outcome_counts": dict(self.orphan_outcome_counts),
            "runtime_ms": {k: round(float(v), 4) for k, v in self.runtime_ms.items()},
        }
