"""Per-Camera Hierarchical Motorcycle and Rider Tracking Pipeline.

Integrates:
1. Motorcycle-level Kalman filtering and Two-Stage ByteTrack association (`MotoTrack`).
2. Spatial rider-to-motorcycle coupling (`MotoRiderCoupler`).
3. Independent rider subtracks (`RiderSubTrack`) and orphan rider tracks (`OrphanRiderTrack`),
   each connected to the preserved Phase 10 temporal verification engine (`Track`,
   `BaseConfidenceAggregator`, `TemporalStateMachine`, and `ViolationDecisionEngine`).
4. Disjoint per-camera track namespaces (`front_*` vs. `back_*`) and strict adherence to
   Phase 12C's `should_update_<cam>_state` frame deduplication flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from src.detection.camera_detector import FrameDetectionBundle, StructuredDetection
from src.temporal.rider_tracker import BoundingBox, Track
from src.temporal.state_machine import StateMachineConfig, TemporalState
from src.temporal.violation_decision import DecisionConfig, ViolationAlert, ViolationDecisionEngine
from src.tracking.kalman_bytetrack import (
    ByteTrackConfig,
    KalmanBoxFilter,
    TrackLifecycleState,
    bbox_iou_xyxy,
    hungarian_iou_match,
    two_stage_bytetrack_assign,
)
from src.tracking.moto_rider_coupler import (
    CoupledMotorcycleObservation,
    CoupledRiderObservation,
    CouplerConfig,
    FrameCouplingResult,
    MotoRiderCoupler,
    OrphanRiderObservation,
)
from src.tracking.rider_kinematics import (
    RelativeKinematicsConfig,
    RiderRelativeKalmanState,
    bbox_to_moto_relative,
    clamp_bbox_to_frame,
    relative_distance,
    relative_scale_consistency,
    role_name_for_rank,
    score_longitudinal_ranks,
)


@dataclass
class Phase10AdapterConfig:
    """Configuration bridging RiderSubTrack/OrphanRiderTrack to Phase 10 components."""
    aggregator_method: str = "ema"
    sliding_window_size: int = 10
    ema_alpha: float = 0.35
    state_machine_config: StateMachineConfig = field(default_factory=StateMachineConfig)
    decision_config: DecisionConfig = field(default_factory=DecisionConfig)


_HELMET_CLASSES = ("helmet", "no_helmet")
_ZERO_AC: dict[str, float] = {"helmet": 0.0, "no_helmet": 0.0}


def derive_ema_helmet_state(
    ac: dict[str, float] | None, *, tie_break_helmet: bool = False
) -> str:
    """HELMET / NO_HELMET / UNCERTAIN from an aggregated-confidence dict.

    `tie_break_helmet=True` reproduces the legacy Stage K rule exactly
    (max >= 0.5 -> HELMET if helmet >= no_helmet else NO_HELMET).
    """
    if not ac:
        return "UNCERTAIN"
    p_h = float(ac.get("helmet", 0.0))
    p_nh = float(ac.get("no_helmet", 0.0))
    if max(p_h, p_nh) < 0.5:
        return "UNCERTAIN"
    if p_h > p_nh:
        return "HELMET"
    if p_nh > p_h:
        return "NO_HELMET"
    return "HELMET" if tie_break_helmet else "UNCERTAIN"


def derive_majority_helmet_state(helmet_count: int, no_helmet_count: int) -> str:
    """Strict majority over genuine coupled observations; tie or zero -> UNCERTAIN."""
    if helmet_count > no_helmet_count:
        return "HELMET"
    if no_helmet_count > helmet_count:
        return "NO_HELMET"
    return "UNCERTAIN"


def derive_conf_weighted_helmet_state(helmet_weight: float, no_helmet_weight: float) -> str:
    """Confidence-weighted majority; exact tie or zero mass -> UNCERTAIN."""
    if helmet_weight + no_helmet_weight <= 1e-8:
        return "UNCERTAIN"
    if helmet_weight > no_helmet_weight + 1e-9:
        return "HELMET"
    if no_helmet_weight > helmet_weight + 1e-9:
        return "NO_HELMET"
    return "UNCERTAIN"


class HelmetEvidenceAccumulator:
    """O(1)-memory track-lifecycle evidence over genuine coupled head observations.

    Updated ONLY from the matched-observation path. Unmatched / Kalman-propagated /
    abstained frames never reach it, so they cannot add evidence. A repeated record
    for a frame index not strictly after the last recorded one is ignored.
    """

    def __init__(self) -> None:
        self.observations: int = 0
        self.helmet_count: int = 0
        self.no_helmet_count: int = 0
        self.helmet_conf_sum: float = 0.0
        self.no_helmet_conf_sum: float = 0.0
        self.state_switches: int = 0
        self.last_frame_index: int | None = None
        self.last_pts_sec: float | None = None
        self.last_class_name: str | None = None
        self.last_aggregated_confidence: dict[str, float] = dict(_ZERO_AC)

    def record(
        self,
        class_name: str,
        confidence: float,
        frame_index: int,
        pts_sec: float | None,
        aggregated_confidence: dict[str, float],
    ) -> bool:
        if class_name not in _HELMET_CLASSES:
            return False
        if self.last_frame_index is not None and int(frame_index) <= self.last_frame_index:
            return False  # repeated-update protection
        self.observations += 1
        if class_name == "helmet":
            self.helmet_count += 1
            self.helmet_conf_sum += float(confidence)
        else:
            self.no_helmet_count += 1
            self.no_helmet_conf_sum += float(confidence)
        if self.last_class_name is not None and class_name != self.last_class_name:
            self.state_switches += 1
        self.last_class_name = class_name
        self.last_frame_index = int(frame_index)
        self.last_pts_sec = (
            round(float(pts_sec), 6)
            if pts_sec is not None and float(pts_sec) == float(pts_sec)
            else None
        )
        self.last_aggregated_confidence = {
            "helmet": float(aggregated_confidence.get("helmet", 0.0)),
            "no_helmet": float(aggregated_confidence.get("no_helmet", 0.0)),
        }
        return True

    def export(self, ema_final_ac: dict[str, float]) -> dict[str, Any]:
        majority = derive_majority_helmet_state(self.helmet_count, self.no_helmet_count)
        return {
            "track_helmet_state": majority,
            "track_helmet_state_majority": majority,
            "track_helmet_state_conf_weighted": derive_conf_weighted_helmet_state(
                self.helmet_conf_sum, self.no_helmet_conf_sum
            ),
            "track_helmet_state_last_coupled": derive_ema_helmet_state(
                self.last_aggregated_confidence
            ),
            "track_helmet_state_ema_final": derive_ema_helmet_state(
                ema_final_ac, tie_break_helmet=True
            ),
            "last_coupled_aggregated_confidence": dict(self.last_aggregated_confidence),
            "coupled_confidence_weights": {
                "helmet": round(self.helmet_conf_sum, 6),
                "no_helmet": round(self.no_helmet_conf_sum, 6),
            },
            "coupled_observations_count": self.observations,
            "coupled_helmet_count": self.helmet_count,
            "coupled_no_helmet_count": self.no_helmet_count,
            "coupled_state_switches": self.state_switches,
            "last_coupled_frame_index": self.last_frame_index,
            "last_coupled_pts_sec": self.last_pts_sec,
        }


class RiderSubTrack:
    """
    Independent rider subtrack bound to a slot (0: driver, 1: pillion_1, 2: pillion_2)
    on a parent `MotoTrack`.

    Wraps the existing Phase 10 `Track` (which drives `BaseConfidenceAggregator` and
    `TemporalStateMachine`) and `ViolationDecisionEngine` without modifying Phase 10 logic.
    """

    def __init__(
        self,
        camera_id: str,
        parent_moto_id: str,
        numeric_track_id: int,
        slot_index: int,
        slot_role: str,
        ordering_confident: bool,
        initial_obs: CoupledRiderObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        p10_config: Phase10AdapterConfig,
        bt_config: ByteTrackConfig | None = None,
        parent_moto_bbox: tuple[float, float, float, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.parent_moto_id = parent_moto_id
        self.numeric_track_id = numeric_track_id
        self.slot_index = int(slot_index)
        self.slot_role = slot_role
        self.ordering_confident = ordering_confident
        self.slot_role_provisional = bool(getattr(initial_obs, "slot_role_provisional", False))
        self.raw_obs_slot_role = str(initial_obs.slot_role)
        self.subtrack_id = f"{parent_moto_id}_rider_s{slot_index}"
        self.p10_config = p10_config
        self.bt_config = bt_config or ByteTrackConfig()
        self.abstained_updates_count: int = 0
        self.frame_dimensions: tuple[int, int] | None = frame_dimensions
        self.last_frame_index: int = int(frame_index)
        self.unmatched_frames: int = 0
        self.is_coast_expired: bool = False
        self.coast_state: str = "ACTIVE"
        self.coasted_frames_count: int = 0
        self.off_frame_count: int = 0

        head = initial_obs.head_detection
        p10_bbox = BoundingBox(
            x1=head.x1,
            y1=head.y1,
            x2=head.x2,
            y2=head.y2,
            confidence=head.confidence,
            class_id=head.class_id,
            class_name=head.class_name,
        )
        self.phase10_track = Track(
            track_id=numeric_track_id,
            initial_bbox=p10_bbox,
            frame_index=frame_index,
            aggregator_method=p10_config.aggregator_method,
            window_size=p10_config.sliding_window_size,
            sm_config=p10_config.state_machine_config,
            ema_alpha=p10_config.ema_alpha,
        )
        self.decision_engine = ViolationDecisionEngine(config=p10_config.decision_config)
        self.kalman = KalmanBoxFilter(
            head.bbox,
            std_weight_pos=self.bt_config.kalman_std_weight_pos,
            std_weight_vel=self.bt_config.kalman_std_weight_vel,
        )
        self.rel_kinematics: RiderRelativeKalmanState | None = None
        if (
            self.bt_config.use_motorcycle_relative_kinematics
            and parent_moto_bbox is not None
        ):
            self.rel_kinematics = RiderRelativeKalmanState(
                initial_head_bbox=head.bbox,
                initial_moto_bbox=parent_moto_bbox,
                config=self._build_rel_config(),
                frame_dimensions=frame_dimensions,
            )
            self.kalman.sync_bbox_state(self.rel_kinematics.current_bbox_xyxy, vx=0.0, vy=0.0)

        self.pts_history: list[tuple[int, float, tuple[float, float, float, float], str, float]] = [
            (frame_index, pts_timestamp_sec, head.bbox, head.class_name, head.confidence)
        ]
        self.observation_log: list[dict[str, Any]] = [
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "head_bbox": tuple(float(c) for c in head.bbox),
                "head_conf": float(head.confidence),
                "head_class_name": str(head.class_name),
                "person_bbox": (
                    tuple(float(c) for c in initial_obs.person_detection.bbox)
                    if initial_obs.person_detection is not None
                    else None
                ),
                "person_conf": (
                    float(initial_obs.person_detection.confidence)
                    if initial_obs.person_detection is not None
                    else None
                ),
                "coupling_score": float(initial_obs.coupling_score),
                "slot_role": str(self.slot_role),
                "raw_obs_slot_role": str(initial_obs.slot_role),
                "ordering_confident": bool(self.ordering_confident),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        ]
        self.helmet_evidence = HelmetEvidenceAccumulator()
        self.helmet_evidence.record(
            head.class_name, head.confidence, frame_index, pts_timestamp_sec,
            self.phase10_track.aggregator.get_aggregated_confidence(),
        )
        self.missed_frames: list[int] = []
        self.last_coupling_score = initial_obs.coupling_score
        self.alerts: list[ViolationAlert] = []
        self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def _build_rel_config(self) -> RelativeKinematicsConfig:
        return RelativeKinematicsConfig(
            vel_damping=float(self.bt_config.rel_damping_factor),
            pos_alpha=float(self.bt_config.rel_pos_alpha),
            size_alpha=float(self.bt_config.rel_size_alpha),
            vel_beta=float(self.bt_config.rel_vel_beta),
            max_rel_vel=float(self.bt_config.rel_max_vel),
            clamp_non_negative=bool(self.bt_config.clamp_non_negative_coords),
        )

    def _effective_frame_dims(
        self,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        if frame_dimensions is not None:
            return frame_dimensions
        if self.frame_dimensions is not None:
            return self.frame_dimensions
        if self.bt_config.frame_width and self.bt_config.frame_height:
            return (int(self.bt_config.frame_width), int(self.bt_config.frame_height))
        return None

    def _sync_consensus_role(
        self,
        resolved_role: str,
        ordering_confident: bool,
        provisional: bool = False,
    ) -> None:
        self.slot_role = str(resolved_role)
        self.ordering_confident = bool(ordering_confident)
        self.slot_role_provisional = bool(provisional)
        for entry in self.observation_log:
            entry["slot_role"] = self.slot_role
            entry["ordering_confident"] = self.ordering_confident

    def anchor_to_parent(
        self,
        parent_moto_bbox: tuple[float, float, float, float],
        frame_dimensions: tuple[int, int] | None = None,
    ) -> tuple[float, float, float, float]:
        """Re-project normalized relative state onto freshly updated parent moto bbox."""
        if (
            self.rel_kinematics is None
            or not self.bt_config.use_motorcycle_relative_kinematics
            or (self.bt_config.finite_rider_coast_enabled and self.is_coast_expired)
        ):
            return self.kalman.current_bbox_xyxy
        dims = self._effective_frame_dims(frame_dimensions)
        self.rel_kinematics.last_moto_bbox = parent_moto_bbox
        from src.tracking.rider_kinematics import moto_relative_to_bbox

        proj, off_f = moto_relative_to_bbox(
            self.rel_kinematics.rel_tuple,
            parent_moto_bbox,
            frame_dimensions=dims,
            config=self.rel_kinematics.config,
        )
        self.rel_kinematics.current_bbox_xyxy = proj
        self.rel_kinematics.last_off_frame = off_f
        self.kalman.sync_bbox_state(proj)
        return proj

    def _evaluate_alerts(self, frame_index: int, pts_timestamp_sec: float) -> list[ViolationAlert]:
        eff_fps = frame_index / pts_timestamp_sec if pts_timestamp_sec > 1e-4 else 60.0
        new_alerts = self.decision_engine.process_tracks(
            frame_index=frame_index,
            tracks=[self.phase10_track],
            fps=max(1.0, eff_fps),
        )
        self.alerts.extend(new_alerts)
        return new_alerts

    def predict(
        self,
        dt_frames: float = 1.0,
        parent_moto_bbox: tuple[float, float, float, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> tuple[float, float, float, float]:
        if self.bt_config.finite_rider_coast_enabled and self.is_coast_expired:
            return self.kalman.current_bbox_xyxy
        if (
            self.rel_kinematics is not None
            and self.bt_config.use_motorcycle_relative_kinematics
        ):
            dims = self._effective_frame_dims(frame_dimensions)
            proj = self.rel_kinematics.predict(
                moto_bbox=parent_moto_bbox,
                dt_frames=dt_frames,
                frame_dimensions=dims,
                coasting=True,
            )
            mw = max(
                1.0,
                self.rel_kinematics.last_moto_bbox[2] - self.rel_kinematics.last_moto_bbox[0],
            )
            mh = max(
                1.0,
                self.rel_kinematics.last_moto_bbox[3] - self.rel_kinematics.last_moto_bbox[1],
            )
            self.kalman.sync_bbox_state(
                proj,
                vx=self.rel_kinematics.du * mw,
                vy=self.rel_kinematics.dv * mh,
            )
            return proj
        return self.kalman.predict(dt_frames=dt_frames)

    def update_matched(
        self,
        obs: CoupledRiderObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        preserve_slot_role: bool = False,
        parent_moto_bbox: tuple[float, float, float, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
        parent_velocity_xy: tuple[float, float] = (0.0, 0.0),
    ) -> list[ViolationAlert]:
        head = obs.head_detection
        dt = max(1.0, float(frame_index - self.last_frame_index))
        self.last_frame_index = int(frame_index)
        self.unmatched_frames = 0
        self.is_coast_expired = False
        self.coast_state = "ACTIVE"

        if (
            self.bt_config.use_motorcycle_relative_kinematics
            and parent_moto_bbox is not None
        ):
            dims = self._effective_frame_dims(frame_dimensions)
            if self.rel_kinematics is None:
                self.rel_kinematics = RiderRelativeKalmanState(
                    initial_head_bbox=head.bbox,
                    initial_moto_bbox=parent_moto_bbox,
                    config=self._build_rel_config(),
                    frame_dimensions=dims,
                )
                proj = self.rel_kinematics.current_bbox_xyxy
            else:
                proj = self.rel_kinematics.update(
                    obs_head_bbox=head.bbox,
                    moto_bbox=parent_moto_bbox,
                    frame_dimensions=dims,
                    dt_frames=dt,
                )
            mw = max(1.0, parent_moto_bbox[2] - parent_moto_bbox[0])
            mh = max(1.0, parent_moto_bbox[3] - parent_moto_bbox[1])
            comp_vx = float(parent_velocity_xy[0]) + self.rel_kinematics.du * mw
            comp_vy = float(parent_velocity_xy[1]) + self.rel_kinematics.dv * mh
            self.kalman.sync_bbox_state(proj, vx=comp_vx, vy=comp_vy)
        else:
            self.kalman.update(head.bbox)

        preserve_established_confidence = (
            self.bt_config.role_consensus_enabled
            and self.ordering_confident
            and not self.slot_role_provisional
            and not obs.ordering_confident
        )
        if not preserve_slot_role:
            if obs.ordering_confident or self.slot_role.startswith("ambiguous_slot_"):
                self.slot_role = obs.slot_role
                self.slot_role_provisional = bool(getattr(obs, "slot_role_provisional", False))
            if not preserve_established_confidence:
                self.ordering_confident = obs.ordering_confident
        else:
            # Spatial continuity overrode a conflicting or partial-visibility slot index;
            # preserve established semantic slot_role ("driver" / "pillion_1" / "pillion_2")
            if not preserve_established_confidence:
                self.ordering_confident = obs.ordering_confident
        self.last_coupling_score = obs.coupling_score

        p10_bbox = BoundingBox(
            x1=head.x1,
            y1=head.y1,
            x2=head.x2,
            y2=head.y2,
            confidence=head.confidence,
            class_id=head.class_id,
            class_name=head.class_name,
        )
        self.phase10_track.update_matched(p10_bbox, frame_index=frame_index)
        self.helmet_evidence.record(
            head.class_name, head.confidence, frame_index, pts_timestamp_sec,
            self.phase10_track.aggregator.get_aggregated_confidence(),
        )
        self.pts_history.append(
            (frame_index, pts_timestamp_sec, head.bbox, head.class_name, head.confidence)
        )
        self.observation_log.append(
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "head_bbox": tuple(float(c) for c in head.bbox),
                "head_conf": float(head.confidence),
                "head_class_name": str(head.class_name),
                "person_bbox": (
                    tuple(float(c) for c in obs.person_detection.bbox)
                    if obs.person_detection is not None
                    else None
                ),
                "person_conf": (
                    float(obs.person_detection.confidence)
                    if obs.person_detection is not None
                    else None
                ),
                "coupling_score": float(obs.coupling_score),
                "slot_role": str(self.slot_role),
                "raw_obs_slot_role": str(obs.slot_role),
                "ordering_confident": bool(self.ordering_confident),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        )
        return self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def update_unmatched(
        self,
        frame_index: int,
        parent_moto_bbox: tuple[float, float, float, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        self.missed_frames.append(int(frame_index))
        self.last_frame_index = int(frame_index)
        self.unmatched_frames += 1
        self.phase10_track.update_unmatched(frame_index=frame_index)

        if (
            self.bt_config.use_motorcycle_relative_kinematics
            and self.rel_kinematics is not None
            and parent_moto_bbox is not None
            and not (self.bt_config.finite_rider_coast_enabled and self.is_coast_expired)
        ):
            self.anchor_to_parent(parent_moto_bbox, frame_dimensions=frame_dimensions)

        if self.bt_config.finite_rider_coast_enabled:
            is_unconfirmed_single = (
                self.bt_config.unlock_slot_conflicts
                and len(self.pts_history) < max(1, int(self.bt_config.min_confirm_hits))
            )
            coast_limit = (
                max(0, int(self.bt_config.tentative_rider_coast_frames))
                if is_unconfirmed_single
                else max(0, int(self.bt_config.max_rider_coast_frames))
            )
            if self.unmatched_frames > coast_limit:
                self.is_coast_expired = True
                self.coast_state = "COAST_EXPIRED"
            else:
                self.is_coast_expired = False
                self.coasted_frames_count += 1
                if self.rel_kinematics is not None and self.rel_kinematics.last_off_frame:
                    self.coast_state = "OFF_FRAME"
                    self.off_frame_count += 1
                else:
                    self.coast_state = "COASTED"
        else:
            self.is_coast_expired = False
            self.coasted_frames_count += 1
            self.coast_state = "COASTED"

    @property
    def temporal_state(self) -> TemporalState:
        return self.phase10_track.state

    @property
    def aggregated_confidence(self) -> dict[str, float]:
        return self.phase10_track.aggregator.get_aggregated_confidence()

    @property
    def track_helmet_state(self) -> str:
        return self.helmet_evidence.export(self.aggregated_confidence)["track_helmet_state"]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.helmet_evidence.export(self.aggregated_confidence),
            "subtrack_id": self.subtrack_id,
            "parent_moto_id": self.parent_moto_id,
            "camera_id": self.camera_id,
            "slot_index": self.slot_index,
            "slot_role": self.slot_role,
            "ordering_confident": self.ordering_confident,
            "slot_role_provisional": self.slot_role_provisional,
            "coast_state": self.coast_state,
            "unmatched_frames": self.unmatched_frames,
            "is_coast_expired": self.is_coast_expired,
            "coasted_frames_count": self.coasted_frames_count,
            "off_frame_count": self.off_frame_count,
            "relative_state": (
                [round(v, 4) for v in self.rel_kinematics.rel_tuple]
                if self.rel_kinematics is not None
                else None
            ),
            "temporal_state": self.temporal_state.value,
            "hits": self.phase10_track.hits,
            "age": self.phase10_track.age,
            "time_since_update": self.phase10_track.time_since_update,
            "consecutive_no_helmet_frames": self.phase10_track.state_machine.consecutive_no_helmet_frames,
            "consecutive_helmet_frames": self.phase10_track.state_machine.consecutive_helmet_frames,
            "aggregated_confidence": self.aggregated_confidence,
            "last_coupling_score": round(self.last_coupling_score, 4),
            "bbox": [round(c, 2) for c in self.kalman.current_bbox_xyxy],
            "alerts_emitted": len(self.alerts),
            "abstained_updates_count": self.abstained_updates_count,
        }


class MotoTrack:
    """
    Per-camera motorcycle trajectory with Kalman state estimation, 5-state lifecycle
    management (`TENTATIVE`, `CONFIRMED`, `LOST`, `RECOVERED`, `TERMINATED`), and
    1..3 independent `RiderSubTrack` slots.
    """

    def __init__(
        self,
        camera_id: str,
        numeric_id: int,
        initial_coupled: CoupledMotorcycleObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        bt_config: ByteTrackConfig,
        p10_config: Phase10AdapterConfig,
        rider_id_alloc_fn: Any,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.numeric_id = numeric_id
        self.moto_track_id = f"{camera_id}_moto_{numeric_id:04d}"
        self.bt_config = bt_config
        self.p10_config = p10_config
        self._alloc_rider_id = rider_id_alloc_fn
        self.frame_dimensions: tuple[int, int] | None = frame_dimensions

        det = initial_coupled.motorcycle_detection
        self.kalman = KalmanBoxFilter(
            det.bbox,
            std_weight_pos=self.bt_config.kalman_std_weight_pos,
            std_weight_vel=self.bt_config.kalman_std_weight_vel,
        )
        self.current_bbox = det.bbox
        self.current_confidence = det.confidence

        self.age: int = 1
        self.hits: int = 1
        self.time_since_update: int = 0
        self.last_frame_index: int = frame_index
        self.first_pts_sec: float = pts_timestamp_sec
        self.last_pts_sec: float = pts_timestamp_sec
        self.stage2_low_conf_matches: int = 0
        self.recovery_count: int = 0
        self.slot_spatial_conflicts_detected: int = 0
        self.rider_abstained_updates: int = 0
        self.spatial_identity_overrides: int = 0
        self._ever_confirmed: bool = (self.hits >= self.bt_config.min_confirm_hits)

        self.lifecycle_state = (
            TrackLifecycleState.CONFIRMED
            if self._ever_confirmed
            else TrackLifecycleState.TENTATIVE
        )

        self.trajectory: list[tuple[int, float, tuple[float, float, float, float], float]] = [
            (frame_index, pts_timestamp_sec, det.bbox, det.confidence)
        ]
        self.observation_log: list[dict[str, Any]] = [
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "raw_bbox": tuple(float(c) for c in det.bbox),
                "kalman_bbox": tuple(float(c) for c in self.current_bbox),
                "confidence": float(det.confidence),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
                "matched_in_stage2": False,
                "observed_rider_count": len(initial_coupled.riders),
            }
        ]
        self.missed_frames: list[int] = []

        self.rider_subtracks: dict[int, RiderSubTrack] = {}
        self.discarded_tentative_subtracks: list[RiderSubTrack] = []
        self.role_votes: dict[int, dict[int, int]] = {}
        self.multi_rider_consensus_frames: int = 0
        self.new_alerts_this_frame: list[ViolationAlert] = []
        self._sync_riders_on_match(initial_coupled.riders, frame_index, pts_timestamp_sec, is_init=True)

    def _frame_dimensions(self) -> tuple[int, int] | None:
        if self.frame_dimensions is not None:
            return self.frame_dimensions
        if self.bt_config.frame_width and self.bt_config.frame_height:
            return (int(self.bt_config.frame_width), int(self.bt_config.frame_height))
        return None

    def _rel_config(self) -> RelativeKinematicsConfig:
        return RelativeKinematicsConfig(
            vel_damping=float(self.bt_config.rel_damping_factor),
            pos_alpha=float(self.bt_config.rel_pos_alpha),
            size_alpha=float(self.bt_config.rel_size_alpha),
            vel_beta=float(self.bt_config.rel_vel_beta),
            max_rel_vel=float(self.bt_config.rel_max_vel),
            clamp_non_negative=bool(self.bt_config.clamp_non_negative_coords),
        )

    def predict(self, current_frame_index: int) -> tuple[float, float, float, float]:
        dt = max(1, current_frame_index - self.last_frame_index)
        self.current_bbox = self.kalman.predict(dt_frames=float(dt))
        fdims = self._frame_dimensions()
        for r_sub in self.rider_subtracks.values():
            r_sub.predict(
                dt_frames=float(dt),
                parent_moto_bbox=self.current_bbox,
                frame_dimensions=fdims,
            )
        return self.current_bbox

    def _next_free_slot_key(self) -> int:
        k = 0
        while k in self.rider_subtracks:
            k += 1
        return k

    def _prune_expired_tentative_subtracks(self, force_all: bool = False) -> None:
        """Move coast-expired unconfirmed 1-hit subtracks out of active slot management."""
        if not (self.bt_config.finite_rider_coast_enabled and self.bt_config.unlock_slot_conflicts):
            return
        confirm_hits = max(1, int(self.bt_config.min_confirm_hits))
        expired_tentative = [
            k
            for k in sorted(self.rider_subtracks.keys())
            if self.rider_subtracks[k].is_coast_expired
            and len(self.rider_subtracks[k].pts_history) < confirm_hits
        ]
        if not expired_tentative:
            return
        if not force_all:
            if len(self.rider_subtracks) <= 1:
                return
            if len(expired_tentative) >= len(self.rider_subtracks):
                expired_tentative = expired_tentative[1:]
        for k in expired_tentative:
            sub = self.rider_subtracks.pop(k, None)
            if sub is not None:
                self.discarded_tentative_subtracks.append(sub)
            self.role_votes.pop(k, None)

    def _estimate_travel_velocity(self) -> tuple[float, float]:
        """Return motorcycle travel velocity, using finite-difference raw center motion during warmup."""
        kx, ky = self.kalman.velocity_xy
        kspeed = math.hypot(float(kx), float(ky))
        if kspeed >= self.bt_config.role_min_moto_speed_px:
            return (float(kx), float(ky))
        if len(self.observation_log) >= 2:
            lookback = min(3, len(self.observation_log) - 1)
            prev_e = self.observation_log[-1 - lookback]
            curr_e = self.observation_log[-1]
            dt = max(1, int(curr_e["frame_index"]) - int(prev_e["frame_index"]))
            pb = prev_e["raw_bbox"]
            cb = curr_e["raw_bbox"]
            vx = (0.5 * (cb[0] + cb[2]) - 0.5 * (pb[0] + pb[2])) / float(dt)
            vy = (0.5 * (cb[1] + cb[3]) - 0.5 * (pb[1] + pb[3])) / float(dt)
            if math.hypot(vx, vy) >= self.bt_config.role_min_moto_speed_px:
                return (vx, vy)
        return (float(kx), float(ky))

    def _reconcile_roles(
        self,
        matched_slots: set[int],
        frame_index: int,
    ) -> None:
        if not self.bt_config.role_consensus_enabled or not self.rider_subtracks:
            return

        active_slots = [
            k
            for k in sorted(self.rider_subtracks.keys())
            if not (self.bt_config.finite_rider_coast_enabled and self.rider_subtracks[k].is_coast_expired)
        ]
        matched_active = sorted(k for k in matched_slots if k in self.rider_subtracks and not self.rider_subtracks[k].is_coast_expired)

        # Accumulate longitudinal rank votes across co-observed matched riders in this frame
        if len(matched_active) >= 2:
            centroids: dict[int, tuple[float, float]] = {}
            for k in matched_active:
                sub = self.rider_subtracks[k]
                if sub.observation_log:
                    hb = sub.observation_log[-1]["head_bbox"]
                    centroids[k] = (0.5 * (hb[0] + hb[2]), 0.5 * (hb[1] + hb[3]))
            vel_xy = self._estimate_travel_velocity()
            ordered_slots, mode_label = score_longitudinal_ranks(
                centroids,
                camera_id=self.camera_id,
                velocity_xy=vel_xy,
                min_speed_px=self.bt_config.role_min_moto_speed_px,
                min_sep_px=self.bt_config.role_min_proj_sep_px,
            )
            if ordered_slots is not None:
                self.multi_rider_consensus_frames += 1
                # Direct travel-vector observations carry 2x vote weight over startup perspective fallback
                vote_weight = 2 if mode_label == "motion_vector" else 1
                for rank_idx, s_key in enumerate(ordered_slots):
                    vmap = self.role_votes.setdefault(s_key, {})
                    vmap[rank_idx] = vmap.get(rank_idx, 0) + vote_weight

        # Resolve longitudinal consensus roles once minimum evidence frames are gathered;
        # require >= min_confirm_hits when unlocked so 1-hit transient ghosts never demote confirmed riders.
        min_vote_hits = max(1, int(self.bt_config.min_confirm_hits)) if self.bt_config.unlock_slot_conflicts else 1
        voted_keys = [
            k
            for k in self.role_votes
            if k in self.rider_subtracks
            and not self.rider_subtracks[k].is_coast_expired
            and len(self.rider_subtracks[k].observation_log) >= min_vote_hits
        ]
        if (
            self.multi_rider_consensus_frames >= self.bt_config.min_role_consensus_frames
            and len(voted_keys) >= 2
        ):
            # Net rank score: average rank (lower = more likely driver); check for exact tie
            avg_ranks: list[tuple[float, int, int]] = []
            for k in sorted(voted_keys):
                vmap = self.role_votes[k]
                total_v = sum(vmap.values())
                mean_r = sum(r * c for r, c in vmap.items()) / max(1, total_v)
                avg_ranks.append((mean_r, -total_v, k))
            avg_ranks_sorted = sorted(avg_ranks)

            # Check if adjacent ranks are strictly tied on mean rank
            has_tie = any(
                abs(avg_ranks_sorted[i][0] - avg_ranks_sorted[i + 1][0]) < 1e-6
                for i in range(len(avg_ranks_sorted) - 1)
            )
            if has_tie:
                for _, _, k in avg_ranks_sorted:
                    sub = self.rider_subtracks[k]
                    if sub.slot_role in ("driver", "pillion_1", "pillion_2") or sub.slot_role.startswith("ambiguous_slot_"):
                        sub._sync_consensus_role("uncertain", ordering_confident=False, provisional=True)
            else:
                for assigned_rank, (_, _, k) in enumerate(avg_ranks_sorted):
                    sub = self.rider_subtracks[k]
                    target_role = role_name_for_rank(assigned_rank)
                    sub._sync_consensus_role(target_role, ordering_confident=True, provisional=False)
        elif (
            (self.multi_rider_consensus_frames == 0 or self.bt_config.unlock_slot_conflicts)
            and len(active_slots) == 1
        ):
            # Single surviving sustained rider with zero multi-rider co-occurrence
            sole_k = active_slots[0]
            sole_sub = self.rider_subtracks[sole_k]
            other_sustained = any(
                len(self.rider_subtracks[k].observation_log) >= self.bt_config.min_role_consensus_frames
                for k in self.rider_subtracks
                if k != sole_k
            )
            if not other_sustained and (
                not sole_sub.ordering_confident
                or sole_sub.slot_role_provisional
                or sole_sub.slot_role != "driver"
                or any(e.get("slot_role") != "driver" for e in sole_sub.observation_log)
            ):
                min_sole_frames = (
                    self.bt_config.min_role_consensus_frames
                    if len(self.rider_subtracks) == 1
                    else max(5, self.bt_config.min_role_consensus_frames)
                )
                if len(sole_sub.observation_log) >= min_sole_frames:
                    sole_sub._sync_consensus_role("driver", ordering_confident=True, provisional=False)

        # Guarantee intra-tracklet role consistency between observation_log and subtrack.slot_role
        for sub in self.rider_subtracks.values():
            if any(e.get("slot_role") != sub.slot_role for e in sub.observation_log):
                sub._sync_consensus_role(
                    sub.slot_role,
                    ordering_confident=sub.ordering_confident,
                    provisional=sub.slot_role_provisional,
                )

    def _spawn_subtrack(
        self,
        r_obs: CoupledRiderObservation,
        target_slot: int,
        frame_index: int,
        pts_timestamp_sec: float,
        init_role: str | None = None,
        provisional: bool | None = None,
    ) -> RiderSubTrack:
        role_val = init_role if init_role is not None else r_obs.slot_role
        sub = RiderSubTrack(
            camera_id=self.camera_id,
            parent_moto_id=self.moto_track_id,
            numeric_track_id=self._alloc_rider_id(),
            slot_index=target_slot,
            slot_role=role_val,
            ordering_confident=r_obs.ordering_confident if init_role is None else False,
            initial_obs=r_obs,
            frame_index=frame_index,
            pts_timestamp_sec=pts_timestamp_sec,
            p10_config=self.p10_config,
            bt_config=self.bt_config,
            parent_moto_bbox=self.current_bbox,
            frame_dimensions=self._frame_dimensions(),
        )
        if provisional is not None:
            sub.slot_role_provisional = provisional
        self.rider_subtracks[target_slot] = sub
        self.new_alerts_this_frame.extend(sub.alerts)
        return sub

    def _sync_riders_on_match(
        self,
        coupled_riders: list[CoupledRiderObservation],
        frame_index: int,
        pts_timestamp_sec: float,
        is_init: bool = False,
    ) -> None:
        self.new_alerts_this_frame = []
        fdims = self._frame_dimensions()
        parent_vel = self.kalman.velocity_xy

        if self.bt_config.unlock_slot_conflicts and len(coupled_riders) >= 2:
            # Suppress co-located dual-class duplicate head detections on the same frame
            by_conf = sorted(
                enumerate(coupled_riders),
                key=lambda item: (-item[1].head_detection.confidence, -item[1].coupling_score, item[0]),
            )
            kept_indices: list[int] = []
            for orig_idx, cand_obs in by_conf:
                if any(
                    bbox_iou_xyxy(
                        cand_obs.head_detection.bbox,
                        coupled_riders[k_idx].head_detection.bbox,
                    )
                    >= self.bt_config.duplicate_head_iou_threshold
                    for k_idx in kept_indices
                ):
                    continue
                kept_indices.append(orig_idx)
            if len(kept_indices) < len(coupled_riders):
                kept_sorted = sorted(kept_indices)
                deduped: list[CoupledRiderObservation] = []
                for new_rank, orig_idx in enumerate(kept_sorted):
                    obs_item = coupled_riders[orig_idx]
                    if len(kept_sorted) == 1:
                        base_first = coupled_riders[0]
                        obs_item = replace(
                            obs_item,
                            slot_index=0,
                            slot_role="driver",
                            ordering_confident=(
                                base_first.ordering_confident if base_first.slot_index == 0 else False
                            ),
                            slot_role_provisional=bool(
                                getattr(base_first, "slot_role_provisional", False)
                                if base_first.slot_index == 0
                                else True
                            ),
                        )
                    elif obs_item.slot_index != new_rank:
                        obs_item = replace(
                            obs_item,
                            slot_index=new_rank,
                            slot_role=role_name_for_rank(new_rank),
                        )
                    deduped.append(obs_item)
                coupled_riders = deduped

        if is_init or not self.rider_subtracks:
            matched_slots_init: set[int] = set()
            for r_obs in coupled_riders:
                s_idx = r_obs.slot_index
                if s_idx in self.rider_subtracks:
                    if (
                        len(self.rider_subtracks) + len(self.discarded_tentative_subtracks)
                        >= self.bt_config.max_total_subtracks_per_moto
                    ):
                        continue
                    s_idx = self._next_free_slot_key()
                self._spawn_subtrack(r_obs, s_idx, frame_index, pts_timestamp_sec)
                matched_slots_init.add(s_idx)
            self._reconcile_roles(matched_slots_init, frame_index)
            return

        if not coupled_riders:
            for sub in self.rider_subtracks.values():
                sub.update_unmatched(
                    frame_index=frame_index,
                    parent_moto_bbox=self.current_bbox,
                    frame_dimensions=fdims,
                )
            self._prune_expired_tentative_subtracks(force_all=False)
            return

        active_slots = [
            k
            for k in sorted(self.rider_subtracks.keys())
            if not (self.bt_config.finite_rider_coast_enabled and self.rider_subtracks[k].is_coast_expired)
        ]

        # If all historical subtracks expired coast horizon, allow clean fresh initialization
        if not active_slots and self.bt_config.unlock_slot_conflicts:
            self._prune_expired_tentative_subtracks(force_all=True)
            new_matched: set[int] = set()
            for r_obs in coupled_riders:
                if (
                    len(self.rider_subtracks) + len(self.discarded_tentative_subtracks)
                    >= self.bt_config.max_total_subtracks_per_moto
                ):
                    break
                s_idx = r_obs.slot_index if r_obs.slot_index not in self.rider_subtracks else self._next_free_slot_key()
                self._spawn_subtrack(r_obs, s_idx, frame_index, pts_timestamp_sec)
                new_matched.add(s_idx)
            self._reconcile_roles(new_matched, frame_index)
            return

        # Re-anchor active subtracks onto the parent motorcycle's freshly updated frame bbox
        use_rel = bool(self.bt_config.use_motorcycle_relative_kinematics)
        rel_cfg = self._rel_config()
        if use_rel:
            for s_key in active_slots:
                self.rider_subtracks[s_key].anchor_to_parent(self.current_bbox, frame_dimensions=fdims)

        existing_slots = active_slots if active_slots else sorted(self.rider_subtracks.keys())
        m_sub = len(existing_slots)
        k_obs = len(coupled_riders)

        obs_rels: list[tuple[tuple[float, float, float, float], bool]] = [
            bbox_to_moto_relative(r.head_detection.bbox, self.current_bbox, rel_cfg)
            if use_rel
            else ((0.0, 0.0, 0.0, 0.0), False)
            for r in coupled_riders
        ]

        # 1. Compare every incoming rider observation against predicted bounding boxes
        #    and normalized motorcycle-relative states of all active RiderSubTracks
        iou_mat = np.zeros((m_sub, k_obs), dtype=np.float64)
        dist_mat = np.zeros((m_sub, k_obs), dtype=np.float64)
        rel_dist_mat = np.full((m_sub, k_obs), 1e6, dtype=np.float64)
        aff_mat = np.zeros((m_sub, k_obs), dtype=np.float64)

        for i, s_key in enumerate(existing_slots):
            sub_i = self.rider_subtracks[s_key]
            pred_box = sub_i.kalman.current_bbox_xyxy
            px1, py1, px2, py2 = pred_box
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            pcx, pcy = px1 + 0.5 * pw, py1 + 0.5 * ph
            p_diag = max(1.0, float(np.hypot(pw, ph)))
            sub_rel = sub_i.rel_kinematics.rel_tuple if (use_rel and sub_i.rel_kinematics is not None) else None

            for j, r_obs in enumerate(coupled_riders):
                obox = r_obs.head_detection.bbox
                iou_val = bbox_iou_xyxy(pred_box, obox)
                ocx, ocy = r_obs.head_detection.center
                norm_dist = float(np.hypot(pcx - ocx, pcy - ocy)) / p_diag
                iou_mat[i, j] = iou_val
                dist_mat[i, j] = norm_dist

                if use_rel and sub_rel is not None and obs_rels[j][1]:
                    rel_d = relative_distance(sub_rel, obs_rels[j][0])
                    scale_c = relative_scale_consistency(sub_rel, obs_rels[j][0])
                    rel_dist_mat[i, j] = rel_d
                else:
                    rel_d = float("inf")
                    scale_c = 0.0

                if iou_val >= self.bt_config.rider_match_min_iou:
                    if use_rel and rel_d < float("inf"):
                        rel_term = max(0.0, 1.0 - rel_d / max(1e-4, self.bt_config.rider_rel_match_max_dist))
                        aff_mat[i, j] = max(iou_val, 0.65 * iou_val + 0.35 * rel_term)
                    else:
                        aff_mat[i, j] = iou_val
                elif (
                    use_rel
                    and sub_rel is not None
                    and obs_rels[j][1]
                    and rel_d <= self.bt_config.rider_rel_match_max_dist
                    and scale_c >= 0.25
                ):
                    rel_term = max(0.05, 1.0 - rel_d / max(1e-4, self.bt_config.rider_rel_match_max_dist))
                    aff_mat[i, j] = max(iou_val, 0.55 * rel_term)
                elif iou_val >= 0.05 and norm_dist <= self.bt_config.rider_max_norm_center_dist:
                    aff_mat[i, j] = max(
                        iou_val,
                        0.15 * (1.0 - norm_dist / max(1e-4, self.bt_config.rider_max_norm_center_dist)),
                    )
                else:
                    aff_mat[i, j] = 0.0

        # 2. Detect cross-subtrack spatial ambiguity (contamination prevention)
        abstained_obs: set[int] = set()
        abstained_slots: set[int] = set()
        if m_sub >= 2:
            for j in range(k_obs):
                cand_i = [
                    i
                    for i in range(m_sub)
                    if iou_mat[i, j] > 0.02
                    or dist_mat[i, j] <= self.bt_config.rider_max_norm_center_dist
                    or (use_rel and rel_dist_mat[i, j] <= self.bt_config.rider_rel_match_max_dist)
                ]
                if len(cand_i) >= 2:
                    cand_sorted = sorted(
                        cand_i,
                        key=lambda idx: (iou_mat[idx, j], -rel_dist_mat[idx, j], -dist_mat[idx, j]),
                        reverse=True,
                    )
                    best_i, second_i = cand_sorted[0], cand_sorted[1]
                    iou_margin = float(iou_mat[best_i, j] - iou_mat[second_i, j])
                    rel_margin = (
                        abs(float(rel_dist_mat[second_i, j] - rel_dist_mat[best_i, j]))
                        if use_rel
                        else 0.0
                    )
                    is_ambig = (
                        iou_margin < self.bt_config.rider_ambiguity_iou_margin
                        and (not use_rel or rel_margin < self.bt_config.rider_rel_ambiguity_margin)
                    )
                    if is_ambig:
                        abstained_obs.add(j)
                        abstained_slots.add(existing_slots[best_i])
                        abstained_slots.add(existing_slots[second_i])
                        self.rider_abstained_updates += 1
                        self.rider_subtracks[existing_slots[best_i]].abstained_updates_count += 1
                        self.rider_subtracks[existing_slots[second_i]].abstained_updates_count += 1

        if k_obs >= 2:
            for i in range(m_sub):
                if existing_slots[i] in abstained_slots:
                    continue
                cand_j = [
                    j
                    for j in range(k_obs)
                    if j not in abstained_obs
                    and (
                        iou_mat[i, j] > 0.02
                        or dist_mat[i, j] <= self.bt_config.rider_max_norm_center_dist
                        or (use_rel and rel_dist_mat[i, j] <= self.bt_config.rider_rel_match_max_dist)
                    )
                ]
                if len(cand_j) >= 2:
                    cand_j_sorted = sorted(
                        cand_j,
                        key=lambda jdx: (iou_mat[i, jdx], -rel_dist_mat[i, jdx], -dist_mat[i, jdx]),
                        reverse=True,
                    )
                    b_j, s_j = cand_j_sorted[0], cand_j_sorted[1]
                    s_j_other_max = max(
                        [float(iou_mat[i2, s_j]) for i2 in range(m_sub) if i2 != i], default=0.0
                    )
                    s_j_other_rel = min(
                        [float(rel_dist_mat[i2, s_j]) for i2 in range(m_sub) if i2 != i], default=1e6
                    )
                    rel_j_margin = (
                        abs(float(rel_dist_mat[i, s_j] - rel_dist_mat[i, b_j]))
                        if use_rel
                        else 0.0
                    )
                    other_has_claim = (
                        s_j_other_max >= self.bt_config.rider_match_min_iou
                        or (use_rel and s_j_other_rel <= self.bt_config.rider_rel_match_max_dist)
                    )
                    if (
                        float(iou_mat[i, b_j] - iou_mat[i, s_j]) < self.bt_config.rider_ambiguity_iou_margin
                        and (not use_rel or rel_j_margin < self.bt_config.rider_rel_ambiguity_margin)
                        and not other_has_claim
                    ):
                        abstained_obs.add(b_j)
                        abstained_obs.add(s_j)
                        abstained_slots.add(existing_slots[i])
                        self.rider_abstained_updates += 1
                        self.rider_subtracks[existing_slots[i]].abstained_updates_count += 1

        # 3. Solve bipartite spatial assignment over non-abstained candidates
        cost_mat = np.full((m_sub, k_obs), 1e6, dtype=np.float64)
        for i in range(m_sub):
            if existing_slots[i] in abstained_slots:
                continue
            for j in range(k_obs):
                if j in abstained_obs:
                    continue
                if aff_mat[i, j] > 0.0:
                    cost_mat[i, j] = 1.0 - aff_mat[i, j]

        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matched_slots: set[int] = set()
        used_obs: set[int] = set(abstained_obs)
        matched_obs_boxes: list[tuple[float, float, float, float]] = []
        existing_set = set(existing_slots)

        # 4. Evaluate spatial consistency vs. slot identity and resolve or abstain conservatively
        for i, j in zip(row_ind, col_ind):
            if cost_mat[i, j] >= 1.0:
                continue
            slot_k = existing_slots[i]
            r_obs = coupled_riders[j]
            obs_slot = r_obs.slot_index
            iou_val = float(iou_mat[i, j])
            rel_val = float(rel_dist_mat[i, j])
            other_sub_max_iou = max(
                [float(iou_mat[i2, j]) for i2 in range(m_sub) if i2 != i], default=0.0
            )
            other_sub_min_rel = min(
                [float(rel_dist_mat[i2, j]) for i2 in range(m_sub) if i2 != i], default=1e6
            )

            single_to_multi_transition = (
                self.bt_config.unlock_slot_conflicts
                and m_sub == 1
                and obs_slot not in existing_set
            )

            if obs_slot == slot_k or single_to_multi_transition:
                if (
                    not r_obs.ordering_confident
                    and m_sub >= 2
                    and (iou_val - other_sub_max_iou) < self.bt_config.rider_ambiguity_iou_margin
                    and (not use_rel or (other_sub_min_rel - rel_val) < self.bt_config.rider_rel_ambiguity_margin)
                ):
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    self.rider_subtracks[slot_k].abstained_updates_count += 1
                    abstained_slots.add(slot_k)
                    used_obs.add(j)
                    continue
                preserve_existing = bool(
                    self.bt_config.role_consensus_enabled
                    and (
                        self.multi_rider_consensus_frames >= 1
                        or getattr(r_obs, "slot_role_provisional", False)
                        or not r_obs.ordering_confident
                    )
                    and not self.rider_subtracks[slot_k].slot_role.startswith("ambiguous_slot_")
                )
                alerts = self.rider_subtracks[slot_k].update_matched(
                    r_obs,
                    frame_index,
                    pts_timestamp_sec,
                    preserve_slot_role=preserve_existing,
                    parent_moto_bbox=self.current_bbox,
                    frame_dimensions=fdims,
                    parent_velocity_xy=parent_vel,
                )
                self.new_alerts_this_frame.extend(alerts)
                matched_slots.add(slot_k)
                used_obs.add(j)
                matched_obs_boxes.append(r_obs.head_detection.bbox)
            else:
                # Conflict between coupler slot_index and spatial trajectory assignment
                self.slot_spatial_conflicts_detected += 1
                other_iou_cap = 0.10 if self.bt_config.unlock_slot_conflicts else 0.05
                rel_match_ok = (
                    use_rel
                    and rel_val <= self.bt_config.rider_rel_match_max_dist
                    and other_sub_max_iou < other_iou_cap
                    and (
                        (self.bt_config.unlock_slot_conflicts and m_sub == 1)
                        or (other_sub_min_rel - rel_val) >= self.bt_config.rider_rel_ambiguity_margin
                    )
                )
                is_partial_visibility_match = (
                    (k_obs < m_sub or (self.bt_config.unlock_slot_conflicts and m_sub == 1))
                    and (iou_val >= self.bt_config.rider_match_min_iou or rel_match_ok)
                    and other_sub_max_iou < other_iou_cap
                    and (
                        (self.bt_config.unlock_slot_conflicts and m_sub == 1)
                        or (iou_val - other_sub_max_iou) >= self.bt_config.rider_ambiguity_iou_margin
                        or rel_match_ok
                    )
                )
                is_high_conf_spatial_override = (
                    iou_val >= self.bt_config.rider_spatial_override_min_iou
                    and other_sub_max_iou < other_iou_cap
                    and (iou_val - other_sub_max_iou) >= self.bt_config.rider_spatial_override_min_iou
                )
                if is_partial_visibility_match or is_high_conf_spatial_override:
                    self.spatial_identity_overrides += 1
                    alerts = self.rider_subtracks[slot_k].update_matched(
                        r_obs,
                        frame_index,
                        pts_timestamp_sec,
                        preserve_slot_role=True,
                        parent_moto_bbox=self.current_bbox,
                        frame_dimensions=fdims,
                        parent_velocity_xy=parent_vel,
                    )
                    self.new_alerts_this_frame.extend(alerts)
                    matched_slots.add(slot_k)
                    used_obs.add(j)
                    matched_obs_boxes.append(r_obs.head_detection.bbox)
                else:
                    self.rider_abstained_updates += 1
                    self.rider_subtracks[slot_k].abstained_updates_count += 1
                    abstained_slots.add(slot_k)
                    if obs_slot in self.rider_subtracks and obs_slot in existing_set:
                        self.rider_subtracks[obs_slot].abstained_updates_count += 1
                        abstained_slots.add(obs_slot)
                    used_obs.add(j)

        # 5. Resolve remaining unmatched observations: guard spatial conflicts vs unlock sibling spawn
        for o_idx, r_obs in enumerate(coupled_riders):
            if o_idx in used_obs:
                continue
            obs_slot = r_obs.slot_index

            if not self.bt_config.unlock_slot_conflicts:
                if obs_slot in self.rider_subtracks and obs_slot not in matched_slots:
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    self.rider_subtracks[obs_slot].abstained_updates_count += 1
                    abstained_slots.add(obs_slot)
                    used_obs.add(o_idx)
                    continue
                if len(matched_slots) < len(self.rider_subtracks) or len(abstained_slots) > 0:
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    used_obs.add(o_idx)
                    continue
            else:
                # Check spatial conflict against unmatched ACTIVE subtracks only
                unmatched_active = [k for k in existing_slots if k not in matched_slots]
                conflicts_unmatched = False
                for uk in unmatched_active:
                    u_sub = self.rider_subtracks[uk]
                    u_iou = bbox_iou_xyxy(u_sub.kalman.current_bbox_xyxy, r_obs.head_detection.bbox)
                    if u_iou >= 0.05:
                        conflicts_unmatched = True
                        break
                    if use_rel and u_sub.rel_kinematics is not None and obs_rels[o_idx][1]:
                        u_reld = relative_distance(u_sub.rel_kinematics.rel_tuple, obs_rels[o_idx][0])
                        if u_reld <= self.bt_config.conflict_distinct_min_rel_dist:
                            conflicts_unmatched = True
                            break
                if conflicts_unmatched:
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    if obs_slot in existing_set and obs_slot not in matched_slots:
                        self.rider_subtracks[obs_slot].abstained_updates_count += 1
                        abstained_slots.add(obs_slot)
                    used_obs.add(o_idx)
                    continue

                # Guard against spawning duplicate sibling overlapping an already-matched head
                overlaps_matched = any(
                    bbox_iou_xyxy(mbox, r_obs.head_detection.bbox) >= self.bt_config.sibling_overlap_max_iou
                    for mbox in matched_obs_boxes
                )
                if overlaps_matched:
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    used_obs.add(o_idx)
                    continue

            active_now = [
                k
                for k in self.rider_subtracks
                if not (self.bt_config.finite_rider_coast_enabled and self.rider_subtracks[k].is_coast_expired)
            ]
            if (
                len(active_now) >= self.bt_config.max_active_riders_per_moto
                or (len(self.rider_subtracks) + len(self.discarded_tentative_subtracks))
                >= self.bt_config.max_total_subtracks_per_moto
            ):
                continue

            target_slot = obs_slot if obs_slot not in self.rider_subtracks else self._next_free_slot_key()
            has_active_driver = any(
                self.rider_subtracks[k].slot_role == "driver" for k in active_now
            )
            init_role = None
            provisional_flag = None
            if has_active_driver and r_obs.slot_role == "driver":
                init_role = role_name_for_rank(len(active_now))
                provisional_flag = True

            self._spawn_subtrack(
                r_obs,
                target_slot,
                frame_index,
                pts_timestamp_sec,
                init_role=init_role,
                provisional=provisional_flag,
            )
            matched_slots.add(target_slot)
            matched_obs_boxes.append(r_obs.head_detection.bbox)

        # 6. Advance unmatched (or abstained) existing rider subtracks (bounded finite coasting)
        for s_idx, sub in self.rider_subtracks.items():
            if s_idx not in matched_slots:
                sub.update_unmatched(
                    frame_index=frame_index,
                    parent_moto_bbox=self.current_bbox,
                    frame_dimensions=fdims,
                )
        self._prune_expired_tentative_subtracks(force_all=False)

        # 7. Update longitudinal multi-frame role consensus & observation consistency
        self._reconcile_roles(matched_slots, frame_index)

    def update_matched(
        self,
        coupled_obs: CoupledMotorcycleObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        matched_in_stage2: bool = False,
    ) -> list[ViolationAlert]:
        det = coupled_obs.motorcycle_detection
        self.current_bbox = self.kalman.update(det.bbox)
        self.current_confidence = det.confidence
        self.age += 1
        self.hits += 1
        if matched_in_stage2:
            self.stage2_low_conf_matches += 1

        was_lost = (self.lifecycle_state == TrackLifecycleState.LOST)
        self.time_since_update = 0
        self.last_frame_index = frame_index
        self.last_pts_sec = pts_timestamp_sec
        self.trajectory.append((frame_index, pts_timestamp_sec, self.current_bbox, det.confidence))
        self.observation_log.append(
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "raw_bbox": tuple(float(c) for c in det.bbox),
                "kalman_bbox": tuple(float(c) for c in self.current_bbox),
                "confidence": float(det.confidence),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
                "matched_in_stage2": bool(matched_in_stage2),
                "observed_rider_count": len(coupled_obs.riders),
            }
        )

        if was_lost:
            self.lifecycle_state = TrackLifecycleState.RECOVERED
            self.recovery_count += 1
            self._ever_confirmed = True
        elif self.hits >= self.bt_config.min_confirm_hits or self._ever_confirmed:
            self._ever_confirmed = True
            if self.lifecycle_state != TrackLifecycleState.RECOVERED:
                self.lifecycle_state = TrackLifecycleState.CONFIRMED
        else:
            self.lifecycle_state = TrackLifecycleState.TENTATIVE

        self._sync_riders_on_match(coupled_obs.riders, frame_index, pts_timestamp_sec, is_init=False)
        return self.new_alerts_this_frame

    def update_unmatched(self, frame_index: int) -> None:
        self.age += 1
        self.time_since_update += 1
        self.last_frame_index = frame_index
        self.missed_frames.append(int(frame_index))
        self.new_alerts_this_frame = []
        fdims = self._frame_dimensions()

        for sub in self.rider_subtracks.values():
            sub.update_unmatched(
                frame_index=frame_index,
                parent_moto_bbox=self.current_bbox,
                frame_dimensions=fdims,
            )
        self._prune_expired_tentative_subtracks(force_all=False)

        if not self._ever_confirmed:
            # Tentative track that missed before reaching min_confirm_hits is terminated
            self.lifecycle_state = TrackLifecycleState.TERMINATED
        elif self.time_since_update > self.bt_config.max_lost_frames:
            self.lifecycle_state = TrackLifecycleState.TERMINATED
        else:
            self.lifecycle_state = TrackLifecycleState.LOST

    @property
    def is_eligible_for_stage2(self) -> bool:
        """Only confirmed/recovered/lost tracks participate in ByteTrack Stage 2 low-conf recovery."""
        return self._ever_confirmed and self.lifecycle_state in (
            TrackLifecycleState.CONFIRMED,
            TrackLifecycleState.RECOVERED,
            TrackLifecycleState.LOST,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "moto_track_id": self.moto_track_id,
            "camera_id": self.camera_id,
            "lifecycle_state": self.lifecycle_state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "stage2_low_conf_matches": self.stage2_low_conf_matches,
            "recovery_count": self.recovery_count,
            "slot_spatial_conflicts_detected": self.slot_spatial_conflicts_detected,
            "rider_abstained_updates": self.rider_abstained_updates,
            "spatial_identity_overrides": self.spatial_identity_overrides,
            "multi_rider_consensus_frames": self.multi_rider_consensus_frames,
            "first_pts_sec": round(self.first_pts_sec, 6),
            "last_pts_sec": round(self.last_pts_sec, 6),
            "bbox": [round(c, 2) for c in self.current_bbox],
            "velocity_xy": [round(v, 3) for v in self.kalman.velocity_xy],
            "confidence": round(self.current_confidence, 4),
            "rider_subtracks": {
                s_idx: sub.to_dict() for s_idx, sub in sorted(self.rider_subtracks.items())
            },
        }


class OrphanRiderTrack:
    """
    Independent trajectory for a rider head that could not be reliably coupled to a
    motorcycle parent (`OrphanRiderObservation`). Connects directly to Phase 10's
    temporal verification components so uncoupled riders are never silently dropped.
    """

    def __init__(
        self,
        camera_id: str,
        numeric_id: int,
        initial_orphan: OrphanRiderObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        bt_config: ByteTrackConfig,
        p10_config: Phase10AdapterConfig,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.numeric_id = numeric_id
        self.orphan_track_id = f"{camera_id}_orphan_{numeric_id:04d}"
        self.bt_config = bt_config
        self.p10_config = p10_config
        self.frame_dimensions = frame_dimensions

        head = initial_orphan.head_detection
        self.kalman = KalmanBoxFilter(
            head.bbox,
            std_weight_pos=self.bt_config.kalman_std_weight_pos,
            std_weight_vel=self.bt_config.kalman_std_weight_vel,
        )
        self.last_rejection_reason = initial_orphan.rejection_reason
        self.last_frame_index = frame_index
        self._last_matched_frame_index = int(frame_index)
        self.first_pts_sec = pts_timestamp_sec
        self.last_pts_sec = pts_timestamp_sec
        self.genuine_hits: int = 1
        self.tentative_expired: bool = False

        p10_bbox = BoundingBox(
            x1=head.x1,
            y1=head.y1,
            x2=head.x2,
            y2=head.y2,
            confidence=head.confidence,
            class_id=head.class_id,
            class_name=head.class_name,
        )
        self.phase10_track = Track(
            track_id=numeric_id,
            initial_bbox=p10_bbox,
            frame_index=frame_index,
            aggregator_method=p10_config.aggregator_method,
            window_size=p10_config.sliding_window_size,
            sm_config=p10_config.state_machine_config,
            ema_alpha=p10_config.ema_alpha,
        )
        self.decision_engine = ViolationDecisionEngine(config=p10_config.decision_config)
        self._ever_confirmed: bool = (self.genuine_hits >= self.confirm_hits_threshold)
        self.lifecycle_state = (
            TrackLifecycleState.CONFIRMED
            if self._ever_confirmed
            else TrackLifecycleState.TENTATIVE
        )
        self.pts_history: list[tuple[int, float, tuple[float, float, float, float], str, float]] = [
            (frame_index, pts_timestamp_sec, head.bbox, head.class_name, head.confidence)
        ]
        self.observation_log: list[dict[str, Any]] = [
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "head_bbox": tuple(float(c) for c in head.bbox),
                "head_conf": float(head.confidence),
                "head_class_name": str(head.class_name),
                "person_bbox": (
                    tuple(float(c) for c in initial_orphan.person_detection.bbox)
                    if initial_orphan.person_detection is not None
                    else None
                ),
                "person_conf": (
                    float(initial_orphan.person_detection.confidence)
                    if initial_orphan.person_detection is not None
                    else None
                ),
                "rejection_reason": str(initial_orphan.rejection_reason),
                "best_candidate_score": float(initial_orphan.best_candidate_score),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        ]
        self.helmet_evidence = HelmetEvidenceAccumulator()
        self.helmet_evidence.record(
            head.class_name, head.confidence, frame_index, pts_timestamp_sec,
            self.phase10_track.aggregator.get_aggregated_confidence(),
        )
        self.missed_frames: list[int] = []
        self.alerts: list[ViolationAlert] = []
        self._evaluate_alerts(frame_index, pts_timestamp_sec)

    @property
    def confirm_hits_threshold(self) -> int:
        if self.bt_config.orphan_lifecycle_gating_enabled:
            return max(1, int(self.bt_config.orphan_min_confirm_hits))
        return max(1, int(self.bt_config.min_confirm_hits))

    def _effective_frame_dims(self) -> tuple[int, int] | None:
        if self.frame_dimensions is not None:
            return self.frame_dimensions
        if self.bt_config.frame_width and self.bt_config.frame_height:
            return (int(self.bt_config.frame_width), int(self.bt_config.frame_height))
        return None

    def _evaluate_alerts(self, frame_index: int, pts_timestamp_sec: float) -> list[ViolationAlert]:
        eff_fps = frame_index / pts_timestamp_sec if pts_timestamp_sec > 1e-4 else 60.0
        new_alerts = self.decision_engine.process_tracks(
            frame_index=frame_index,
            tracks=[self.phase10_track],
            fps=max(1.0, eff_fps),
        )
        self.alerts.extend(new_alerts)
        return new_alerts

    def predict(self, current_frame_index: int) -> tuple[float, float, float, float]:
        dt = max(1, current_frame_index - self.last_frame_index)
        pred = self.kalman.predict(dt_frames=float(dt))
        fdims = self._effective_frame_dims()
        if fdims is not None or self.bt_config.clamp_non_negative_coords:
            clamped, _ = clamp_bbox_to_frame(
                pred,
                frame_dimensions=fdims,
                clamp_non_negative=bool(self.bt_config.clamp_non_negative_coords),
            )
            self.kalman.sync_bbox_state(clamped)
            return clamped
        return pred

    def update_matched(
        self,
        orphan_obs: OrphanRiderObservation,
        frame_index: int,
        pts_timestamp_sec: float,
    ) -> list[ViolationAlert]:
        head = orphan_obs.head_detection
        self.kalman.update(head.bbox)
        self.last_rejection_reason = orphan_obs.rejection_reason
        self.last_frame_index = frame_index
        self.last_pts_sec = pts_timestamp_sec

        was_lost = (self.lifecycle_state == TrackLifecycleState.LOST)
        is_new_genuine_frame = int(frame_index) > self._last_matched_frame_index
        if is_new_genuine_frame:
            self.genuine_hits += 1
            self._last_matched_frame_index = int(frame_index)

        p10_bbox = BoundingBox(
            x1=head.x1,
            y1=head.y1,
            x2=head.x2,
            y2=head.y2,
            confidence=head.confidence,
            class_id=head.class_id,
            class_name=head.class_name,
        )
        self.phase10_track.update_matched(p10_bbox, frame_index=frame_index)
        self.helmet_evidence.record(
            head.class_name, head.confidence, frame_index, pts_timestamp_sec,
            self.phase10_track.aggregator.get_aggregated_confidence(),
        )
        self.pts_history.append(
            (frame_index, pts_timestamp_sec, head.bbox, head.class_name, head.confidence)
        )
        self.observation_log.append(
            {
                "frame_index": int(frame_index),
                "pts_timestamp_sec": float(pts_timestamp_sec),
                "head_bbox": tuple(float(c) for c in head.bbox),
                "head_conf": float(head.confidence),
                "head_class_name": str(head.class_name),
                "person_bbox": (
                    tuple(float(c) for c in orphan_obs.person_detection.bbox)
                    if orphan_obs.person_detection is not None
                    else None
                ),
                "person_conf": (
                    float(orphan_obs.person_detection.confidence)
                    if orphan_obs.person_detection is not None
                    else None
                ),
                "rejection_reason": str(orphan_obs.rejection_reason),
                "best_candidate_score": float(orphan_obs.best_candidate_score),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        )

        if self.genuine_hits >= self.confirm_hits_threshold:
            if was_lost and self._ever_confirmed:
                self.lifecycle_state = TrackLifecycleState.RECOVERED
            else:
                self._ever_confirmed = True
                self.lifecycle_state = TrackLifecycleState.CONFIRMED
        else:
            self.lifecycle_state = TrackLifecycleState.TENTATIVE

        return self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def update_unmatched(self, frame_index: int) -> None:
        self.last_frame_index = frame_index
        self.missed_frames.append(int(frame_index))
        self.phase10_track.update_unmatched(frame_index=frame_index)
        if self.bt_config.orphan_lifecycle_gating_enabled and not self._ever_confirmed:
            if self.phase10_track.time_since_update > max(
                1, int(self.bt_config.orphan_tentative_max_lost_frames)
            ):
                self.tentative_expired = True
                self.lifecycle_state = TrackLifecycleState.TERMINATED
            else:
                self.lifecycle_state = TrackLifecycleState.TENTATIVE
            return

        if (
            self.phase10_track.time_since_update > self.bt_config.orphan_max_lost_frames
            or self.phase10_track.state == TemporalState.TRACK_LOST
        ):
            self.lifecycle_state = TrackLifecycleState.TERMINATED
        else:
            self.lifecycle_state = TrackLifecycleState.LOST

    @property
    def track_helmet_state(self) -> str:
        return self.helmet_evidence.export(
            self.phase10_track.aggregator.get_aggregated_confidence()
        )["track_helmet_state"]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.helmet_evidence.export(self.phase10_track.aggregator.get_aggregated_confidence()),
            "orphan_track_id": self.orphan_track_id,
            "camera_id": self.camera_id,
            "lifecycle_state": self.lifecycle_state.value,
            "confirmed": self._ever_confirmed,
            "genuine_hits": self.genuine_hits,
            "tentative_expired": self.tentative_expired,
            "temporal_state": self.phase10_track.state.value,
            "last_rejection_reason": self.last_rejection_reason,
            "hits": self.phase10_track.hits,
            "time_since_update": self.phase10_track.time_since_update,
            "aggregated_confidence": self.phase10_track.aggregator.get_aggregated_confidence(),
            "bbox": [round(c, 2) for c in self.kalman.current_bbox_xyxy],
            "alerts_emitted": len(self.alerts),
        }


class CameraTrackingPipeline:
    """
    Complete single-camera (`front` or `back`) hierarchical detection-coupling-tracking
    pipeline with Phase 12C update-gate protection.
    """

    def __init__(
        self,
        camera_id: str,
        coupler_config: CouplerConfig | None = None,
        bytetrack_config: ByteTrackConfig | None = None,
        phase10_config: Phase10AdapterConfig | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        if camera_id not in ("front", "back"):
            raise ValueError(f"camera_id must be 'front' or 'back', got '{camera_id}'")
        self.camera_id = camera_id
        self.coupler_config = coupler_config or CouplerConfig()
        self.bt_config = bytetrack_config or ByteTrackConfig()
        self.p10_config = phase10_config or Phase10AdapterConfig()
        if frame_dimensions is not None:
            self.frame_dimensions: tuple[int, int] | None = (
                int(frame_dimensions[0]),
                int(frame_dimensions[1]),
            )
        elif self.bt_config.frame_width and self.bt_config.frame_height:
            self.frame_dimensions = (
                int(self.bt_config.frame_width),
                int(self.bt_config.frame_height),
            )
        else:
            self.frame_dimensions = None

        self.coupler = MotoRiderCoupler(config=self.coupler_config)

        self._next_moto_id: int = 1
        self._next_rider_id: int = 1
        self._next_orphan_id: int = 1

        self.active_moto_tracks: dict[str, MotoTrack] = {}
        self.terminated_moto_tracks: list[MotoTrack] = []

        self.active_orphan_tracks: dict[str, OrphanRiderTrack] = {}
        self.terminated_orphan_tracks: list[OrphanRiderTrack] = []
        self.discarded_tentative_orphan_tracks: list[OrphanRiderTrack] = []

        self.low_conf_orphan_filtered: int = 0
        self.duplicate_orphan_suppressed: int = 0
        self.frames_updated: int = 0
        self.duplicate_ticks_skipped: int = 0
        self._last_processed_frame_index: int = -1
        self._last_step_summary: dict[str, Any] | None = None

    @classmethod
    def from_yaml(
        cls,
        camera_id: str,
        yaml_path: str | Path = "configs/tracking_config.yaml",
        frame_dimensions: tuple[int, int] | None = None,
    ) -> CameraTrackingPipeline:
        """Instantiate a CameraTrackingPipeline from `configs/tracking_config.yaml`."""
        path = Path(yaml_path)
        if not path.exists():
            return cls(camera_id=camera_id, frame_dimensions=frame_dimensions)

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        c_raw = raw.get("moto_rider_coupler", {})
        w_with = c_raw.get("weights_with_person", {})
        w_no = c_raw.get("weights_without_person", {})
        coupler_cfg = CouplerConfig(
            max_riders_per_motorcycle=int(c_raw.get("max_riders_per_motorcycle", 3)),
            min_coupling_score=float(c_raw.get("min_coupling_score", 0.35)),
            ambiguity_margin_threshold=float(c_raw.get("ambiguity_margin_threshold", 0.08)),
            head_upper_cone_height_ratio=float(c_raw.get("head_upper_cone_height_ratio", 1.25)),
            head_horizontal_expansion_ratio=float(c_raw.get("head_horizontal_expansion_ratio", 0.25)),
            max_head_center_below_moto_ratio=float(c_raw.get("max_head_center_below_moto_ratio", 0.55)),
            max_head_to_moto_width_ratio=float(c_raw.get("max_head_to_moto_width_ratio", 1.15)),
            person_mount_min_iou=float(c_raw.get("person_mount_min_iou", 0.12)),
            person_mount_min_ioa=float(c_raw.get("person_mount_min_ioa", 0.25)),
            head_in_person_upper_ratio=float(c_raw.get("head_in_person_upper_ratio", 0.45)),
            head_in_person_min_ioa=float(c_raw.get("head_in_person_min_ioa", 0.35)),
            head_person_ambiguity_margin=float(c_raw.get("head_person_ambiguity_margin", 0.08)),
            slot_ordering_min_separation_px=float(c_raw.get("slot_ordering_min_separation_px", 8.0)),
            provisional_single_rider_unconfident=bool(
                c_raw.get("provisional_single_rider_unconfident", True)
            ),
            w_with_person_cone=float(w_with.get("cone_ioa", 0.40)),
            w_with_person_bridge=float(w_with.get("person_bridge", 0.30)),
            w_with_person_lateral=float(w_with.get("lateral_align", 0.20)),
            w_with_person_vertical=float(w_with.get("vertical_scale", 0.10)),
            w_no_person_cone=float(w_no.get("cone_ioa", 0.55)),
            w_no_person_lateral=float(w_no.get("lateral_align", 0.30)),
            w_no_person_vertical=float(w_no.get("vertical_scale", 0.15)),
        )

        bt_raw = raw.get("kalman_bytetrack", {})
        bt_cfg = ByteTrackConfig(
            high_conf_threshold=float(bt_raw.get("high_conf_threshold", 0.45)),
            low_conf_threshold=float(bt_raw.get("low_conf_threshold", 0.18)),
            new_track_min_conf=float(bt_raw.get("new_track_min_conf", 0.45)),
            stage1_iou_threshold=float(bt_raw.get("stage1_iou_threshold", 0.25)),
            stage2_iou_threshold=float(bt_raw.get("stage2_iou_threshold", 0.20)),
            min_confirm_hits=int(bt_raw.get("min_confirm_hits", 2)),
            max_lost_frames=int(bt_raw.get("max_lost_frames", 10)),
            orphan_iou_threshold=float(bt_raw.get("orphan_iou_threshold", 0.25)),
            orphan_max_lost_frames=int(bt_raw.get("orphan_max_lost_frames", 8)),
            kalman_std_weight_pos=float(bt_raw.get("kalman_std_weight_pos", 0.05)),
            kalman_std_weight_vel=float(bt_raw.get("kalman_std_weight_vel", 0.00625)),
            rider_match_min_iou=float(bt_raw.get("rider_match_min_iou", 0.15)),
            rider_spatial_override_min_iou=float(bt_raw.get("rider_spatial_override_min_iou", 0.35)),
            rider_ambiguity_iou_margin=float(bt_raw.get("rider_ambiguity_iou_margin", 0.15)),
            rider_max_norm_center_dist=float(bt_raw.get("rider_max_norm_center_dist", 0.65)),
            use_motorcycle_relative_kinematics=bool(bt_raw.get("use_motorcycle_relative_kinematics", True)),
            rel_damping_factor=float(bt_raw.get("rel_damping_factor", 0.20)),
            rel_pos_alpha=float(bt_raw.get("rel_pos_alpha", 0.70)),
            rel_size_alpha=float(bt_raw.get("rel_size_alpha", 0.50)),
            rel_vel_beta=float(bt_raw.get("rel_vel_beta", 0.25)),
            rel_max_vel=float(bt_raw.get("rel_max_vel", 0.35)),
            rider_rel_match_max_dist=float(bt_raw.get("rider_rel_match_max_dist", 0.58)),
            rider_rel_ambiguity_margin=float(bt_raw.get("rider_rel_ambiguity_margin", 0.08)),
            finite_rider_coast_enabled=bool(bt_raw.get("finite_rider_coast_enabled", True)),
            max_rider_coast_frames=int(bt_raw.get("max_rider_coast_frames", 15)),
            tentative_rider_coast_frames=int(bt_raw.get("tentative_rider_coast_frames", 3)),
            unlock_slot_conflicts=bool(bt_raw.get("unlock_slot_conflicts", True)),
            duplicate_head_iou_threshold=float(bt_raw.get("duplicate_head_iou_threshold", 0.55)),
            sibling_overlap_max_iou=float(bt_raw.get("sibling_overlap_max_iou", 0.35)),
            conflict_distinct_min_rel_dist=float(bt_raw.get("conflict_distinct_min_rel_dist", 0.58)),
            max_active_riders_per_moto=int(bt_raw.get("max_active_riders_per_moto", 3)),
            max_total_subtracks_per_moto=int(bt_raw.get("max_total_subtracks_per_moto", 6)),
            role_consensus_enabled=bool(bt_raw.get("role_consensus_enabled", True)),
            min_role_consensus_frames=int(bt_raw.get("min_role_consensus_frames", 3)),
            role_min_moto_speed_px=float(bt_raw.get("role_min_moto_speed_px", 2.0)),
            role_min_proj_sep_px=float(bt_raw.get("role_min_proj_sep_px", 8.0)),
            orphan_lifecycle_gating_enabled=bool(bt_raw.get("orphan_lifecycle_gating_enabled", True)),
            orphan_min_confirm_hits=int(bt_raw.get("orphan_min_confirm_hits", 2)),
            orphan_tentative_max_lost_frames=int(bt_raw.get("orphan_tentative_max_lost_frames", 3)),
            orphan_tentative_min_conf=float(bt_raw.get("orphan_tentative_min_conf", 0.18)),
            export_confirmed_orphans_only=bool(bt_raw.get("export_confirmed_orphans_only", True)),
            frame_width=(
                int(bt_raw["frame_width"])
                if bt_raw.get("frame_width") is not None
                else None
            ),
            frame_height=(
                int(bt_raw["frame_height"])
                if bt_raw.get("frame_height") is not None
                else None
            ),
            clamp_non_negative_coords=bool(bt_raw.get("clamp_non_negative_coords", True)),
        )

        p10_raw = raw.get("phase10_temporal_verification", {})
        sm_raw = p10_raw.get("state_machine", {})
        dec_raw = p10_raw.get("violation_decision", {})
        p10_cfg = Phase10AdapterConfig(
            aggregator_method=str(p10_raw.get("aggregator_method", "ema")),
            sliding_window_size=int(p10_raw.get("sliding_window_size", 10)),
            ema_alpha=float(p10_raw.get("ema_alpha", 0.35)),
            state_machine_config=StateMachineConfig(
                min_observation_frames=int(sm_raw.get("min_observation_frames", 3)),
                confirm_frames=int(sm_raw.get("confirm_frames", 5)),
                max_occlusion_frames=int(sm_raw.get("max_occlusion_frames", 5)),
                track_lost_frames=int(sm_raw.get("track_lost_frames", 15)),
                suspect_threshold=float(sm_raw.get("suspect_threshold", 0.50)),
                confirm_threshold=float(sm_raw.get("confirm_threshold", 0.70)),
                clear_threshold=float(sm_raw.get("clear_threshold", 0.40)),
                helmet_threshold=float(sm_raw.get("helmet_threshold", 0.65)),
            ),
            decision_config=DecisionConfig(
                violation_alert_threshold=float(dec_raw.get("violation_alert_threshold", 0.65)),
                min_confirmation_frames=int(dec_raw.get("min_confirmation_frames", 5)),
                suppress_duplicates_per_track=bool(dec_raw.get("suppress_duplicates_per_track", True)),
                cooldown_frames=int(dec_raw.get("cooldown_frames", 90)),
            ),
        )

        return cls(
            camera_id=camera_id,
            coupler_config=coupler_cfg,
            bytetrack_config=bt_cfg,
            phase10_config=p10_cfg,
            frame_dimensions=frame_dimensions,
        )

    def exported_orphan_tracks(self, include_tentative: bool | None = None) -> list[OrphanRiderTrack]:
        """Return deduplicated orphan tracks for downstream export, enforcing confirmation gating when enabled."""
        confirmed_only = (
            (self.bt_config.orphan_lifecycle_gating_enabled and self.bt_config.export_confirmed_orphans_only)
            if include_tentative is None
            else (not include_tentative)
        )
        raw: list[OrphanRiderTrack] = (
            list(self.terminated_orphan_tracks)
            + list(self.active_orphan_tracks.values())
        )
        if not confirmed_only:
            raw.extend(self.discarded_tentative_orphan_tracks)
        seen: set[str] = set()
        exported: list[OrphanRiderTrack] = []
        for o in raw:
            if o.orphan_track_id in seen:
                continue
            if confirmed_only and not getattr(o, "_ever_confirmed", False):
                continue
            seen.add(o.orphan_track_id)
            exported.append(o)
        return exported

    def _alloc_rider_id(self) -> int:
        rid = self._next_rider_id
        self._next_rider_id += 1
        return rid

    def step(
        self,
        detection_bundle: FrameDetectionBundle,
        should_update_state: bool = True,
    ) -> dict[str, Any]:
        """
        Advance per-camera tracking by one physical frame.
        If `should_update_state` is False (e.g., repeated 30 fps frame on a 60 Hz tick)
        or `frame_index == self._last_processed_frame_index`, skips state advancement
        and returns the cached state with `did_advance_state = False`.
        """
        if not should_update_state or (
            detection_bundle.frame_index == self._last_processed_frame_index
            and self._last_processed_frame_index >= 0
        ):
            self.duplicate_ticks_skipped += 1
            if self._last_step_summary is not None:
                cached = dict(self._last_step_summary)
                cached["did_advance_state"] = False
                cached["duplicate_ticks_skipped"] = self.duplicate_ticks_skipped
                return cached
            return {
                "camera_id": self.camera_id,
                "frame_index": detection_bundle.frame_index,
                "pts_timestamp_sec": detection_bundle.pts_timestamp_sec,
                "did_advance_state": False,
                "duplicate_ticks_skipped": self.duplicate_ticks_skipped,
                "active_moto_tracks": [],
                "active_orphan_tracks": [],
                "new_alerts": [],
            }

        frame_idx = detection_bundle.frame_index
        pts_sec = detection_bundle.pts_timestamp_sec
        self._last_processed_frame_index = frame_idx
        self.frames_updated += 1

        # 1. Spatial Rider-to-Motorcycle Coupling
        coupling: FrameCouplingResult = self.coupler.couple_frame(
            camera_id=self.camera_id,
            frame_index=frame_idx,
            pts_timestamp_sec=pts_sec,
            motorcycles=detection_bundle.motorcycles,
            persons=detection_bundle.persons,
            heads=detection_bundle.all_heads,
        )

        # 2. Motorcycle-Level Kalman Prediction & Two-Stage ByteTrack Association
        moto_ids = list(self.active_moto_tracks.keys())
        pred_moto_boxes = [
            self.active_moto_tracks[mid].predict(frame_idx) for mid in moto_ids
        ]
        stage2_elig = [
            self.active_moto_tracks[mid].is_eligible_for_stage2 for mid in moto_ids
        ]

        coupled_motos = coupling.coupled_motorcycles
        det_moto_boxes = [cm.motorcycle_detection.bbox for cm in coupled_motos]
        det_moto_confs = [cm.motorcycle_detection.confidence for cm in coupled_motos]

        bt_res = two_stage_bytetrack_assign(
            track_boxes=pred_moto_boxes,
            det_boxes=det_moto_boxes,
            det_confidences=det_moto_confs,
            config=self.bt_config,
            track_can_use_stage2=stage2_elig,
        )

        new_alerts: list[ViolationAlert] = []

        for t_idx, d_idx, _ in bt_res["stage1_matches"]:
            mid = moto_ids[t_idx]
            alerts = self.active_moto_tracks[mid].update_matched(
                coupled_motos[d_idx], frame_idx, pts_sec, matched_in_stage2=False
            )
            new_alerts.extend(alerts)

        for t_idx, d_idx, _ in bt_res["stage2_matches"]:
            mid = moto_ids[t_idx]
            alerts = self.active_moto_tracks[mid].update_matched(
                coupled_motos[d_idx], frame_idx, pts_sec, matched_in_stage2=True
            )
            new_alerts.extend(alerts)

        for t_idx in bt_res["unmatched_tracks"]:
            mid = moto_ids[t_idx]
            self.active_moto_tracks[mid].update_unmatched(frame_idx)

        # Initialize new MotoTracks for unmatched high-confidence motorcycle detections
        for d_idx in bt_res["unmatched_high_detections"]:
            cm = coupled_motos[d_idx]
            if cm.motorcycle_detection.confidence >= self.bt_config.new_track_min_conf:
                num_id = self._next_moto_id
                self._next_moto_id += 1
                m_track = MotoTrack(
                    camera_id=self.camera_id,
                    numeric_id=num_id,
                    initial_coupled=cm,
                    frame_index=frame_idx,
                    pts_timestamp_sec=pts_sec,
                    bt_config=self.bt_config,
                    p10_config=self.p10_config,
                    rider_id_alloc_fn=self._alloc_rider_id,
                    frame_dimensions=self.frame_dimensions,
                )
                self.active_moto_tracks[m_track.moto_track_id] = m_track
                new_alerts.extend(m_track.new_alerts_this_frame)

        # Any riders that were coupled to an unmatched low-confidence motorcycle that failed
        # to spawn a MotoTrack are rescued into the orphan rider pool so no head is lost
        rescued_orphans: list[OrphanRiderObservation] = list(coupling.orphan_riders)
        for d_idx in bt_res["unmatched_low_detections"]:
            cm = coupled_motos[d_idx]
            for r_obs in cm.riders:
                rescued_orphans.append(
                    OrphanRiderObservation(
                        head_detection=r_obs.head_detection,
                        person_detection=r_obs.person_detection,
                        rejection_reason="LOW_CONF_UNMATCHED_PARENT_MOTO",
                        best_candidate_moto_idx=cm.moto_index,
                        best_candidate_score=r_obs.coupling_score,
                        second_best_score=r_obs.second_best_score,
                        candidate_features=r_obs.geometric_features,
                    )
                )

        # Purge terminated MotoTracks
        surviving_motos: dict[str, MotoTrack] = {}
        for mid, m_track in self.active_moto_tracks.items():
            if m_track.lifecycle_state == TrackLifecycleState.TERMINATED:
                self.terminated_moto_tracks.append(m_track)
            else:
                surviving_motos[mid] = m_track
        self.active_moto_tracks = surviving_motos

        # 3. OrphanRiderTrack Kalman Prediction & IoU Association
        orphan_ids = list(self.active_orphan_tracks.keys())
        pred_orphan_boxes = [
            self.active_orphan_tracks[oid].predict(frame_idx) for oid in orphan_ids
        ]
        det_orphan_boxes = [o.head_detection.bbox for o in rescued_orphans]

        o_matches, o_unmatched_t, o_unmatched_d = hungarian_iou_match(
            pred_orphan_boxes, det_orphan_boxes, self.bt_config.orphan_iou_threshold
        )

        for t_idx, d_idx, _ in o_matches:
            oid = orphan_ids[t_idx]
            alerts = self.active_orphan_tracks[oid].update_matched(
                rescued_orphans[d_idx], frame_idx, pts_sec
            )
            new_alerts.extend(alerts)

        for t_idx in o_unmatched_t:
            oid = orphan_ids[t_idx]
            self.active_orphan_tracks[oid].update_unmatched(frame_idx)

        orphan_spawn_order = (
            sorted(
                o_unmatched_d,
                key=lambda idx: -float(rescued_orphans[idx].head_detection.confidence),
            )
            if self.bt_config.orphan_lifecycle_gating_enabled
            else o_unmatched_d
        )
        for d_idx in orphan_spawn_order:
            o_obs = rescued_orphans[d_idx]
            if self.bt_config.orphan_lifecycle_gating_enabled:
                if o_obs.head_detection.confidence < self.bt_config.orphan_tentative_min_conf:
                    self.low_conf_orphan_filtered += 1
                    continue
                if any(
                    bbox_iou_xyxy(o_obs.head_detection.bbox, existing.kalman.current_bbox_xyxy) >= 0.50
                    for existing in self.active_orphan_tracks.values()
                ):
                    self.duplicate_orphan_suppressed += 1
                    continue
            num_oid = self._next_orphan_id
            self._next_orphan_id += 1
            o_track = OrphanRiderTrack(
                camera_id=self.camera_id,
                numeric_id=num_oid,
                initial_orphan=o_obs,
                frame_index=frame_idx,
                pts_timestamp_sec=pts_sec,
                bt_config=self.bt_config,
                p10_config=self.p10_config,
                frame_dimensions=self.frame_dimensions,
            )
            self.active_orphan_tracks[o_track.orphan_track_id] = o_track
            new_alerts.extend(o_track.alerts)

        surviving_orphans: dict[str, OrphanRiderTrack] = {}
        confirmed_export_only = (
            self.bt_config.orphan_lifecycle_gating_enabled
            and self.bt_config.export_confirmed_orphans_only
        )
        for oid, o_track in self.active_orphan_tracks.items():
            if o_track.lifecycle_state == TrackLifecycleState.TERMINATED:
                if confirmed_export_only and not getattr(o_track, "_ever_confirmed", False):
                    self.discarded_tentative_orphan_tracks.append(o_track)
                else:
                    self.terminated_orphan_tracks.append(o_track)
            else:
                surviving_orphans[oid] = o_track
        self.active_orphan_tracks = surviving_orphans

        step_summary = {
            "camera_id": self.camera_id,
            "frame_index": frame_idx,
            "pts_timestamp_sec": round(pts_sec, 6),
            "did_advance_state": True,
            "frames_updated": self.frames_updated,
            "duplicate_ticks_skipped": self.duplicate_ticks_skipped,
            "coupling": coupling.to_dict(),
            "stage1_moto_matches": len(bt_res["stage1_matches"]),
            "stage2_moto_matches": len(bt_res["stage2_matches"]),
            "active_moto_tracks": [m.to_dict() for m in self.active_moto_tracks.values()],
            "active_orphan_tracks": [o.to_dict() for o in self.active_orphan_tracks.values()],
            "new_alerts": [a.to_dict() for a in new_alerts],
        }
        self._last_step_summary = step_summary
        return step_summary
