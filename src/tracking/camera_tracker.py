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

from dataclasses import dataclass, field
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


@dataclass
class Phase10AdapterConfig:
    """Configuration bridging RiderSubTrack/OrphanRiderTrack to Phase 10 components."""
    aggregator_method: str = "ema"
    sliding_window_size: int = 10
    ema_alpha: float = 0.35
    state_machine_config: StateMachineConfig = field(default_factory=StateMachineConfig)
    decision_config: DecisionConfig = field(default_factory=DecisionConfig)


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
    ) -> None:
        self.camera_id = camera_id
        self.parent_moto_id = parent_moto_id
        self.numeric_track_id = numeric_track_id
        self.slot_index = int(slot_index)
        self.slot_role = slot_role
        self.ordering_confident = ordering_confident
        self.subtrack_id = f"{parent_moto_id}_rider_s{slot_index}"
        self.p10_config = p10_config
        self.bt_config = bt_config or ByteTrackConfig()
        self.abstained_updates_count: int = 0

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
        )
        self.decision_engine = ViolationDecisionEngine(config=p10_config.decision_config)
        self.kalman = KalmanBoxFilter(
            head.bbox,
            std_weight_pos=self.bt_config.kalman_std_weight_pos,
            std_weight_vel=self.bt_config.kalman_std_weight_vel,
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
                "ordering_confident": bool(self.ordering_confident),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        ]
        self.missed_frames: list[int] = []
        self.last_coupling_score = initial_obs.coupling_score
        self.alerts: list[ViolationAlert] = []
        self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def _evaluate_alerts(self, frame_index: int, pts_timestamp_sec: float) -> list[ViolationAlert]:
        eff_fps = frame_index / pts_timestamp_sec if pts_timestamp_sec > 1e-4 else 60.0
        new_alerts = self.decision_engine.process_tracks(
            frame_index=frame_index,
            tracks=[self.phase10_track],
            fps=max(1.0, eff_fps),
        )
        self.alerts.extend(new_alerts)
        return new_alerts

    def predict(self, dt_frames: float = 1.0) -> tuple[float, float, float, float]:
        return self.kalman.predict(dt_frames=dt_frames)

    def update_matched(
        self,
        obs: CoupledRiderObservation,
        frame_index: int,
        pts_timestamp_sec: float,
        preserve_slot_role: bool = False,
    ) -> list[ViolationAlert]:
        head = obs.head_detection
        self.kalman.update(head.bbox)
        if not preserve_slot_role:
            if obs.ordering_confident or self.slot_role.startswith("ambiguous_slot_"):
                self.slot_role = obs.slot_role
            self.ordering_confident = obs.ordering_confident
        else:
            # Spatial continuity overrode a conflicting or partial-visibility slot index;
            # preserve established semantic slot_role ("driver" / "pillion_1" / "pillion_2")
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
                "ordering_confident": bool(self.ordering_confident),
                "kalman_bbox": tuple(float(c) for c in self.kalman.current_bbox_xyxy),
                "velocity_xy": tuple(float(v) for v in self.kalman.velocity_xy),
            }
        )
        return self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def update_unmatched(self, frame_index: int) -> None:
        self.missed_frames.append(int(frame_index))
        self.phase10_track.update_unmatched(frame_index=frame_index)

    @property
    def temporal_state(self) -> TemporalState:
        return self.phase10_track.state

    @property
    def aggregated_confidence(self) -> dict[str, float]:
        return self.phase10_track.aggregator.get_aggregated_confidence()

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtrack_id": self.subtrack_id,
            "parent_moto_id": self.parent_moto_id,
            "camera_id": self.camera_id,
            "slot_index": self.slot_index,
            "slot_role": self.slot_role,
            "ordering_confident": self.ordering_confident,
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
    ) -> None:
        self.camera_id = camera_id
        self.numeric_id = numeric_id
        self.moto_track_id = f"{camera_id}_moto_{numeric_id:04d}"
        self.bt_config = bt_config
        self.p10_config = p10_config
        self._alloc_rider_id = rider_id_alloc_fn

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
        self.new_alerts_this_frame: list[ViolationAlert] = []
        self._sync_riders_on_match(initial_coupled.riders, frame_index, pts_timestamp_sec, is_init=True)

    def predict(self, current_frame_index: int) -> tuple[float, float, float, float]:
        dt = max(1, current_frame_index - self.last_frame_index)
        for r_sub in self.rider_subtracks.values():
            r_sub.predict(dt_frames=float(dt))
        self.current_bbox = self.kalman.predict(dt_frames=float(dt))
        return self.current_bbox

    def _sync_riders_on_match(
        self,
        coupled_riders: list[CoupledRiderObservation],
        frame_index: int,
        pts_timestamp_sec: float,
        is_init: bool = False,
    ) -> None:
        self.new_alerts_this_frame = []
        if is_init or not self.rider_subtracks:
            for r_obs in coupled_riders:
                s_idx = r_obs.slot_index
                if s_idx in self.rider_subtracks:
                    free_slots = [k for k in range(3) if k not in self.rider_subtracks]
                    if not free_slots:
                        continue
                    s_idx = free_slots[0]
                self.rider_subtracks[s_idx] = RiderSubTrack(
                    camera_id=self.camera_id,
                    parent_moto_id=self.moto_track_id,
                    numeric_track_id=self._alloc_rider_id(),
                    slot_index=s_idx,
                    slot_role=r_obs.slot_role,
                    ordering_confident=r_obs.ordering_confident,
                    initial_obs=r_obs,
                    frame_index=frame_index,
                    pts_timestamp_sec=pts_timestamp_sec,
                    p10_config=self.p10_config,
                    bt_config=self.bt_config,
                )
                self.new_alerts_this_frame.extend(self.rider_subtracks[s_idx].alerts)
            return

        if not coupled_riders:
            for sub in self.rider_subtracks.values():
                sub.update_unmatched(frame_index=frame_index)
            return

        existing_slots = sorted(self.rider_subtracks.keys())
        m_sub = len(existing_slots)
        k_obs = len(coupled_riders)

        # 1. Compare every incoming rider observation against predicted bounding boxes
        #    of all existing RiderSubTracks
        iou_mat = np.zeros((m_sub, k_obs), dtype=np.float64)
        dist_mat = np.zeros((m_sub, k_obs), dtype=np.float64)
        aff_mat = np.zeros((m_sub, k_obs), dtype=np.float64)

        for i, s_key in enumerate(existing_slots):
            pred_box = self.rider_subtracks[s_key].kalman.current_bbox_xyxy
            px1, py1, px2, py2 = pred_box
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            pcx, pcy = px1 + 0.5 * pw, py1 + 0.5 * ph
            p_diag = max(1.0, float(np.hypot(pw, ph)))
            for j, r_obs in enumerate(coupled_riders):
                obox = r_obs.head_detection.bbox
                iou_val = bbox_iou_xyxy(pred_box, obox)
                ocx, ocy = r_obs.head_detection.center
                norm_dist = float(np.hypot(pcx - ocx, pcy - ocy)) / p_diag
                iou_mat[i, j] = iou_val
                dist_mat[i, j] = norm_dist
                if iou_val >= self.bt_config.rider_match_min_iou:
                    aff_mat[i, j] = iou_val
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
                    if iou_mat[i, j] > 0.02 or dist_mat[i, j] <= self.bt_config.rider_max_norm_center_dist
                ]
                if len(cand_i) >= 2:
                    cand_sorted = sorted(
                        cand_i, key=lambda idx: (iou_mat[idx, j], -dist_mat[idx, j]), reverse=True
                    )
                    best_i, second_i = cand_sorted[0], cand_sorted[1]
                    iou_margin = float(iou_mat[best_i, j] - iou_mat[second_i, j])
                    if iou_margin < self.bt_config.rider_ambiguity_iou_margin:
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
                    and (iou_mat[i, j] > 0.02 or dist_mat[i, j] <= self.bt_config.rider_max_norm_center_dist)
                ]
                if len(cand_j) >= 2:
                    cand_j_sorted = sorted(
                        cand_j, key=lambda jdx: (iou_mat[i, jdx], -dist_mat[i, jdx]), reverse=True
                    )
                    b_j, s_j = cand_j_sorted[0], cand_j_sorted[1]
                    s_j_other_max = max(
                        [float(iou_mat[i2, s_j]) for i2 in range(m_sub) if i2 != i], default=0.0
                    )
                    if (
                        float(iou_mat[i, b_j] - iou_mat[i, s_j]) < self.bt_config.rider_ambiguity_iou_margin
                        and s_j_other_max < self.bt_config.rider_match_min_iou
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

        # 4. Evaluate spatial consistency vs. slot identity and resolve or abstain conservatively
        for i, j in zip(row_ind, col_ind):
            if cost_mat[i, j] >= 1.0:
                continue
            slot_k = existing_slots[i]
            r_obs = coupled_riders[j]
            obs_slot = r_obs.slot_index
            iou_val = float(iou_mat[i, j])
            other_sub_max_iou = max(
                [float(iou_mat[i2, j]) for i2 in range(m_sub) if i2 != i], default=0.0
            )

            if obs_slot == slot_k:
                # Slot identity and spatial assignment agree
                if (
                    not r_obs.ordering_confident
                    and m_sub >= 2
                    and (iou_val - other_sub_max_iou) < self.bt_config.rider_ambiguity_iou_margin
                ):
                    self.slot_spatial_conflicts_detected += 1
                    self.rider_abstained_updates += 1
                    self.rider_subtracks[slot_k].abstained_updates_count += 1
                    abstained_slots.add(slot_k)
                    used_obs.add(j)
                    continue
                alerts = self.rider_subtracks[slot_k].update_matched(
                    r_obs, frame_index, pts_timestamp_sec, preserve_slot_role=False
                )
                self.new_alerts_this_frame.extend(alerts)
                matched_slots.add(slot_k)
                used_obs.add(j)
            else:
                # Conflict between coupler slot_index and spatial trajectory assignment
                self.slot_spatial_conflicts_detected += 1
                is_partial_visibility_match = (
                    k_obs < m_sub
                    and iou_val >= self.bt_config.rider_match_min_iou
                    and other_sub_max_iou < 0.05
                    and (iou_val - other_sub_max_iou) >= self.bt_config.rider_ambiguity_iou_margin
                )
                is_high_conf_spatial_override = (
                    iou_val >= self.bt_config.rider_spatial_override_min_iou
                    and other_sub_max_iou < 0.05
                    and (iou_val - other_sub_max_iou) >= self.bt_config.rider_spatial_override_min_iou
                )
                if is_partial_visibility_match or is_high_conf_spatial_override:
                    self.spatial_identity_overrides += 1
                    alerts = self.rider_subtracks[slot_k].update_matched(
                        r_obs, frame_index, pts_timestamp_sec, preserve_slot_role=True
                    )
                    self.new_alerts_this_frame.extend(alerts)
                    matched_slots.add(slot_k)
                    used_obs.add(j)
                else:
                    self.rider_abstained_updates += 1
                    self.rider_subtracks[slot_k].abstained_updates_count += 1
                    abstained_slots.add(slot_k)
                    if obs_slot in self.rider_subtracks:
                        self.rider_subtracks[obs_slot].abstained_updates_count += 1
                        abstained_slots.add(obs_slot)
                    used_obs.add(j)

        # 5. Check remaining unmatched observations for slot-vs-trajectory conflicts
        #    before allowing any new RiderSubTrack creation (prevent duplicate tracks)
        for o_idx, r_obs in enumerate(coupled_riders):
            if o_idx in used_obs:
                continue
            obs_slot = r_obs.slot_index
            if obs_slot in self.rider_subtracks and obs_slot not in matched_slots:
                # Observation claims an existing unmatched slot but failed spatial consistency with it!
                self.slot_spatial_conflicts_detected += 1
                self.rider_abstained_updates += 1
                self.rider_subtracks[obs_slot].abstained_updates_count += 1
                abstained_slots.add(obs_slot)
                used_obs.add(o_idx)
                continue
            if len(matched_slots) < len(self.rider_subtracks) or len(abstained_slots) > 0:
                # At least one existing rider subtrack is unaccounted for in this frame;
                # abstain rather than spawning a duplicate rider track!
                self.slot_spatial_conflicts_detected += 1
                self.rider_abstained_updates += 1
                used_obs.add(o_idx)
                continue

            target_slot = obs_slot
            if target_slot in self.rider_subtracks:
                free_slots = [k for k in range(3) if k not in self.rider_subtracks]
                if not free_slots:
                    continue
                target_slot = free_slots[0]
            sub = RiderSubTrack(
                camera_id=self.camera_id,
                parent_moto_id=self.moto_track_id,
                numeric_track_id=self._alloc_rider_id(),
                slot_index=target_slot,
                slot_role=r_obs.slot_role,
                ordering_confident=r_obs.ordering_confident,
                initial_obs=r_obs,
                frame_index=frame_index,
                pts_timestamp_sec=pts_timestamp_sec,
                p10_config=self.p10_config,
                bt_config=self.bt_config,
            )
            self.rider_subtracks[target_slot] = sub
            matched_slots.add(target_slot)
            self.new_alerts_this_frame.extend(sub.alerts)

        # 6. Advance unmatched (or abstained) existing rider subtracks (occlusion handling per rider)
        for s_idx, sub in self.rider_subtracks.items():
            if s_idx not in matched_slots:
                sub.update_unmatched(frame_index=frame_index)

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

        for sub in self.rider_subtracks.values():
            sub.update_unmatched(frame_index=frame_index)

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
    ) -> None:
        self.camera_id = camera_id
        self.numeric_id = numeric_id
        self.orphan_track_id = f"{camera_id}_orphan_{numeric_id:04d}"
        self.bt_config = bt_config
        self.p10_config = p10_config

        head = initial_orphan.head_detection
        self.kalman = KalmanBoxFilter(
            head.bbox,
            std_weight_pos=self.bt_config.kalman_std_weight_pos,
            std_weight_vel=self.bt_config.kalman_std_weight_vel,
        )
        self.last_rejection_reason = initial_orphan.rejection_reason
        self.last_frame_index = frame_index
        self.first_pts_sec = pts_timestamp_sec
        self.last_pts_sec = pts_timestamp_sec

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
        )
        self.decision_engine = ViolationDecisionEngine(config=p10_config.decision_config)
        self.lifecycle_state = (
            TrackLifecycleState.CONFIRMED
            if self.phase10_track.hits >= bt_config.min_confirm_hits
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
        self.missed_frames: list[int] = []
        self.alerts: list[ViolationAlert] = []
        self._evaluate_alerts(frame_index, pts_timestamp_sec)

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
        return self.kalman.predict(dt_frames=float(dt))

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

        if was_lost:
            self.lifecycle_state = TrackLifecycleState.RECOVERED
        elif self.phase10_track.hits >= self.bt_config.min_confirm_hits:
            self.lifecycle_state = TrackLifecycleState.CONFIRMED
        else:
            self.lifecycle_state = TrackLifecycleState.TENTATIVE

        return self._evaluate_alerts(frame_index, pts_timestamp_sec)

    def update_unmatched(self, frame_index: int) -> None:
        self.last_frame_index = frame_index
        self.missed_frames.append(int(frame_index))
        self.phase10_track.update_unmatched(frame_index=frame_index)
        if (
            self.phase10_track.time_since_update > self.bt_config.orphan_max_lost_frames
            or self.phase10_track.state == TemporalState.TRACK_LOST
        ):
            self.lifecycle_state = TrackLifecycleState.TERMINATED
        else:
            self.lifecycle_state = TrackLifecycleState.LOST

    def to_dict(self) -> dict[str, Any]:
        return {
            "orphan_track_id": self.orphan_track_id,
            "camera_id": self.camera_id,
            "lifecycle_state": self.lifecycle_state.value,
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
    ) -> None:
        if camera_id not in ("front", "back"):
            raise ValueError(f"camera_id must be 'front' or 'back', got '{camera_id}'")
        self.camera_id = camera_id
        self.coupler_config = coupler_config or CouplerConfig()
        self.bt_config = bytetrack_config or ByteTrackConfig()
        self.p10_config = phase10_config or Phase10AdapterConfig()

        self.coupler = MotoRiderCoupler(config=self.coupler_config)

        self._next_moto_id: int = 1
        self._next_rider_id: int = 1
        self._next_orphan_id: int = 1

        self.active_moto_tracks: dict[str, MotoTrack] = {}
        self.terminated_moto_tracks: list[MotoTrack] = []

        self.active_orphan_tracks: dict[str, OrphanRiderTrack] = {}
        self.terminated_orphan_tracks: list[OrphanRiderTrack] = []

        self.frames_updated: int = 0
        self.duplicate_ticks_skipped: int = 0
        self._last_processed_frame_index: int = -1
        self._last_step_summary: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, camera_id: str, yaml_path: str | Path = "configs/tracking_config.yaml") -> CameraTrackingPipeline:
        """Instantiate a CameraTrackingPipeline from `configs/tracking_config.yaml`."""
        path = Path(yaml_path)
        if not path.exists():
            return cls(camera_id=camera_id)

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
        )

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

        for d_idx in o_unmatched_d:
            o_obs = rescued_orphans[d_idx]
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
            )
            self.active_orphan_tracks[o_track.orphan_track_id] = o_track
            new_alerts.extend(o_track.alerts)

        surviving_orphans: dict[str, OrphanRiderTrack] = {}
        for oid, o_track in self.active_orphan_tracks.items():
            if o_track.lifecycle_state == TrackLifecycleState.TERMINATED:
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
