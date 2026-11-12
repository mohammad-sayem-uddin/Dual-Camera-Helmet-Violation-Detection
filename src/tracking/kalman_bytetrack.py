"""Motorcycle-Level Kalman Filter and Two-Stage ByteTrack Association Engine.

Implements an 8-state Constant-Velocity Kalman Filter `[cx, cy, a, h, vx, vy, va, vh]`
and Two-Stage ByteTrack association (Stage 1: high-confidence detections; Stage 2:
low-confidence occluded/blurred detections) with explicit 5-state track lifecycle
management (`TENTATIVE`, `CONFIRMED`, `LOST`, `RECOVERED`, `TERMINATED`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackLifecycleState(str, Enum):
    """Explicit 5-state lifecycle for per-camera motorcycle and orphan tracks."""
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    RECOVERED = "RECOVERED"
    TERMINATED = "TERMINATED"


@dataclass
class ByteTrackConfig:
    """Configurable parameters for Kalman filtering, two-stage ByteTrack association, and rider identity protection."""
    high_conf_threshold: float = 0.45
    low_conf_threshold: float = 0.18
    new_track_min_conf: float = 0.45
    stage1_iou_threshold: float = 0.25
    stage2_iou_threshold: float = 0.20
    min_confirm_hits: int = 2
    max_lost_frames: int = 10
    orphan_iou_threshold: float = 0.25
    orphan_max_lost_frames: int = 8
    # Kalman filter process & measurement noise scale factors (hypotheses; 1/20 and 1/160)
    kalman_std_weight_pos: float = 0.05
    kalman_std_weight_vel: float = 0.00625
    # RiderSubTrack spatial identity protection thresholds (engineering hypotheses)
    rider_match_min_iou: float = 0.15
    rider_spatial_override_min_iou: float = 0.35
    rider_ambiguity_iou_margin: float = 0.15
    rider_max_norm_center_dist: float = 0.65
    # GAP-C: Motorcycle-anchored rider kinematics, finite coasting & role/orphan controls
    use_motorcycle_relative_kinematics: bool = True
    rel_damping_factor: float = 0.20
    rel_pos_alpha: float = 0.70
    rel_size_alpha: float = 0.50
    rel_vel_beta: float = 0.25
    rel_max_vel: float = 0.35
    rider_rel_match_max_dist: float = 0.58
    rider_rel_ambiguity_margin: float = 0.08
    finite_rider_coast_enabled: bool = True
    max_rider_coast_frames: int = 15
    tentative_rider_coast_frames: int = 3
    unlock_slot_conflicts: bool = True
    duplicate_head_iou_threshold: float = 0.55
    sibling_overlap_max_iou: float = 0.35
    conflict_distinct_min_rel_dist: float = 0.58
    max_active_riders_per_moto: int = 3
    max_total_subtracks_per_moto: int = 6
    role_consensus_enabled: bool = True
    min_role_consensus_frames: int = 3
    role_min_moto_speed_px: float = 2.0
    role_min_proj_sep_px: float = 8.0
    orphan_lifecycle_gating_enabled: bool = True
    orphan_min_confirm_hits: int = 2
    orphan_tentative_max_lost_frames: int = 3
    orphan_tentative_min_conf: float = 0.18
    export_confirmed_orphans_only: bool = True
    frame_width: int | None = None
    frame_height: int | None = None
    clamp_non_negative_coords: bool = True


def bbox_iou_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


class KalmanBoxFilter:
    """
    8-state Constant-Velocity Kalman Filter for bounding box tracking in image space.
    State vector x = [cx, cy, a, h, vx, vy, va, vh]^T, where:
      - (cx, cy): bounding box center in pixels
      - a = w / h: aspect ratio
      - h: bounding box height in pixels
    """

    def __init__(
        self,
        initial_bbox_xyxy: tuple[float, float, float, float],
        std_weight_pos: float = 0.05,
        std_weight_vel: float = 0.00625,
    ) -> None:
        self._std_weight_pos = float(std_weight_pos)
        self._std_weight_vel = float(std_weight_vel)

        z = self._xyxy_to_cxcyah(initial_bbox_xyxy)
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z

        h = max(10.0, float(z[3]))
        std = np.array([
            2.0 * self._std_weight_pos * h,
            2.0 * self._std_weight_pos * h,
            1e-2,
            2.0 * self._std_weight_pos * h,
            10.0 * self._std_weight_vel * h,
            10.0 * self._std_weight_vel * h,
            1e-5,
            10.0 * self._std_weight_vel * h,
        ], dtype=np.float64)
        self.P = np.diag(np.square(std))

        self.H = np.eye(4, 8, dtype=np.float64)
        self._synced_bbox_xyxy: tuple[float, float, float, float] | None = None

    @staticmethod
    def _xyxy_to_cxcyah(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float(x1) + 0.5 * w
        cy = float(y1) + 0.5 * h
        a = w / h
        return np.array([cx, cy, a, h], dtype=np.float64)

    @staticmethod
    def _cxcyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
        cx, cy, a, h = float(state[0]), float(state[1]), max(0.05, float(state[2])), max(1.0, float(state[3]))
        w = a * h
        return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)

    def predict(self, dt_frames: float = 1.0) -> tuple[float, float, float, float]:
        """Advance state and covariance by dt_frames steps and return predicted xyxy box."""
        dt = max(1.0, float(dt_frames))
        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = dt

        h = max(10.0, float(self.x[3]))
        std_pos = [
            self._std_weight_pos * h * dt,
            self._std_weight_pos * h * dt,
            1e-2 * dt,
            self._std_weight_pos * h * dt,
        ]
        std_vel = [
            self._std_weight_vel * h * dt,
            self._std_weight_vel * h * dt,
            1e-5 * dt,
            self._std_weight_vel * h * dt,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.x = F @ self.x
        self.x[2] = max(0.05, float(self.x[2]))
        self.x[3] = max(2.0, float(self.x[3]))
        self.P = F @ self.P @ F.T + Q
        self._synced_bbox_xyxy = None
        return self._cxcyah_to_xyxy(self.x)

    def update(self, bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Correct Kalman state with observed bounding box and return updated xyxy box."""
        z = self._xyxy_to_cxcyah(bbox_xyxy)
        h = max(10.0, float(self.x[3]))
        std_meas = np.array([
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-1,
            self._std_weight_pos * h,
        ], dtype=np.float64)
        R = np.diag(np.square(std_meas))

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        I_KH = np.eye(8, dtype=np.float64) - (K @ self.H)
        self.x = self.x + (K @ y)
        self.x[2] = max(0.05, float(self.x[2]))
        self.x[3] = max(2.0, float(self.x[3]))
        # Joseph stabilized covariance update
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._synced_bbox_xyxy = None
        return self._cxcyah_to_xyxy(self.x)

    def sync_bbox_state(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        vx: float | None = None,
        vy: float | None = None,
    ) -> None:
        """Synchronize stored spatial box (and optional pixel velocity) from anchored state."""
        z = self._xyxy_to_cxcyah(bbox_xyxy)
        self.x[:4] = z
        self._synced_bbox_xyxy = (
            float(bbox_xyxy[0]),
            float(bbox_xyxy[1]),
            float(bbox_xyxy[2]),
            float(bbox_xyxy[3]),
        )
        if vx is not None:
            self.x[4] = float(vx)
        if vy is not None:
            self.x[5] = float(vy)

    @property
    def current_bbox_xyxy(self) -> tuple[float, float, float, float]:
        if self._synced_bbox_xyxy is not None:
            return self._synced_bbox_xyxy
        return self._cxcyah_to_xyxy(self.x)

    @property
    def velocity_xy(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))


def hungarian_iou_match(
    track_boxes: list[tuple[float, float, float, float]],
    det_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Solve minimum-cost bipartite matching between predicted track boxes and detection boxes
    using IoU affinity gated at `iou_threshold`.
    Returns (matches_with_iou, unmatched_track_indices, unmatched_det_indices).
    """
    n_t = len(track_boxes)
    n_d = len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    iou_matrix = np.zeros((n_t, n_d), dtype=np.float64)
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            iou_matrix[i, j] = bbox_iou_xyxy(tb, db)

    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, float]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        iou_val = float(iou_matrix[r, c])
        if iou_val >= iou_threshold:
            matches.append((int(r), int(c), iou_val))
            matched_t.add(int(r))
            matched_d.add(int(c))

    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_d = [j for j in range(n_d) if j not in matched_d]
    return matches, unmatched_t, unmatched_d


def two_stage_bytetrack_assign(
    track_boxes: list[tuple[float, float, float, float]],
    det_boxes: list[tuple[float, float, float, float]],
    det_confidences: list[float],
    config: ByteTrackConfig,
    track_can_use_stage2: list[bool] | None = None,
) -> dict[str, Any]:
    """
    Perform Two-Stage ByteTrack association:
      - Stage 1 matches all eligible tracks against high-confidence detections (conf >= high_conf_threshold).
      - Stage 2 matches remaining confirmed/lost tracks against low-confidence detections
        (low_conf_threshold <= conf < high_conf_threshold) to recover occluded/blurred objects.
    """
    high_det_indices = [
        idx for idx, c in enumerate(det_confidences) if c >= config.high_conf_threshold
    ]
    low_det_indices = [
        idx
        for idx, c in enumerate(det_confidences)
        if config.low_conf_threshold <= c < config.high_conf_threshold
    ]

    high_boxes = [det_boxes[i] for i in high_det_indices]
    s1_matches_local, s1_unmatched_t, s1_unmatched_high_local = hungarian_iou_match(
        track_boxes, high_boxes, config.stage1_iou_threshold
    )

    stage1_matches = [
        (t_idx, high_det_indices[d_local], iou_val)
        for t_idx, d_local, iou_val in s1_matches_local
    ]
    unmatched_high_det_indices = [high_det_indices[d_local] for d_local in s1_unmatched_high_local]

    # Filter remaining tracks eligible for Stage 2 (typically CONFIRMED, RECOVERED, or LOST tracks)
    if track_can_use_stage2 is not None:
        stage2_candidate_t = [t_idx for t_idx in s1_unmatched_t if track_can_use_stage2[t_idx]]
        stage2_ineligible_t = [t_idx for t_idx in s1_unmatched_t if not track_can_use_stage2[t_idx]]
    else:
        stage2_candidate_t = list(s1_unmatched_t)
        stage2_ineligible_t = []

    stage2_track_boxes = [track_boxes[t_idx] for t_idx in stage2_candidate_t]
    low_boxes = [det_boxes[i] for i in low_det_indices]

    s2_matches_local, s2_unmatched_t_local, s2_unmatched_low_local = hungarian_iou_match(
        stage2_track_boxes, low_boxes, config.stage2_iou_threshold
    )

    stage2_matches = [
        (stage2_candidate_t[t_local], low_det_indices[d_local], iou_val)
        for t_local, d_local, iou_val in s2_matches_local
    ]
    final_unmatched_tracks = (
        [stage2_candidate_t[t_local] for t_local in s2_unmatched_t_local]
        + stage2_ineligible_t
    )
    unmatched_low_det_indices = [low_det_indices[d_local] for d_local in s2_unmatched_low_local]

    return {
        "stage1_matches": stage1_matches,
        "stage2_matches": stage2_matches,
        "all_matches": stage1_matches + stage2_matches,
        "unmatched_tracks": final_unmatched_tracks,
        "unmatched_high_detections": unmatched_high_det_indices,
        "unmatched_low_detections": unmatched_low_det_indices,
    }
