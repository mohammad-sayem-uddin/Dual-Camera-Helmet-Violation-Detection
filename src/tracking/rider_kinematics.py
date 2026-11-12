"""Motorcycle-anchored relative rider kinematics and longitudinal role arbitration.

Provides dimensionless motorcycle-relative coordinates ``(u, v, w_tilde, h_tilde)``
anchored to parent motorcycle bounding boxes, frame-clamped inverse projection,
damped relative velocity prediction for finite coasting, and motion-aligned
multi-frame longitudinal rank scoring for driver/pillion role consensus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RelativeKinematicsConfig:
    """Configuration for motorcycle-relative rider state normalization and prediction."""

    u_min: float = -0.35
    u_max: float = 1.35
    v_min: float = -1.25
    v_max: float = 0.55
    w_min: float = 0.04
    w_max: float = 1.05
    h_min: float = 0.05
    h_max: float = 1.15
    vel_damping: float = 0.35
    pos_alpha: float = 0.85
    size_alpha: float = 0.60
    vel_beta: float = 0.45
    max_rel_vel: float = 0.35
    min_moto_dim_px: float = 1.0
    clamp_non_negative: bool = True
    min_visible_area_px: float = 4.0


def is_valid_bbox(bbox: Sequence[float], min_dim: float = 1e-3) -> bool:
    """Return True if ``bbox`` has 4 finite coordinates with positive width and height."""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if not (
        math.isfinite(x1)
        and math.isfinite(y1)
        and math.isfinite(x2)
        and math.isfinite(y2)
    ):
        return False
    return (x2 - x1) > min_dim and (y2 - y1) > min_dim


def sanitize_moto_bbox(
    moto_bbox: Sequence[float],
    min_dim: float = 1.0,
) -> tuple[float, float, float, float] | None:
    """Validate parent motorcycle box and enforce minimum width/height, or return None."""
    if len(moto_bbox) != 4:
        return None
    mx1, my1, mx2, my2 = (float(v) for v in moto_bbox)
    if not (
        math.isfinite(mx1)
        and math.isfinite(my1)
        and math.isfinite(mx2)
        and math.isfinite(my2)
    ):
        return None
    if mx2 <= mx1 or my2 <= my1:
        return None
    if (mx2 - mx1) < min_dim:
        cx = 0.5 * (mx1 + mx2)
        mx1, mx2 = cx - 0.5 * min_dim, cx + 0.5 * min_dim
    if (my2 - my1) < min_dim:
        cy = 0.5 * (my1 + my2)
        my1, my2 = cy - 0.5 * min_dim, cy + 0.5 * min_dim
    return (mx1, my1, mx2, my2)


def clamp_relative_state(
    u: float,
    v: float,
    w_tilde: float,
    h_tilde: float,
    config: RelativeKinematicsConfig | None = None,
) -> tuple[float, float, float, float]:
    """Clamp dimensionless relative state ``(u, v, w_tilde, h_tilde)`` to plausible bounds."""
    cfg = config or RelativeKinematicsConfig()
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(w_tilde) and math.isfinite(h_tilde)):
        return (0.5, -0.32, 0.30, 0.26)
    cu = min(max(float(u), cfg.u_min), cfg.u_max)
    cv = min(max(float(v), cfg.v_min), cfg.v_max)
    cw = min(max(float(w_tilde), cfg.w_min), cfg.w_max)
    ch = min(max(float(h_tilde), cfg.h_min), cfg.h_max)
    return (cu, cv, cw, ch)


def bbox_to_moto_relative(
    head_bbox: Sequence[float],
    moto_bbox: Sequence[float],
    config: RelativeKinematicsConfig | None = None,
    clamp: bool = True,
) -> tuple[tuple[float, float, float, float], bool]:
    """Transform pixel ``head_bbox`` into parent ``moto_bbox`` top-left normalized coords.

    Returns ``((u, v, w_tilde, h_tilde), valid)``. When geometry is invalid (non-finite or
    degenerate), returns a safe centered fallback with ``valid=False``.
    """
    cfg = config or RelativeKinematicsConfig()
    m_safe = sanitize_moto_bbox(moto_bbox, min_dim=cfg.min_moto_dim_px)
    if m_safe is None or not is_valid_bbox(head_bbox, min_dim=1e-3):
        return ((0.5, -0.32, 0.30, 0.26), False)
    mx1, my1, mx2, my2 = m_safe
    hx1, hy1, hx2, hy2 = (float(v) for v in head_bbox)
    mw = max(cfg.min_moto_dim_px, mx2 - mx1)
    mh = max(cfg.min_moto_dim_px, my2 - my1)
    u = (0.5 * (hx1 + hx2) - mx1) / mw
    v = (0.5 * (hy1 + hy2) - my1) / mh
    w_tilde = max(1e-4, (hx2 - hx1) / mw)
    h_tilde = max(1e-4, (hy2 - hy1) / mh)
    if clamp:
        rel = clamp_relative_state(u, v, w_tilde, h_tilde, cfg)
    else:
        rel = (u, v, w_tilde, h_tilde)
    return (rel, True)


def clamp_bbox_to_frame(
    bbox: Sequence[float],
    frame_dimensions: tuple[int, int] | None = None,
    clamp_non_negative: bool = True,
    min_visible_area_px: float = 4.0,
) -> tuple[tuple[float, float, float, float], bool]:
    """Clip ``bbox`` to valid image boundaries cleanly.

    Returns ``(clipped_bbox, is_off_frame)`` where ``is_off_frame`` is True only when
    the pre-clip box lies completely outside frame bounds (visible area < ``min_visible_area_px``).
    """
    if len(bbox) != 4:
        return ((0.0, 0.0, 2.0, 2.0), True)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return ((0.0, 0.0, 2.0, 2.0), True)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    raw_w = max(1.0, x2 - x1)
    raw_h = max(1.0, y2 - y1)

    if frame_dimensions is not None:
        fw = float(frame_dimensions[0])
        fh = float(frame_dimensions[1])
        if fw <= 1.0 or fh <= 1.0:
            return ((0.0, 0.0, 2.0, 2.0), True)
        cx1 = min(max(x1, 0.0), fw)
        cy1 = min(max(y1, 0.0), fh)
        cx2 = min(max(x2, 0.0), fw)
        cy2 = min(max(y2, 0.0), fh)
        vis_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        if vis_area < min_visible_area_px:
            # Clamp to nearest visible minimal sliver along boundary, mark off_frame=True
            ax1 = min(max(x1, 0.0), max(0.0, fw - 2.0))
            ay1 = min(max(y1, 0.0), max(0.0, fh - 2.0))
            return ((ax1, ay1, min(fw, ax1 + 2.0), min(fh, ay1 + 2.0)), True)
        return ((cx1, cy1, cx2, cy2), False)

    if clamp_non_negative:
        nx1 = max(0.0, x1)
        ny1 = max(0.0, y1)
        nx2 = max(nx1 + min(raw_w, 2.0), x2)
        ny2 = max(ny1 + min(raw_h, 2.0), y2)
        off_frame = (x2 <= 0.0) or (y2 <= 0.0)
        return ((nx1, ny1, nx2, ny2), off_frame)

    return ((x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2)), False)


def moto_relative_to_bbox(
    rel_state: Sequence[float],
    moto_bbox: Sequence[float],
    frame_dimensions: tuple[int, int] | None = None,
    config: RelativeKinematicsConfig | None = None,
) -> tuple[tuple[float, float, float, float], bool]:
    """Back-project relative state ``(u, v, w_tilde, h_tilde)`` onto ``moto_bbox`` pixels."""
    cfg = config or RelativeKinematicsConfig()
    m_safe = sanitize_moto_bbox(moto_bbox, min_dim=cfg.min_moto_dim_px)
    if m_safe is None or len(rel_state) != 4:
        return ((0.0, 0.0, 2.0, 2.0), True)
    u, v, wt, ht = clamp_relative_state(
        float(rel_state[0]),
        float(rel_state[1]),
        float(rel_state[2]),
        float(rel_state[3]),
        cfg,
    )
    mx1, my1, mx2, my2 = m_safe
    mw = mx2 - mx1
    mh = my2 - my1
    cx = mx1 + u * mw
    cy = my1 + v * mh
    bw = wt * mw
    bh = ht * mh
    raw_box = (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)
    return clamp_bbox_to_frame(
        raw_box,
        frame_dimensions=frame_dimensions,
        clamp_non_negative=cfg.clamp_non_negative,
        min_visible_area_px=cfg.min_visible_area_px,
    )


def relative_distance(
    rel_a: Sequence[float],
    rel_b: Sequence[float],
) -> float:
    """Normalized Euclidean centroid distance in motorcycle-relative ``(u, v)`` coordinates."""
    if len(rel_a) < 2 or len(rel_b) < 2:
        return float("inf")
    du = float(rel_a[0]) - float(rel_b[0])
    dv = float(rel_a[1]) - float(rel_b[1])
    if not (math.isfinite(du) and math.isfinite(dv)):
        return float("inf")
    return math.hypot(du, dv)


def relative_scale_consistency(
    rel_a: Sequence[float],
    rel_b: Sequence[float],
) -> float:
    """Return worst-axis normalized scale ratio in ``[0.0, 1.0]`` between two relative sizes."""
    if len(rel_a) < 4 or len(rel_b) < 4:
        return 0.0
    wa, ha = max(1e-4, float(rel_a[2])), max(1e-4, float(rel_a[3]))
    wb, hb = max(1e-4, float(rel_b[2])), max(1e-4, float(rel_b[3]))
    rw = min(wa, wb) / max(wa, wb)
    rh = min(ha, hb) / max(ha, hb)
    return min(rw, rh)


class RiderRelativeKalmanState:
    """Motorcycle-anchored relative kinematics filter with damped coast prediction.

    Tracks the rider head in parent-normalized ``(u, v, w_tilde, h_tilde)`` coordinates
    plus exponentially smoothed relative velocity ``(du, dv)``. When predicting across
    unmatched frames, relative velocity decays exponentially via ``rho ** steps`` while
    re-projecting against the live parent ``moto_bbox``.
    """

    def __init__(
        self,
        initial_head_bbox: Sequence[float],
        initial_moto_bbox: Sequence[float],
        config: RelativeKinematicsConfig | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> None:
        self.config = config or RelativeKinematicsConfig()
        rel, valid = bbox_to_moto_relative(initial_head_bbox, initial_moto_bbox, self.config)
        self.u: float = rel[0]
        self.v: float = rel[1]
        self.w_tilde: float = rel[2]
        self.h_tilde: float = rel[3]
        self.du: float = 0.0
        self.dv: float = 0.0
        self.valid_initialized: bool = valid
        self.steps_coasted: int = 0
        self.last_off_frame: bool = False
        self.last_moto_bbox: tuple[float, float, float, float] = (
            sanitize_moto_bbox(initial_moto_bbox, self.config.min_moto_dim_px)
            or (0.0, 0.0, 100.0, 100.0)
        )
        proj, off_f = moto_relative_to_bbox(
            self.rel_tuple,
            self.last_moto_bbox,
            frame_dimensions=frame_dimensions,
            config=self.config,
        )
        self.current_bbox_xyxy: tuple[float, float, float, float] = proj
        self.last_off_frame = off_f

    @property
    def rel_tuple(self) -> tuple[float, float, float, float]:
        return (self.u, self.v, self.w_tilde, self.h_tilde)

    def predict(
        self,
        moto_bbox: Sequence[float] | None = None,
        dt_frames: float = 1.0,
        frame_dimensions: tuple[int, int] | None = None,
        coasting: bool = True,
    ) -> tuple[float, float, float, float]:
        """Predict rider bbox anchored to ``moto_bbox`` with damped relative velocity."""
        if moto_bbox is not None:
            m_safe = sanitize_moto_bbox(moto_bbox, self.config.min_moto_dim_px)
            if m_safe is not None:
                self.last_moto_bbox = m_safe
        dt = max(1.0, min(float(dt_frames), 10.0))
        if coasting:
            self.steps_coasted += 1
            decay = self.config.vel_damping ** max(1, self.steps_coasted)
        else:
            decay = self.config.vel_damping
        pred_u = self.u + decay * self.du * dt
        pred_v = self.v + decay * self.dv * dt
        cu, cv, cw, ch = clamp_relative_state(
            pred_u, pred_v, self.w_tilde, self.h_tilde, self.config
        )
        self.u, self.v, self.w_tilde, self.h_tilde = cu, cv, cw, ch
        self.du *= decay
        self.dv *= decay
        proj, off_f = moto_relative_to_bbox(
            self.rel_tuple,
            self.last_moto_bbox,
            frame_dimensions=frame_dimensions,
            config=self.config,
        )
        self.current_bbox_xyxy = proj
        self.last_off_frame = off_f
        return proj

    def update(
        self,
        obs_head_bbox: Sequence[float],
        moto_bbox: Sequence[float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
        dt_frames: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """Filter observed head box in parent motorcycle coordinates."""
        if moto_bbox is not None:
            m_safe = sanitize_moto_bbox(moto_bbox, self.config.min_moto_dim_px)
            if m_safe is not None:
                self.last_moto_bbox = m_safe
        obs_rel, valid = bbox_to_moto_relative(obs_head_bbox, self.last_moto_bbox, self.config)
        if valid:
            dt = max(1.0, float(dt_frames))
            ou, ov, ow, oh = obs_rel
            inst_du = min(
                max((ou - self.u) / dt, -self.config.max_rel_vel),
                self.config.max_rel_vel,
            )
            inst_dv = min(
                max((ov - self.v) / dt, -self.config.max_rel_vel),
                self.config.max_rel_vel,
            )
            beta = self.config.vel_beta
            self.du = min(
                max(beta * inst_du + (1.0 - beta) * self.du, -self.config.max_rel_vel),
                self.config.max_rel_vel,
            )
            self.dv = min(
                max(beta * inst_dv + (1.0 - beta) * self.dv, -self.config.max_rel_vel),
                self.config.max_rel_vel,
            )
            ap = self.config.pos_alpha
            asize = self.config.size_alpha
            cu, cv, cw, ch = clamp_relative_state(
                ap * ou + (1.0 - ap) * self.u,
                ap * ov + (1.0 - ap) * self.v,
                asize * ow + (1.0 - asize) * self.w_tilde,
                asize * oh + (1.0 - asize) * self.h_tilde,
                self.config,
            )
            self.u, self.v, self.w_tilde, self.h_tilde = cu, cv, cw, ch
            self.steps_coasted = 0
        proj, off_f = moto_relative_to_bbox(
            self.rel_tuple,
            self.last_moto_bbox,
            frame_dimensions=frame_dimensions,
            config=self.config,
        )
        self.current_bbox_xyxy = proj
        self.last_off_frame = off_f
        return proj


def compute_moto_travel_axis(
    velocity_xy: Sequence[float],
    min_speed_px: float = 2.0,
) -> tuple[tuple[float, float] | None, float]:
    """Return normalized motorcycle travel unit vector ``(dx, dy)`` and speed if above floor."""
    if len(velocity_xy) != 2:
        return None, 0.0
    vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
    if not (math.isfinite(vx) and math.isfinite(vy)):
        return None, 0.0
    speed = math.hypot(vx, vy)
    if speed < min_speed_px:
        return None, speed
    return (vx / speed, vy / speed), speed


def score_longitudinal_ranks(
    centroids_by_slot: Mapping[int, tuple[float, float]],
    camera_id: str,
    velocity_xy: Sequence[float],
    min_speed_px: float = 2.0,
    min_sep_px: float = 8.0,
) -> tuple[list[int] | None, str]:
    """Order multi-rider slots along travel axis (motion vector primary, perspective fallback).

    Returns ``(ordered_slots_driver_first, mode_label)`` or ``(None, reason)`` if riders
    are insufficiently separated along the longitudinal axis for a confident rank vote.
    """
    if len(centroids_by_slot) < 2:
        return None, "single_or_empty"

    travel_dir, _ = compute_moto_travel_axis(velocity_xy, min_speed_px=min_speed_px)
    items = list(centroids_by_slot.items())

    if travel_dir is not None:
        dx, dy = travel_dir
        # Project onto forward travel vector (higher = ahead along motion direction)
        projs = [(slot, cx * dx + cy * dy) for slot, (cx, cy) in items]
        projs_sorted = sorted(projs, key=lambda p: (-p[1], p[0]))
        min_gap = min(
            abs(projs_sorted[i][1] - projs_sorted[i + 1][1])
            for i in range(len(projs_sorted) - 1)
        )
        if min_gap >= min_sep_px:
            return [slot for slot, _ in projs_sorted], "motion_vector"

    # Perspective fallback: front camera approaching -> higher y is closer/driver;
    # back camera receding -> lower y is further forward/driver.
    if camera_id == "front":
        ordered_y = sorted(items, key=lambda kv: (-kv[1][1], kv[0]))
    else:
        ordered_y = sorted(items, key=lambda kv: (kv[1][1], kv[0]))
    min_y_gap = min(
        abs(ordered_y[i][1][1] - ordered_y[i + 1][1][1])
        for i in range(len(ordered_y) - 1)
    )
    if min_y_gap >= min_sep_px:
        return [slot for slot, _ in ordered_y], "perspective_y"

    return None, "insufficient_longitudinal_separation"


def role_name_for_rank(rank: int) -> str:
    """Canonical slot role name for integer longitudinal rank (0=driver, 1+=pillion_k)."""
    if rank == 0:
        return "driver"
    return f"pillion_{rank}"
