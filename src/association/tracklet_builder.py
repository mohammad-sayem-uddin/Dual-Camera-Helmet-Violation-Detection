"""Per-Camera Tracklet Extraction and Summary Builder (`TrackletBuilder`).

Converts per-camera `MotoTrack`, `RiderSubTrack`, and `OrphanRiderTrack` objects
from Phase 12D (`CameraTrackingPipeline`) into structured, versioned, JSON-serializable
`MotorcycleTracklet`, `RiderTracklet`, and `OrphanRiderTracklet` objects.

Enforces Section 3 Invariants:
1. Strictly separates observed frames (`ObservedFrameRecord`) from Kalman-predicted
   frames (`predicted_only_frame_count`, `occlusion_intervals`).
2. Never uses purely predicted frames as measured appearance or bounding-box geometry.
3. Places both Front and Back tracklets onto the canonical Phase 12C reference clock
   (`t_ref = t_back` for Back; `t_ref = t_front + global_offset_sec` for Front) while
   preserving raw stream PTS and synchronization uncertainty (`TimestampQualityRecord`).
4. Preserves per-camera Phase 10 verification states (`TemporalState`, `aggregated_confidence`,
   consecutive frame counters, alerts) as read-only snapshots without modifying them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from src.association.appearance_features import (
    AppearanceConfig,
    extract_tracklet_appearance_descriptor,
)
from src.association.coordinate_space import (
    boxes_consistent_with_frame,
    is_box_near_border,
    normalize_dimensions,
)
from src.association.tracklet_models import (
    AppearanceDescriptorRecord,
    MotorcycleTracklet,
    ObservedFrameRecord,
    OrphanRiderTracklet,
    RiderTracklet,
    SCHEMA_VERSION,
    TimestampQualityRecord,
    TrackletCompleteness,
)
from src.tracking.camera_tracker import (
    CameraTrackingPipeline,
    HelmetEvidenceAccumulator,
    MotoTrack,
    OrphanRiderTrack,
    RiderSubTrack,
)
from src.tracking.kalman_bytetrack import TrackLifecycleState


@dataclass
class TrackletBuilderConfig:
    """Configuration for `TrackletBuilder`."""
    min_observed_frames_moto: int = 2
    min_observed_frames_rider: int = 2
    min_observed_frames_orphan: int = 2
    min_observation_density: float = 0.30
    border_margin_px: float = 45.0
    exclude_tentative_tracks: bool = False
    # Optional EXPLICIT override of the frame size (pixels, in the bounding-box
    # coordinate space). Default None: dimensions must come from the per-call
    # `frame_dimensions` (authoritative decoded-frame size). There is no implicit
    # 1920x1080 fallback; unknown geometry yields an explicit "unknown" border status.
    frame_width_px: int | None = None
    frame_height_px: int | None = None
    appearance_config: AppearanceConfig = field(default_factory=AppearanceConfig)


def _compute_occlusion_intervals(observed_frame_indices: list[int]) -> list[tuple[int, int]]:
    """
    Given sorted observed frame indices, return contiguous `[start_missed, end_missed]`
    frame index intervals where the track was predicted without a matched observation.
    """
    if len(observed_frame_indices) < 2:
        return []
    ordered = sorted(set(int(f) for f in observed_frame_indices))
    intervals: list[tuple[int, int]] = []
    for prev_f, next_f in zip(ordered[:-1], ordered[1:]):
        if next_f - prev_f > 1:
            intervals.append((prev_f + 1, next_f - 1))
    return intervals


def _is_bbox_near_border(
    bbox_xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    margin_px: float,
) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    return bool(
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= (frame_width - margin_px)
        or y2 >= (frame_height - margin_px)
    )


BORDER_UNKNOWN_REASON = "BORDER_GEOMETRY_UNKNOWN_NO_FRAME_DIMENSIONS"
BORDER_MISMATCH_REASON = "BORDER_COORDINATE_SPACE_MISMATCH"


def _bbox_area(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class TrackletBuilder:
    """
    Builds `MotorcycleTracklet`, `RiderTracklet`, and `OrphanRiderTracklet` objects
    from single-camera tracking pipelines or track lists.
    """

    def __init__(self, config: TrackletBuilderConfig | None = None) -> None:
        self.config = config or TrackletBuilderConfig()

    def _resolve_frame_dimensions(
        self, frame_dimensions: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        """Per-call authoritative `(width, height)` first, then explicit config override."""
        dims = normalize_dimensions(frame_dimensions)
        if dims is not None:
            return dims
        if self.config.frame_width_px is not None and self.config.frame_height_px is not None:
            return normalize_dimensions((self.config.frame_width_px, self.config.frame_height_px))
        return None

    def _border_flags(
        self,
        entry_box: tuple[float, float, float, float],
        exit_box: tuple[float, float, float, float],
        all_boxes: list[tuple[float, float, float, float]],
        dims: tuple[int, int] | None,
    ) -> tuple[bool, bool, str | None]:
        """Return `(entry_near_border, exit_near_border, status_reason)`.

        `status_reason` is None when classification is valid; otherwise the flags are
        False (not classified) and the reason documents why.
        """
        if dims is None:
            return False, False, BORDER_UNKNOWN_REASON
        if not boxes_consistent_with_frame(all_boxes or [entry_box, exit_box], dims):
            return False, False, BORDER_MISMATCH_REASON
        m = self.config.border_margin_px
        return (
            is_box_near_border(entry_box, dims, m),
            is_box_near_border(exit_box, dims, m),
            None,
        )

    @staticmethod
    def _export_helmet_evidence(track: Any, ema_ac: dict[str, float]) -> dict[str, Any]:
        """Copy the TASK-04 export of a source track (single implementation: the tracker's).

        A track without a `helmet_evidence` accumulator exports an EMPTY accumulator, whose
        canonical state is UNCERTAIN with zero observations - no evidence is invented.
        """
        acc = getattr(track, "helmet_evidence", None)
        if not isinstance(acc, HelmetEvidenceAccumulator):
            acc = HelmetEvidenceAccumulator()
        return acc.export(ema_ac)

    def _to_canonical_pts(
        self,
        camera_id: str,
        frame_index: int,
        camera_pts_sec: float,
        timestamp_quality: TimestampQualityRecord,
        canonical_pts_by_frame: dict[int, float] | None = None,
    ) -> float:
        if canonical_pts_by_frame and frame_index in canonical_pts_by_frame:
            return float(canonical_pts_by_frame[frame_index])
        if camera_id == "front":
            return float(camera_pts_sec + timestamp_quality.global_offset_sec)
        return float(camera_pts_sec)

    def build_rider_tracklet(
        self,
        subtrack: RiderSubTrack,
        timestamp_quality: TimestampQualityRecord,
        frame_images_by_index: dict[int, np.ndarray] | None = None,
        canonical_pts_by_frame: dict[int, float] | None = None,
        delta_pts_ms_by_frame: dict[int, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> RiderTracklet:
        """Build a `RiderTracklet` from a `RiderSubTrack`."""
        obs_records: list[ObservedFrameRecord] = []
        raw_logs = getattr(subtrack, "observation_log", None)

        if raw_logs:
            seen_frames: set[int] = set()
            for entry in raw_logs:
                f_idx = int(entry["frame_index"])
                if f_idx in seen_frames:
                    continue
                seen_frames.add(f_idx)
                cam_pts = float(entry["pts_timestamp_sec"])
                can_pts = self._to_canonical_pts(
                    subtrack.camera_id, f_idx, cam_pts, timestamp_quality, canonical_pts_by_frame
                )
                d_ms = (
                    float(delta_pts_ms_by_frame[f_idx])
                    if delta_pts_ms_by_frame and f_idx in delta_pts_ms_by_frame
                    else (
                        timestamp_quality.mean_delta_pts_front_ms
                        if subtrack.camera_id == "front"
                        else timestamp_quality.mean_delta_pts_back_ms
                    )
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=f_idx,
                        camera_pts_sec=cam_pts,
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in entry["head_bbox"]),
                        kalman_bbox_xyxy=tuple(
                            float(c) for c in entry.get("kalman_bbox", entry["head_bbox"])
                        ),
                        confidence=float(entry["head_conf"]),
                        velocity_xy=tuple(float(v) for v in entry.get("velocity_xy", (0.0, 0.0))),
                        person_bbox_xyxy=(
                            tuple(float(c) for c in entry["person_bbox"])
                            if entry.get("person_bbox") is not None
                            else None
                        ),
                        person_confidence=(
                            float(entry["person_conf"])
                            if entry.get("person_conf") is not None
                            else None
                        ),
                        head_class_name=str(entry.get("head_class_name", "")),
                        slot_role=str(entry.get("slot_role", subtrack.slot_role)),
                        ordering_confident=bool(
                            entry.get("ordering_confident", subtrack.ordering_confident)
                        ),
                        delta_pts_ms=d_ms,
                    )
                )
        else:
            seen_frames = set()
            for f_idx, cam_pts, h_bbox, h_cls, h_conf in subtrack.pts_history:
                if int(f_idx) in seen_frames:
                    continue
                seen_frames.add(int(f_idx))
                can_pts = self._to_canonical_pts(
                    subtrack.camera_id, int(f_idx), float(cam_pts), timestamp_quality, canonical_pts_by_frame
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=int(f_idx),
                        camera_pts_sec=float(cam_pts),
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in h_bbox),
                        kalman_bbox_xyxy=tuple(float(c) for c in h_bbox),
                        confidence=float(h_conf),
                        head_class_name=str(h_cls),
                        slot_role=subtrack.slot_role,
                        ordering_confident=subtrack.ordering_confident,
                    )
                )

        obs_records.sort(key=lambda r: r.frame_index)
        obs_frames = [r.frame_index for r in obs_records]
        occ_intervals = _compute_occlusion_intervals(obs_frames)
        obs_count = len(obs_records)
        if obs_count > 0:
            first_f = obs_records[0].frame_index
            last_f = obs_records[-1].frame_index
            span_frames = max(obs_count, last_f - first_f + 1)
            t_start = obs_records[0].canonical_pts_sec
            t_end = obs_records[-1].canonical_pts_sec
            t_centroid = float(np.mean([r.canonical_pts_sec for r in obs_records]))
            cam_start = obs_records[0].camera_pts_sec
            cam_end = obs_records[-1].camera_pts_sec
            entry_box = obs_records[0].bbox_xyxy
            exit_box = obs_records[-1].bbox_xyxy
            mean_conf = float(np.mean([r.confidence for r in obs_records]))
        else:
            span_frames = 0
            t_start = t_end = t_centroid = cam_start = cam_end = 0.0
            entry_box = exit_box = subtrack.kalman.current_bbox_xyxy
            mean_conf = 0.0

        missed_count = max(0, span_frames - obs_count)
        density = float(obs_count / max(1, span_frames))

        if obs_count < self.config.min_observed_frames_rider:
            completeness = TrackletCompleteness.SHORT_OBSERVATION
        elif density < self.config.min_observation_density:
            completeness = TrackletCompleteness.HEAVILY_OCCLUDED
        else:
            completeness = TrackletCompleteness.COMPLETE

        app_desc = extract_tracklet_appearance_descriptor(
            observed_frames=obs_records,
            frame_images_by_index=frame_images_by_index,
            config=self.config.appearance_config,
            is_motorcycle_tracklet=False,
            native_frame_dimensions=self._resolve_frame_dimensions(frame_dimensions),
        )

        coupling_scores = [
            float(e.get("coupling_score", subtrack.last_coupling_score))
            for e in (raw_logs or [])
        ]
        mean_coupling = (
            float(np.mean(coupling_scores)) if coupling_scores else float(subtrack.last_coupling_score)
        )

        return RiderTracklet(
            schema_version=SCHEMA_VERSION,
            camera_id=subtrack.camera_id,
            subtrack_id=subtrack.subtrack_id,
            parent_moto_track_id=subtrack.parent_moto_id,
            slot_index=int(subtrack.slot_index),
            slot_role=str(subtrack.slot_role),
            ordering_confident=bool(subtrack.ordering_confident),
            t_start_sec=t_start,
            t_end_sec=t_end,
            t_centroid_sec=t_centroid,
            camera_pts_start_sec=cam_start,
            camera_pts_end_sec=cam_end,
            observed_frame_count=obs_count,
            total_span_frames=span_frames,
            missed_frame_count=missed_count,
            observation_density=density,
            occlusion_intervals=occ_intervals,
            entry_head_bbox=entry_box,
            exit_head_bbox=exit_box,
            mean_head_confidence=mean_conf,
            mean_coupling_score=mean_coupling,
            abstained_updates_count=int(getattr(subtrack, "abstained_updates_count", 0)),
            phase10_temporal_state=subtrack.temporal_state.value,
            phase10_aggregated_confidence=dict(subtrack.aggregated_confidence),
            phase10_consecutive_no_helmet_frames=int(
                subtrack.phase10_track.state_machine.consecutive_no_helmet_frames
            ),
            phase10_consecutive_helmet_frames=int(
                subtrack.phase10_track.state_machine.consecutive_helmet_frames
            ),
            phase10_alerts_count=len(subtrack.alerts),
            completeness=completeness,
            observed_frames=obs_records,
            appearance_descriptor=app_desc,
            genuine_helmet_evidence=self._export_helmet_evidence(subtrack, subtrack.aggregated_confidence),
        )

    def build_motorcycle_tracklet(
        self,
        moto_track: MotoTrack,
        timestamp_quality: TimestampQualityRecord,
        frame_images_by_index: dict[int, np.ndarray] | None = None,
        canonical_pts_by_frame: dict[int, float] | None = None,
        delta_pts_ms_by_frame: dict[int, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> MotorcycleTracklet:
        """Build a `MotorcycleTracklet` from a single-camera `MotoTrack`."""
        obs_records: list[ObservedFrameRecord] = []
        raw_logs = getattr(moto_track, "observation_log", None)

        if raw_logs:
            seen_frames: set[int] = set()
            for entry in raw_logs:
                f_idx = int(entry["frame_index"])
                if f_idx in seen_frames:
                    continue
                seen_frames.add(f_idx)
                cam_pts = float(entry["pts_timestamp_sec"])
                can_pts = self._to_canonical_pts(
                    moto_track.camera_id, f_idx, cam_pts, timestamp_quality, canonical_pts_by_frame
                )
                d_ms = (
                    float(delta_pts_ms_by_frame[f_idx])
                    if delta_pts_ms_by_frame and f_idx in delta_pts_ms_by_frame
                    else (
                        timestamp_quality.mean_delta_pts_front_ms
                        if moto_track.camera_id == "front"
                        else timestamp_quality.mean_delta_pts_back_ms
                    )
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=f_idx,
                        camera_pts_sec=cam_pts,
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in entry["raw_bbox"]),
                        kalman_bbox_xyxy=tuple(
                            float(c) for c in entry.get("kalman_bbox", entry["raw_bbox"])
                        ),
                        confidence=float(entry["confidence"]),
                        velocity_xy=tuple(float(v) for v in entry.get("velocity_xy", (0.0, 0.0))),
                        matched_in_stage2=bool(entry.get("matched_in_stage2", False)),
                        observed_rider_count=int(entry.get("observed_rider_count", 0)),
                        delta_pts_ms=d_ms,
                    )
                )
        else:
            seen_frames = set()
            for f_idx, cam_pts, m_bbox, m_conf in moto_track.trajectory:
                if int(f_idx) in seen_frames:
                    continue
                seen_frames.add(int(f_idx))
                can_pts = self._to_canonical_pts(
                    moto_track.camera_id, int(f_idx), float(cam_pts), timestamp_quality, canonical_pts_by_frame
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=int(f_idx),
                        camera_pts_sec=float(cam_pts),
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in m_bbox),
                        kalman_bbox_xyxy=tuple(float(c) for c in m_bbox),
                        confidence=float(m_conf),
                        velocity_xy=tuple(float(v) for v in moto_track.kalman.velocity_xy),
                    )
                )

        obs_records.sort(key=lambda r: r.frame_index)

        # Build child RiderTracklets first so we can also determine per-frame observed rider counts
        rider_tracklets: dict[int, RiderTracklet] = {}
        rider_obs_by_frame: Counter[int] = Counter()
        for s_idx, sub in sorted(moto_track.rider_subtracks.items()):
            rt = self.build_rider_tracklet(
                subtrack=sub,
                timestamp_quality=timestamp_quality,
                frame_images_by_index=frame_images_by_index,
                canonical_pts_by_frame=canonical_pts_by_frame,
                delta_pts_ms_by_frame=delta_pts_ms_by_frame,
                frame_dimensions=frame_dimensions,
            )
            rider_tracklets[int(s_idx)] = rt
            for r_obs in rt.observed_frames:
                rider_obs_by_frame[r_obs.frame_index] += 1

        # Populate per-frame observed rider counts if not already set
        rider_count_dist: Counter[int] = Counter()
        for rec in obs_records:
            if rec.observed_rider_count == 0 and rider_obs_by_frame[rec.frame_index] > 0:
                rec.observed_rider_count = int(rider_obs_by_frame[rec.frame_index])
            rider_count_dist[int(rec.observed_rider_count)] += 1

        obs_frames = [r.frame_index for r in obs_records]
        occ_intervals = _compute_occlusion_intervals(obs_frames)
        obs_count = len(obs_records)

        if obs_count > 0:
            first_f = obs_records[0].frame_index
            last_f = obs_records[-1].frame_index
            span_frames = max(obs_count, last_f - first_f + 1)
            pred_only_count = max(0, span_frames - obs_count)
            t_start = obs_records[0].canonical_pts_sec
            t_end = obs_records[-1].canonical_pts_sec
            t_centroid = float(np.mean([r.canonical_pts_sec for r in obs_records]))
            cam_start = obs_records[0].camera_pts_sec
            cam_end = obs_records[-1].camera_pts_sec
            entry_box = obs_records[0].bbox_xyxy
            exit_box = obs_records[-1].bbox_xyxy

            peak_rec = max(obs_records, key=lambda r: _bbox_area(r.bbox_xyxy))
            t_peak = peak_rec.canonical_pts_sec
            peak_box = peak_rec.bbox_xyxy

            # Estimate entry/exit/mean velocity from observed trajectory + Kalman velocity
            if obs_count >= 2:
                c0x = 0.5 * (entry_box[0] + entry_box[2])
                c0y = 0.5 * (entry_box[1] + entry_box[3])
                c1x = 0.5 * (exit_box[0] + exit_box[2])
                c1y = 0.5 * (exit_box[1] + exit_box[3])
                dt_f = float(max(1, last_f - first_f))
                empirical_vx = (c1x - c0x) / dt_f
                empirical_vy = (c1y - c0y) / dt_f
                entry_vel = obs_records[0].velocity_xy
                if abs(entry_vel[0]) < 1e-6 and abs(entry_vel[1]) < 1e-6:
                    entry_vel = (empirical_vx, empirical_vy)
                exit_vel = obs_records[-1].velocity_xy
                if abs(exit_vel[0]) < 1e-6 and abs(exit_vel[1]) < 1e-6:
                    exit_vel = (empirical_vx, empirical_vy)
                mean_vel = (empirical_vx, empirical_vy)
            else:
                entry_vel = exit_vel = mean_vel = tuple(float(v) for v in moto_track.kalman.velocity_xy)

            v_norm = float(np.hypot(mean_vel[0], mean_vel[1]))
            if v_norm > 1e-5:
                dir_unit = (float(mean_vel[0] / v_norm), float(mean_vel[1] / v_norm))
            else:
                dir_unit = (0.0, 0.0)

            mean_conf = float(np.mean([r.confidence for r in obs_records]))
        else:
            first_f = last_f = int(moto_track.last_frame_index)
            span_frames = 0
            pred_only_count = int(moto_track.age)
            t_start = t_end = t_peak = t_centroid = cam_start = cam_end = 0.0
            entry_box = exit_box = peak_box = moto_track.current_bbox
            entry_vel = exit_vel = mean_vel = (0.0, 0.0)
            dir_unit = (0.0, 0.0)
            mean_conf = 0.0

        density = float(obs_count / max(1, span_frames))
        dims = self._resolve_frame_dimensions(frame_dimensions)
        entry_border, exit_border, border_status = self._border_flags(
            entry_box, exit_box, [r.bbox_xyxy for r in obs_records], dims
        )

        # Rider count summary
        non_zero_counts = {k: v for k, v in rider_count_dist.items() if k > 0}
        if non_zero_counts:
            # Break ties toward higher observed count (since partial occlusion drops count)
            modal_riders = max(
                non_zero_counts.keys(), key=lambda k: (non_zero_counts[k], k)
            )
            max_riders = max(non_zero_counts.keys())
        elif rider_tracklets:
            modal_riders = len(rider_tracklets)
            max_riders = len(rider_tracklets)
        else:
            modal_riders = 0
            max_riders = 0

        ever_confirmed = bool(
            getattr(moto_track, "_ever_confirmed", moto_track.lifecycle_state != TrackLifecycleState.TENTATIVE)
        )
        completeness_reasons: list[str] = []
        if not ever_confirmed or moto_track.lifecycle_state == TrackLifecycleState.TENTATIVE:
            completeness = TrackletCompleteness.FRAGMENTED_TENTATIVE
            completeness_reasons.append("TRACK_NEVER_CONFIRMED_TENTATIVE")
        elif obs_count < self.config.min_observed_frames_moto:
            completeness = TrackletCompleteness.SHORT_OBSERVATION
            completeness_reasons.append(f"OBSERVED_FRAMES_{obs_count}_BELOW_MIN_{self.config.min_observed_frames_moto}")
        elif density < self.config.min_observation_density:
            completeness = TrackletCompleteness.HEAVILY_OCCLUDED
            completeness_reasons.append(f"OBSERVATION_DENSITY_{density:.2f}_BELOW_{self.config.min_observation_density}")
        else:
            completeness = TrackletCompleteness.COMPLETE
            completeness_reasons.append("SUFFICIENT_OBSERVED_FRAMES_AND_DENSITY")

        if border_status is not None:
            completeness_reasons.append(border_status)
        else:
            if not entry_border and obs_count >= self.config.min_observed_frames_moto:
                completeness_reasons.append("MID_FRAME_INITIATION")
            if not exit_border and moto_track.lifecycle_state == TrackLifecycleState.TERMINATED:
                completeness_reasons.append("MID_FRAME_TERMINATION")

        # Extract chassis appearance and combine with driver/rider cranial & torso histograms
        chassis_desc = extract_tracklet_appearance_descriptor(
            observed_frames=obs_records,
            frame_images_by_index=frame_images_by_index,
            config=self.config.appearance_config,
            is_motorcycle_tracklet=True,
            native_frame_dimensions=dims,
        )
        combined_app = self._combine_motorcycle_and_rider_appearance(
            chassis_desc, rider_tracklets
        )

        return MotorcycleTracklet(
            schema_version=SCHEMA_VERSION,
            camera_id=moto_track.camera_id,
            moto_track_id=moto_track.moto_track_id,
            lifecycle_state=moto_track.lifecycle_state.value,
            ever_confirmed=ever_confirmed,
            t_start_sec=t_start,
            t_end_sec=t_end,
            t_peak_sec=t_peak,
            t_centroid_sec=t_centroid,
            camera_pts_start_sec=cam_start,
            camera_pts_end_sec=cam_end,
            first_frame_index=first_f,
            last_frame_index=last_f,
            observed_frame_count=obs_count,
            predicted_only_frame_count=pred_only_count,
            total_span_frames=span_frames,
            observation_density=density,
            stage2_low_conf_matches=int(moto_track.stage2_low_conf_matches),
            recovery_count=int(moto_track.recovery_count),
            occlusion_intervals=occ_intervals,
            entry_bbox=entry_box,
            exit_bbox=exit_box,
            peak_area_bbox=peak_box,
            entry_velocity_xy=entry_vel,
            exit_velocity_xy=exit_vel,
            mean_velocity_xy=mean_vel,
            motion_direction_unit_xy=dir_unit,
            mean_detection_confidence=mean_conf,
            entry_near_border=entry_border,
            exit_near_border=exit_border,
            rider_count_distribution=dict(rider_count_dist),
            modal_rider_count=int(modal_riders),
            max_rider_count=int(max_riders),
            rider_subtrack_count=len(rider_tracklets),
            slot_spatial_conflicts_detected=int(
                getattr(moto_track, "slot_spatial_conflicts_detected", 0)
            ),
            rider_abstained_updates=int(getattr(moto_track, "rider_abstained_updates", 0)),
            spatial_identity_overrides=int(getattr(moto_track, "spatial_identity_overrides", 0)),
            completeness=completeness,
            completeness_reasons=completeness_reasons,
            timestamp_quality=timestamp_quality,
            rider_tracklets=rider_tracklets,
            observed_frames=obs_records,
            appearance_descriptor=combined_app,
            source_metadata={
                "camera_id": moto_track.camera_id,
                "pair_id": timestamp_quality.pair_id,
                "track_age": int(moto_track.age),
                "track_hits": int(moto_track.hits),
            },
        )

    def _combine_motorcycle_and_rider_appearance(
        self,
        chassis_desc: AppearanceDescriptorRecord,
        rider_tracklets: dict[int, RiderTracklet],
    ) -> AppearanceDescriptorRecord:
        """
        Combine motorcycle chassis histogram with primary rider (driver or slot 0)
        cranial cap and upper torso histograms when available.
        """
        primary_rider: RiderTracklet | None = None
        for s_idx in sorted(rider_tracklets.keys()):
            rt = rider_tracklets[s_idx]
            if rt.appearance_descriptor.available:
                if rt.slot_role == "driver" or primary_rider is None:
                    primary_rider = rt
                    if rt.slot_role == "driver":
                        break

        if not chassis_desc.available and (primary_rider is None or not primary_rider.appearance_descriptor.available):
            return chassis_desc

        qualities: list[float] = []
        valid_crops = 0
        cranial_hist: list[float] = []
        torso_hist: list[float] = []
        chassis_hist: list[float] = []

        if chassis_desc.available:
            chassis_hist = list(chassis_desc.moto_chassis_histogram)
            qualities.append(chassis_desc.mean_crop_quality)
            valid_crops += chassis_desc.valid_observed_crops_count

        if primary_rider is not None and primary_rider.appearance_descriptor.available:
            r_app = primary_rider.appearance_descriptor
            cranial_hist = list(r_app.cranial_cap_histogram)
            torso_hist = list(r_app.upper_torso_histogram)
            qualities.append(r_app.mean_crop_quality)
            valid_crops += r_app.valid_observed_crops_count

        return AppearanceDescriptorRecord(
            available=True,
            unavailable_reason=None,
            valid_observed_crops_count=valid_crops,
            mean_crop_quality=float(np.mean(qualities)) if qualities else 0.0,
            cranial_cap_histogram=cranial_hist,
            upper_torso_histogram=torso_hist,
            moto_chassis_histogram=chassis_hist,
            observed_frames_total=chassis_desc.observed_frames_total,
            observed_frames_with_keyframe=chassis_desc.observed_frames_with_keyframe,
            crop_rejection_counts=dict(chassis_desc.crop_rejection_counts),
        )

    def build_orphan_tracklet(
        self,
        orphan_track: OrphanRiderTrack,
        timestamp_quality: TimestampQualityRecord,
        frame_images_by_index: dict[int, np.ndarray] | None = None,
        canonical_pts_by_frame: dict[int, float] | None = None,
        delta_pts_ms_by_frame: dict[int, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> OrphanRiderTracklet:
        """Build an `OrphanRiderTracklet` from a single-camera `OrphanRiderTrack`."""
        obs_records: list[ObservedFrameRecord] = []
        raw_logs = getattr(orphan_track, "observation_log", None)
        pts_hist = getattr(orphan_track, "pts_history", None)

        if raw_logs:
            seen_frames: set[int] = set()
            for entry in raw_logs:
                f_idx = int(entry["frame_index"])
                if f_idx in seen_frames:
                    continue
                seen_frames.add(f_idx)
                cam_pts = float(entry["pts_timestamp_sec"])
                can_pts = self._to_canonical_pts(
                    orphan_track.camera_id, f_idx, cam_pts, timestamp_quality, canonical_pts_by_frame
                )
                d_ms = (
                    float(delta_pts_ms_by_frame[f_idx])
                    if delta_pts_ms_by_frame and f_idx in delta_pts_ms_by_frame
                    else (
                        timestamp_quality.mean_delta_pts_front_ms
                        if orphan_track.camera_id == "front"
                        else timestamp_quality.mean_delta_pts_back_ms
                    )
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=f_idx,
                        camera_pts_sec=cam_pts,
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in entry["head_bbox"]),
                        kalman_bbox_xyxy=tuple(
                            float(c) for c in entry.get("kalman_bbox", entry["head_bbox"])
                        ),
                        confidence=float(entry["head_conf"]),
                        velocity_xy=tuple(float(v) for v in entry.get("velocity_xy", (0.0, 0.0))),
                        person_bbox_xyxy=(
                            tuple(float(c) for c in entry["person_bbox"])
                            if entry.get("person_bbox") is not None
                            else None
                        ),
                        person_confidence=(
                            float(entry["person_conf"])
                            if entry.get("person_conf") is not None
                            else None
                        ),
                        head_class_name=str(entry.get("head_class_name", "")),
                        delta_pts_ms=d_ms,
                    )
                )
        elif pts_hist:
            seen_frames = set()
            for f_idx, cam_pts, h_bbox, h_cls, h_conf in pts_hist:
                if int(f_idx) in seen_frames:
                    continue
                seen_frames.add(int(f_idx))
                can_pts = self._to_canonical_pts(
                    orphan_track.camera_id, int(f_idx), float(cam_pts), timestamp_quality, canonical_pts_by_frame
                )
                obs_records.append(
                    ObservedFrameRecord(
                        frame_index=int(f_idx),
                        camera_pts_sec=float(cam_pts),
                        canonical_pts_sec=can_pts,
                        bbox_xyxy=tuple(float(c) for c in h_bbox),
                        kalman_bbox_xyxy=tuple(float(c) for c in h_bbox),
                        confidence=float(h_conf),
                        head_class_name=str(h_cls),
                    )
                )

        obs_records.sort(key=lambda r: r.frame_index)
        obs_frames = [r.frame_index for r in obs_records]
        occ_intervals = _compute_occlusion_intervals(obs_frames)
        obs_count = len(obs_records)

        if obs_count > 0:
            first_f = obs_records[0].frame_index
            last_f = obs_records[-1].frame_index
            span_frames = max(obs_count, last_f - first_f + 1)
            t_start = obs_records[0].canonical_pts_sec
            t_end = obs_records[-1].canonical_pts_sec
            t_centroid = float(np.mean([r.canonical_pts_sec for r in obs_records]))
            cam_start = obs_records[0].camera_pts_sec
            cam_end = obs_records[-1].camera_pts_sec
            entry_box = obs_records[0].bbox_xyxy
            exit_box = obs_records[-1].bbox_xyxy
            mean_conf = float(np.mean([r.confidence for r in obs_records]))
            mean_vel = tuple(float(v) for v in orphan_track.kalman.velocity_xy)
        else:
            span_frames = 0
            t_start = t_end = t_centroid = cam_start = cam_end = 0.0
            entry_box = exit_box = orphan_track.kalman.current_bbox_xyxy
            mean_conf = 0.0
            mean_vel = (0.0, 0.0)

        missed_count = max(0, span_frames - obs_count)
        density = float(obs_count / max(1, span_frames))
        if obs_count < self.config.min_observed_frames_orphan:
            completeness = TrackletCompleteness.SHORT_OBSERVATION
        elif density < self.config.min_observation_density:
            completeness = TrackletCompleteness.HEAVILY_OCCLUDED
        else:
            completeness = TrackletCompleteness.COMPLETE

        app_desc = extract_tracklet_appearance_descriptor(
            observed_frames=obs_records,
            frame_images_by_index=frame_images_by_index,
            config=self.config.appearance_config,
            is_motorcycle_tracklet=False,
            native_frame_dimensions=self._resolve_frame_dimensions(frame_dimensions),
        )

        return OrphanRiderTracklet(
            schema_version=SCHEMA_VERSION,
            camera_id=orphan_track.camera_id,
            orphan_track_id=orphan_track.orphan_track_id,
            lifecycle_state=orphan_track.lifecycle_state.value,
            last_rejection_reason=str(orphan_track.last_rejection_reason),
            t_start_sec=t_start,
            t_end_sec=t_end,
            t_centroid_sec=t_centroid,
            camera_pts_start_sec=cam_start,
            camera_pts_end_sec=cam_end,
            observed_frame_count=obs_count,
            total_span_frames=span_frames,
            missed_frame_count=missed_count,
            observation_density=density,
            occlusion_intervals=occ_intervals,
            entry_head_bbox=entry_box,
            exit_head_bbox=exit_box,
            mean_velocity_xy=mean_vel,
            mean_head_confidence=mean_conf,
            phase10_temporal_state=orphan_track.phase10_track.state.value,
            phase10_aggregated_confidence=dict(
                orphan_track.phase10_track.aggregator.get_aggregated_confidence()
            ),
            phase10_alerts_count=len(orphan_track.alerts),
            completeness=completeness,
            timestamp_quality=timestamp_quality,
            observed_frames=obs_records,
            appearance_descriptor=app_desc,
            genuine_helmet_evidence=self._export_helmet_evidence(
                orphan_track, orphan_track.phase10_track.aggregator.get_aggregated_confidence()
            ),
        )

    def extract_from_pipeline(
        self,
        pipeline: CameraTrackingPipeline,
        timestamp_quality: TimestampQualityRecord,
        frame_images_by_index: dict[int, np.ndarray] | None = None,
        canonical_pts_by_frame: dict[int, float] | None = None,
        delta_pts_ms_by_frame: dict[int, float] | None = None,
        frame_dimensions: tuple[int, int] | None = None,
    ) -> tuple[list[MotorcycleTracklet], list[OrphanRiderTracklet]]:
        """
        Extract deduplicated `MotorcycleTracklet` and `OrphanRiderTracklet` lists
        from a single-camera `CameraTrackingPipeline`.
        A recovered `MotoTrack` remains a single `MotorcycleTracklet` with
        `recovery_count >= 1` and `occlusion_intervals` rather than creating duplicates.
        """
        seen_moto_ids: set[str] = set()
        all_motos: list[MotoTrack] = []
        for m in list(pipeline.active_moto_tracks.values()) + list(pipeline.terminated_moto_tracks):
            if m.moto_track_id in seen_moto_ids:
                continue
            seen_moto_ids.add(m.moto_track_id)
            if (
                self.config.exclude_tentative_tracks
                and m.lifecycle_state == TrackLifecycleState.TENTATIVE
                and not getattr(m, "_ever_confirmed", False)
            ):
                continue
            all_motos.append(m)

        moto_tracklets = [
            self.build_motorcycle_tracklet(
                moto_track=m,
                timestamp_quality=timestamp_quality,
                frame_images_by_index=frame_images_by_index,
                canonical_pts_by_frame=canonical_pts_by_frame,
                delta_pts_ms_by_frame=delta_pts_ms_by_frame,
                frame_dimensions=frame_dimensions,
            )
            for m in all_motos
        ]
        moto_tracklets.sort(key=lambda t: (t.t_start_sec, t.moto_track_id))

        seen_orphan_ids: set[str] = set()
        all_orphans: list[OrphanRiderTrack] = []
        for o in list(pipeline.active_orphan_tracks.values()) + list(pipeline.terminated_orphan_tracks):
            if o.orphan_track_id in seen_orphan_ids:
                continue
            seen_orphan_ids.add(o.orphan_track_id)
            all_orphans.append(o)

        orphan_tracklets = [
            self.build_orphan_tracklet(
                orphan_track=o,
                timestamp_quality=timestamp_quality,
                frame_images_by_index=frame_images_by_index,
                canonical_pts_by_frame=canonical_pts_by_frame,
                delta_pts_ms_by_frame=delta_pts_ms_by_frame,
                frame_dimensions=frame_dimensions,
            )
            for o in all_orphans
        ]
        orphan_tracklets.sort(key=lambda t: (t.t_start_sec, t.orphan_track_id))

        return moto_tracklets, orphan_tracklets
