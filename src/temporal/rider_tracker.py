"""Rider Bounding-Box Association and Trajectory Tracker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.temporal.confidence_aggregator import BaseConfidenceAggregator, create_aggregator
from src.temporal.state_machine import StateMachineConfig, TemporalState, TemporalStateMachine


@dataclass
class BoundingBox:
    """Bounding box detection representation in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 1e-8:
            return 0.0
        return inter_area / union_area


class Track:
    """Represents a continuous identity trajectory for a single motorcycle rider."""

    def __init__(
        self,
        track_id: int,
        initial_bbox: BoundingBox,
        frame_index: int,
        aggregator_method: str = "confidence_weighted",
        window_size: int = 10,
        sm_config: StateMachineConfig | None = None,
    ) -> None:
        self.track_id = track_id
        self.current_bbox = initial_bbox
        self.trajectory: list[tuple[int, BoundingBox]] = [(frame_index, initial_bbox)]
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.aggregator = create_aggregator(method=aggregator_method, window_size=window_size)
        self.aggregator.add_observation(
            class_name=initial_bbox.class_name,
            confidence=initial_bbox.confidence,
            is_detected=True,
        )

        self.state_machine = TemporalStateMachine(track_id=track_id, config=sm_config)
        self.state_machine.update(
            frame_index=frame_index,
            aggregated_conf=self.aggregator.get_aggregated_confidence(),
            is_detected=True,
        )

    def predict_next_bbox(self) -> BoundingBox:
        """Estimate next bounding box location using linear velocity projection."""
        if len(self.trajectory) < 2:
            return self.current_bbox

        (f_prev, b_prev) = self.trajectory[-2]
        (f_curr, b_curr) = self.trajectory[-1]
        dt = max(1, f_curr - f_prev)

        dx = (b_curr.center[0] - b_prev.center[0]) / dt
        dy = (b_curr.center[1] - b_prev.center[1]) / dt

        new_x1 = self.current_bbox.x1 + dx
        new_y1 = self.current_bbox.y1 + dy
        new_x2 = self.current_bbox.x2 + dx
        new_y2 = self.current_bbox.y2 + dy

        return BoundingBox(
            x1=new_x1,
            y1=new_y1,
            x2=new_x2,
            y2=new_y2,
            confidence=self.current_bbox.confidence,
            class_id=self.current_bbox.class_id,
            class_name=self.current_bbox.class_name,
        )

    def update_matched(self, bbox: BoundingBox, frame_index: int) -> None:
        """Update track with newly associated bounding box detection."""
        self.current_bbox = bbox
        self.trajectory.append((frame_index, bbox))
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        self.aggregator.add_observation(
            class_name=bbox.class_name,
            confidence=bbox.confidence,
            is_detected=True,
        )
        self.state_machine.update(
            frame_index=frame_index,
            aggregated_conf=self.aggregator.get_aggregated_confidence(),
            is_detected=True,
        )

    def update_unmatched(self, frame_index: int) -> None:
        """Update track when no detection was matched in this frame."""
        self.age += 1
        self.time_since_update += 1

        self.aggregator.add_observation(
            class_name="unknown",
            confidence=0.0,
            is_detected=False,
        )
        self.state_machine.update(
            frame_index=frame_index,
            aggregated_conf=self.aggregator.get_aggregated_confidence(),
            is_detected=False,
        )

    @property
    def state(self) -> TemporalState:
        return self.state_machine.current_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "bbox": [
                round(self.current_bbox.x1, 1),
                round(self.current_bbox.y1, 1),
                round(self.current_bbox.x2, 1),
                round(self.current_bbox.y2, 1),
            ],
            "confidence": round(self.current_bbox.confidence, 4),
            "class_name": self.current_bbox.class_name,
            "aggregated_confidence": self.aggregator.get_aggregated_confidence(),
        }


class RiderTracker:
    """
    Associates frame detections to ongoing rider tracks via spatial IoU matching.
    Manages track creation, updating, occlusion buffering, and expiration.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 2,
        aggregator_method: str = "confidence_weighted",
        window_size: int = 10,
        sm_config: StateMachineConfig | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.aggregator_method = aggregator_method
        self.window_size = window_size
        self.sm_config = sm_config or StateMachineConfig()

        self.next_track_id = 1
        self.active_tracks: dict[int, Track] = {}
        self.dead_tracks: list[Track] = []

    def update(self, frame_index: int, detections: list[BoundingBox]) -> list[Track]:
        """
        Ingest current frame detections and advance all active tracks.
        
        Args:
            frame_index: Sequential integer frame index.
            detections: List of BoundingBox objects for the current frame.
            
        Returns:
            List of currently active Track objects.
        """
        # If no active tracks, initiate all detections
        if not self.active_tracks:
            for det in detections:
                t = Track(
                    track_id=self.next_track_id,
                    initial_bbox=det,
                    frame_index=frame_index,
                    aggregator_method=self.aggregator_method,
                    window_size=self.window_size,
                    sm_config=self.sm_config,
                )
                self.active_tracks[self.next_track_id] = t
                self.next_track_id += 1
            return list(self.active_tracks.values())

        track_ids = list(self.active_tracks.keys())
        predictions = [self.active_tracks[tid].predict_next_bbox() for tid in track_ids]

        # Compute IoU cost matrix
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        if predictions and detections:
            # Build list of potential matches sorted by IoU descending
            candidates = []
            for t_idx, pred_box in enumerate(predictions):
                for d_idx, det_box in enumerate(detections):
                    iou_val = pred_box.iou(det_box)
                    if iou_val >= self.iou_threshold:
                        candidates.append((iou_val, t_idx, d_idx))

            candidates.sort(key=lambda x: x[0], reverse=True)

            for iou_val, t_idx, d_idx in candidates:
                tid = track_ids[t_idx]
                if tid not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(tid)
                    matched_detections.add(d_idx)
                    matches.append((t_idx, d_idx))

        # Update matched tracks
        for t_idx, d_idx in matches:
            tid = track_ids[t_idx]
            self.active_tracks[tid].update_matched(detections[d_idx], frame_index)

        # Update unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.active_tracks[tid].update_unmatched(frame_index)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(
                    track_id=self.next_track_id,
                    initial_bbox=det,
                    frame_index=frame_index,
                    aggregator_method=self.aggregator_method,
                    window_size=self.window_size,
                    sm_config=self.sm_config,
                )
                self.active_tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Purge dead tracks that exceed max_age or reached TRACK_LOST
        surviving_tracks: dict[int, Track] = {}
        for tid, track in self.active_tracks.items():
            if track.time_since_update > self.max_age or track.state == TemporalState.TRACK_LOST:
                self.dead_tracks.append(track)
            else:
                surviving_tracks[tid] = track

        self.active_tracks = surviving_tracks
        return list(self.active_tracks.values())

    def reset(self) -> None:
        """Reset tracker state and clear all tracks."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.dead_tracks.clear()
