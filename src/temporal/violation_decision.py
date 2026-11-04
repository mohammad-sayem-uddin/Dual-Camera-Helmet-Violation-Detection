"""Violation Decision and Alert Generation Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any

from src.temporal.rider_tracker import Track
from src.temporal.state_machine import TemporalState


@dataclass
class ViolationAlert:
    """Authoritative roadside helmet violation citation event."""
    alert_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    bbox: list[float]
    violation_confidence: float
    observation_count: int
    evidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 2),
            "bbox": [round(c, 1) for c in self.bbox],
            "violation_confidence": round(self.violation_confidence, 4),
            "observation_count": self.observation_count,
            "evidence_summary": self.evidence_summary,
        }


@dataclass
class DecisionConfig:
    """Configuration for citation alert thresholding and suppression."""
    violation_alert_threshold: float = 0.65
    min_confirmation_frames: int = 5
    suppress_duplicates_per_track: bool = True
    cooldown_frames: int = 90  # 3 seconds at 30 fps


class ViolationDecisionEngine:
    """
    Decides when to issue authoritative violation citation alerts.
    
    Guarantees:
    1. Only tracks in NO_HELMET_CONFIRMED with aggregated confidence >= violation_alert_threshold
       can trigger an alert.
    2. Zero duplicates: tracks with alert_latched = True are suppressed unless cooldown expires.
    3. Every alert records complete trajectory and evidentiary metrics for judicial review.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self.alerts_log: list[ViolationAlert] = []
        self.last_alert_frame: dict[int, int] = {}  # track_id -> frame_index

    def process_tracks(self, frame_index: int, tracks: list[Track], fps: float = 30.0) -> list[ViolationAlert]:
        """
        Evaluate active tracks in the current frame and emit new alerts where verified.
        
        Args:
            frame_index: Current sequential frame index.
            tracks: List of active Track objects.
            fps: Frame rate for computing real-time timestamps.
            
        Returns:
            List of newly emitted ViolationAlert events in this frame.
        """
        new_alerts: list[ViolationAlert] = []

        for track in tracks:
            # Must be in confirmed violation state
            if track.state != TemporalState.NO_HELMET_CONFIRMED:
                continue

            conf_dict = track.aggregator.get_aggregated_confidence()
            p_no_helmet = conf_dict.get("no_helmet", 0.0)

            # Must satisfy violation confidence threshold
            if p_no_helmet < self.config.violation_alert_threshold:
                continue

            # Must satisfy minimum confirmation length
            if track.state_machine.consecutive_no_helmet_frames < self.config.min_confirmation_frames:
                continue

            # Check duplicate alert suppression
            if self.config.suppress_duplicates_per_track and track.state_machine.alert_latched:
                # Check cooldown
                last_f = self.last_alert_frame.get(track.track_id, -999999)
                if frame_index - last_f < self.config.cooldown_frames:
                    continue

            # Authoritative Alert Generation
            alert = ViolationAlert(
                alert_id=f"VIOLATION-{track.track_id}-{frame_index}-{uuid.uuid4().hex[:6].upper()}",
                track_id=track.track_id,
                frame_index=frame_index,
                timestamp_seconds=float(frame_index) / float(fps) if fps > 0 else 0.0,
                bbox=[
                    track.current_bbox.x1,
                    track.current_bbox.y1,
                    track.current_bbox.x2,
                    track.current_bbox.y2,
                ],
                violation_confidence=p_no_helmet,
                observation_count=track.hits,
                evidence_summary={
                    "consecutive_violation_frames": track.state_machine.consecutive_no_helmet_frames,
                    "aggregated_confidence": conf_dict,
                    "state_history": [s.value for s in track.state_machine.state_history[-10:]],
                    "raw_detection_confidence": round(track.current_bbox.confidence, 4),
                },
            )

            track.state_machine.alert_latched = True
            self.last_alert_frame[track.track_id] = frame_index
            self.alerts_log.append(alert)
            new_alerts.append(alert)

        return new_alerts

    def reset(self) -> None:
        """Clear alert history and cooldown trackers."""
        self.alerts_log.clear()
        self.last_alert_frame.clear()
