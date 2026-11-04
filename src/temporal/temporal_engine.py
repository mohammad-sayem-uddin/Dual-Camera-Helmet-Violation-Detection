"""Central Temporal Verification Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.temporal.rider_tracker import BoundingBox, RiderTracker, Track
from src.temporal.state_machine import StateMachineConfig, TemporalState
from src.temporal.violation_decision import DecisionConfig, ViolationAlert, ViolationDecisionEngine


@dataclass
class TemporalEngineConfig:
    """Full parameter configuration for the Temporal Verification Engine."""
    detection_confidence_threshold: float = 0.25
    iou_match_threshold: float = 0.30
    max_track_age_frames: int = 15
    aggregator_method: str = "confidence_weighted"
    sliding_window_size: int = 10
    ema_alpha: float = 0.35
    state_machine_config: StateMachineConfig = field(default_factory=StateMachineConfig)
    decision_config: DecisionConfig = field(default_factory=DecisionConfig)


class TemporalVerificationEngine:
    """
    Downstream temporal verification pipeline. Ingests raw object detections,
    tracks identities across time, aggregates evidence, and confirms violation citations.
    """

    def __init__(self, config: TemporalEngineConfig | None = None) -> None:
        self.config = config or TemporalEngineConfig()
        self.tracker = RiderTracker(
            iou_threshold=self.config.iou_match_threshold,
            max_age=self.config.max_track_age_frames,
            aggregator_method=self.config.aggregator_method,
            window_size=self.config.sliding_window_size,
            sm_config=self.config.state_machine_config,
        )
        self.decision_engine = ViolationDecisionEngine(config=self.config.decision_config)
        self.total_frames_processed = 0

    def process_frame(
        self,
        frame_index: int,
        raw_detections: list[dict[str, Any]],
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """
        Ingest a video frame's raw YOLO detections and advance temporal state.
        
        Args:
            frame_index: Sequential integer frame index.
            raw_detections: List of dicts, each with keys:
                {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int, "class_name": str}
            fps: Frame rate for timestamp calculations.
            
        Returns:
            Dict containing active tracks, new alerts, and frame metadata.
        """
        self.total_frames_processed += 1

        # 1. Filter raw detections below detection threshold
        filtered_boxes: list[BoundingBox] = []
        for det in raw_detections:
            conf = float(det.get("confidence", 0.0))
            if conf >= self.config.detection_confidence_threshold:
                bbox_coords = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
                cls_id = int(det.get("class_id", 0))
                cls_name = str(det.get("class_name", "no_helmet" if cls_id == 0 else "helmet"))
                filtered_boxes.append(
                    BoundingBox(
                        x1=float(bbox_coords[0]),
                        y1=float(bbox_coords[1]),
                        x2=float(bbox_coords[2]),
                        y2=float(bbox_coords[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        # 2. Advance tracker & update per-track temporal state machines
        active_tracks = self.tracker.update(frame_index=frame_index, detections=filtered_boxes)

        # 3. Evaluate violation decision engine for new citations
        new_alerts = self.decision_engine.process_tracks(frame_index=frame_index, tracks=active_tracks, fps=fps)

        # 4. Compile frame summary
        return {
            "frame_index": frame_index,
            "timestamp_seconds": round(float(frame_index) / float(fps) if fps > 0 else 0.0, 2),
            "raw_detections_count": len(raw_detections),
            "filtered_detections_count": len(filtered_boxes),
            "active_tracks_count": len(active_tracks),
            "new_alerts_count": len(new_alerts),
            "active_tracks": [t.to_dict() for t in active_tracks],
            "new_alerts": [a.to_dict() for a in new_alerts],
        }

    def get_summary(self) -> dict[str, Any]:
        """Produce an overall system summary of processed tracks and alerts."""
        all_tracks = list(self.tracker.active_tracks.values()) + self.tracker.dead_tracks

        state_counts: dict[str, int] = {s.value: 0 for s in TemporalState}
        for t in all_tracks:
            state_counts[t.state.value] += 1

        return {
            "total_frames_processed": self.total_frames_processed,
            "total_tracks_created": len(all_tracks),
            "active_tracks": len(self.tracker.active_tracks),
            "dead_tracks": len(self.tracker.dead_tracks),
            "total_alerts_emitted": len(self.decision_engine.alerts_log),
            "state_distribution": state_counts,
            "configuration": {
                "detection_conf_threshold": self.config.detection_confidence_threshold,
                "iou_match_threshold": self.config.iou_match_threshold,
                "aggregator_method": self.config.aggregator_method,
                "sliding_window_size": self.config.sliding_window_size,
                "violation_alert_threshold": self.config.decision_config.violation_alert_threshold,
                "min_confirmation_frames": self.config.decision_config.min_confirmation_frames,
            },
        }

    def reset(self) -> None:
        """Reset internal state, tracker, and alerts."""
        self.tracker.reset()
        self.decision_engine.reset()
        self.total_frames_processed = 0
