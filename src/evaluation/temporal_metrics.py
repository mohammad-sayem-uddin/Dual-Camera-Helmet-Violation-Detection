"""Quantitative Metrics for Evaluating Temporal Verification Stability and Accuracy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.temporal.rider_tracker import Track
from src.temporal.state_machine import TemporalState


@dataclass
class TemporalMetrics:
    """Summary metrics evaluating temporal verification performance."""
    total_tracks: int
    total_alerts: int
    false_alerts: int
    true_alerts: int
    alert_precision: float
    alert_recall: float
    average_confirmation_delay_frames: float
    average_confirmation_delay_seconds: float
    state_stability_score: float  # 1.0 = perfectly stable; 0.0 = extreme flicker
    abstention_rate: float        # Fraction of observations in UNKNOWN state
    occlusion_recovery_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tracks": self.total_tracks,
            "total_alerts": self.total_alerts,
            "true_alerts": self.true_alerts,
            "false_alerts": self.false_alerts,
            "alert_precision": round(self.alert_precision, 4),
            "alert_recall": round(self.alert_recall, 4),
            "avg_confirmation_delay_frames": round(self.average_confirmation_delay_frames, 2),
            "avg_confirmation_delay_seconds": round(self.average_confirmation_delay_seconds, 3),
            "state_stability_score": round(self.state_stability_score, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "occlusion_recovery_rate": round(self.occlusion_recovery_rate, 4),
        }


def compute_temporal_metrics(
    evaluated_tracks: list[Track],
    ground_truth_labels: dict[int, str],  # track_id -> "helmet" or "no_helmet"
    fps: float = 30.0,
) -> TemporalMetrics:
    """
    Computes rigorous temporal metrics comparing track histories against ground-truth classes.
    
    Args:
        evaluated_tracks: List of completed or active Track objects.
        ground_truth_labels: Dictionary mapping track_id to true class.
        fps: Frame rate for computing real-time latency.
        
    Returns:
        TemporalMetrics object.
    """
    if not evaluated_tracks:
        return TemporalMetrics(
            total_tracks=0,
            total_alerts=0,
            false_alerts=0,
            true_alerts=0,
            alert_precision=1.0,
            alert_recall=0.0,
            average_confirmation_delay_frames=0.0,
            average_confirmation_delay_seconds=0.0,
            state_stability_score=1.0,
            abstention_rate=0.0,
            occlusion_recovery_rate=1.0,
        )

    true_alerts = 0
    false_alerts = 0
    gt_violations_count = sum(1 for v in ground_truth_labels.values() if v == "no_helmet")

    confirmation_delays: list[int] = []
    flicker_count = 0
    total_state_observations = 0
    unknown_observations = 0

    occlusion_events = 0
    occlusion_recoveries = 0

    for track in evaluated_tracks:
        tid = track.track_id
        gt_class = ground_truth_labels.get(tid, "unknown")
        history = track.state_machine.state_history
        transitions = track.state_machine.transition_log

        # 1. Alert validity
        has_alert = track.state_machine.alert_latched
        if has_alert:
            if gt_class == "no_helmet":
                true_alerts += 1
            else:
                false_alerts += 1

        # 2. Confirmation delay
        if gt_class == "no_helmet":
            for f_idx, state in enumerate(history):
                if state == TemporalState.NO_HELMET_CONFIRMED:
                    confirmation_delays.append(f_idx)
                    break

        # 3. State stability & flicker
        # Flicker: state A -> state B -> state A within short window
        for i in range(len(history) - 2):
            if history[i] != history[i + 1] and history[i] == history[i + 2]:
                flicker_count += 1

        total_state_observations += len(history)
        unknown_observations += sum(1 for s in history if s == TemporalState.UNKNOWN)

        # 4. Occlusion recovery
        for i, trans in enumerate(transitions[:-1]):
            if trans.new_state == TemporalState.TEMPORARILY_OCCLUDED:
                occlusion_events += 1
                next_trans = transitions[i + 1]
                if next_trans.new_state in (TemporalState.HELMET_CONFIRMED, TemporalState.NO_HELMET_CONFIRMED):
                    occlusion_recoveries += 1

    total_alerts = true_alerts + false_alerts
    precision = float(true_alerts) / float(total_alerts) if total_alerts > 0 else 1.0
    recall = float(true_alerts) / float(gt_violations_count) if gt_violations_count > 0 else 1.0

    avg_delay_frames = float(sum(confirmation_delays)) / float(len(confirmation_delays)) if confirmation_delays else 0.0
    avg_delay_sec = avg_delay_frames / float(fps) if fps > 0 else 0.0

    stability_score = max(0.0, 1.0 - (float(flicker_count) / float(max(1, total_state_observations))))
    abstention_rate = float(unknown_observations) / float(max(1, total_state_observations))
    recovery_rate = float(occlusion_recoveries) / float(max(1, occlusion_events)) if occlusion_events > 0 else 1.0

    return TemporalMetrics(
        total_tracks=len(evaluated_tracks),
        total_alerts=total_alerts,
        false_alerts=false_alerts,
        true_alerts=true_alerts,
        alert_precision=precision,
        alert_recall=recall,
        average_confirmation_delay_frames=avg_delay_frames,
        average_confirmation_delay_seconds=avg_delay_sec,
        state_stability_score=stability_score,
        abstention_rate=abstention_rate,
        occlusion_recovery_rate=recovery_rate,
    )
