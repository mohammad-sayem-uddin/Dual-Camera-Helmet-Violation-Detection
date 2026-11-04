"""Six-State Temporal Finite State Machine with Hysteresis for Helmet Violation Verification."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TemporalState(str, Enum):
    """Temporal verification states for tracked rider entities."""
    UNKNOWN = "UNKNOWN"
    HELMET_CONFIRMED = "HELMET_CONFIRMED"
    NO_HELMET_SUSPECTED = "NO_HELMET_SUSPECTED"
    NO_HELMET_CONFIRMED = "NO_HELMET_CONFIRMED"
    TEMPORARILY_OCCLUDED = "TEMPORARILY_OCCLUDED"
    TRACK_LOST = "TRACK_LOST"


@dataclass
class StateMachineConfig:
    """Configurable thresholds and temporal criteria for the state machine."""
    min_observation_frames: int = 3
    confirm_frames: int = 5
    max_occlusion_frames: int = 5
    track_lost_frames: int = 15
    suspect_threshold: float = 0.50
    confirm_threshold: float = 0.70
    clear_threshold: float = 0.40
    helmet_threshold: float = 0.65


@dataclass
class StateTransition:
    """Record of an individual state transition event."""
    frame_index: int
    previous_state: TemporalState
    new_state: TemporalState
    reason: str
    aggregated_confidence: dict[str, float]


class TemporalStateMachine:
    """
    Maintains and evolves the temporal verification state for a single tracked rider.
    
    Invariants:
    1. A track starts in UNKNOWN.
    2. A transition to NO_HELMET_CONFIRMED requires sustained evidence across confirm_frames
       with aggregated violation confidence >= confirm_threshold.
    3. Hysteresis prevents instant de-escalation: clearing a confirmed violation requires
       violation confidence < clear_threshold or helmet confidence >= helmet_threshold.
    4. Missing detections transition to TEMPORARILY_OCCLUDED before escalating to TRACK_LOST.
    """

    def __init__(self, track_id: int, config: StateMachineConfig | None = None) -> None:
        self.track_id = track_id
        self.config = config or StateMachineConfig()
        self.current_state = TemporalState.UNKNOWN
        self.frames_in_current_state = 0
        self.total_observed_frames = 0
        self.consecutive_no_helmet_frames = 0
        self.consecutive_helmet_frames = 0
        self.consecutive_occluded_frames = 0
        self.state_history: list[TemporalState] = [self.current_state]
        self.transition_log: list[StateTransition] = []
        self.alert_latched = False

    def update(
        self,
        frame_index: int,
        aggregated_conf: dict[str, float],
        is_detected: bool,
    ) -> tuple[TemporalState, StateTransition | None]:
        """
        Advance the state machine by one frame.
        
        Args:
            frame_index: Current sequential video frame index.
            aggregated_conf: Mapping containing {"no_helmet": float, "helmet": float}.
            is_detected: Whether a bounding box was detected in this frame for the track.
            
        Returns:
            Tuple of (new_state, transition_event_if_changed).
        """
        p_no_helmet = aggregated_conf.get("no_helmet", 0.0)
        p_helmet = aggregated_conf.get("helmet", 0.0)

        prev_state = self.current_state
        next_state = prev_state
        reason = "No change"

        if is_detected:
            self.total_observed_frames += 1
            self.consecutive_occluded_frames = 0

            if p_no_helmet > p_helmet:
                self.consecutive_no_helmet_frames += 1
                self.consecutive_helmet_frames = 0
            else:
                self.consecutive_helmet_frames += 1
                self.consecutive_no_helmet_frames = 0

            # State transition evaluation
            if prev_state == TemporalState.UNKNOWN:
                if self.total_observed_frames >= self.config.min_observation_frames:
                    if p_helmet >= self.config.helmet_threshold:
                        next_state = TemporalState.HELMET_CONFIRMED
                        reason = f"Helmet confidence {p_helmet:.3f} >= {self.config.helmet_threshold}"
                    elif p_no_helmet >= self.config.suspect_threshold:
                        next_state = TemporalState.NO_HELMET_SUSPECTED
                        reason = f"No-helmet confidence {p_no_helmet:.3f} >= {self.config.suspect_threshold}"

            elif prev_state == TemporalState.NO_HELMET_SUSPECTED:
                if p_helmet >= self.config.helmet_threshold:
                    next_state = TemporalState.HELMET_CONFIRMED
                    reason = f"Compliant helmet detected ({p_helmet:.3f})"
                elif (
                    self.consecutive_no_helmet_frames >= self.config.confirm_frames
                    and p_no_helmet >= self.config.confirm_threshold
                ):
                    next_state = TemporalState.NO_HELMET_CONFIRMED
                    reason = (
                        f"Violation confirmed: {self.consecutive_no_helmet_frames} frames "
                        f"with conf {p_no_helmet:.3f} >= {self.config.confirm_threshold}"
                    )
                elif p_no_helmet < self.config.clear_threshold and p_helmet < self.config.clear_threshold:
                    next_state = TemporalState.UNKNOWN
                    reason = "Evidence dropped below suspicion threshold"

            elif prev_state == TemporalState.NO_HELMET_CONFIRMED:
                # Hysteresis: strict criteria to revoke violation
                if (
                    p_helmet >= self.config.helmet_threshold
                    and self.consecutive_helmet_frames >= 3
                ):
                    next_state = TemporalState.HELMET_CONFIRMED
                    reason = f"Violation revoked: consistent helmet observed ({p_helmet:.3f})"
                elif p_no_helmet < self.config.clear_threshold:
                    next_state = TemporalState.UNKNOWN
                    reason = f"Violation de-escalated: conf {p_no_helmet:.3f} < {self.config.clear_threshold}"

            elif prev_state == TemporalState.HELMET_CONFIRMED:
                if (
                    p_no_helmet >= self.config.suspect_threshold
                    and self.consecutive_no_helmet_frames >= 2
                ):
                    next_state = TemporalState.NO_HELMET_SUSPECTED
                    reason = f"Potential violation suspected on compliant track ({p_no_helmet:.3f})"

            elif prev_state in (TemporalState.TEMPORARILY_OCCLUDED, TemporalState.TRACK_LOST):
                # Track re-acquisition
                if p_helmet >= self.config.helmet_threshold:
                    next_state = TemporalState.HELMET_CONFIRMED
                    reason = f"Track re-acquired with helmet ({p_helmet:.3f})"
                elif (
                    self.consecutive_no_helmet_frames >= self.config.confirm_frames
                    and p_no_helmet >= self.config.confirm_threshold
                ):
                    next_state = TemporalState.NO_HELMET_CONFIRMED
                    reason = f"Track re-acquired with confirmed violation ({p_no_helmet:.3f})"
                elif p_no_helmet >= self.config.suspect_threshold:
                    next_state = TemporalState.NO_HELMET_SUSPECTED
                    reason = f"Track re-acquired, violation suspected ({p_no_helmet:.3f})"
                else:
                    next_state = TemporalState.UNKNOWN
                    reason = "Track re-acquired with uncertain evidence"

        else:
            # No detection in this frame: occlusion / loss progression
            self.consecutive_occluded_frames += 1

            if prev_state in (
                TemporalState.UNKNOWN,
                TemporalState.HELMET_CONFIRMED,
                TemporalState.NO_HELMET_SUSPECTED,
                TemporalState.NO_HELMET_CONFIRMED,
            ):
                if self.consecutive_occluded_frames <= self.config.max_occlusion_frames:
                    next_state = TemporalState.TEMPORARILY_OCCLUDED
                    reason = f"Missing detection {self.consecutive_occluded_frames}/{self.config.max_occlusion_frames}"
                else:
                    next_state = TemporalState.TRACK_LOST
                    reason = f"Occlusion exceeded {self.config.max_occlusion_frames} frames"

            elif prev_state == TemporalState.TEMPORARILY_OCCLUDED:
                if self.consecutive_occluded_frames > self.config.track_lost_frames:
                    next_state = TemporalState.TRACK_LOST
                    reason = f"Track lost after {self.consecutive_occluded_frames} frames without detection"

        # Apply state evolution
        transition: StateTransition | None = None
        if next_state != prev_state:
            transition = StateTransition(
                frame_index=frame_index,
                previous_state=prev_state,
                new_state=next_state,
                reason=reason,
                aggregated_confidence=dict(aggregated_conf),
            )
            self.transition_log.append(transition)
            self.current_state = next_state
            self.frames_in_current_state = 1
        else:
            self.frames_in_current_state += 1

        self.state_history.append(self.current_state)
        return self.current_state, transition

    def is_active(self) -> bool:
        """Return True if the track is not permanently lost."""
        return self.current_state != TemporalState.TRACK_LOST

    def to_dict(self) -> dict[str, Any]:
        """Serialize state machine status."""
        return {
            "track_id": self.track_id,
            "current_state": self.current_state.value,
            "frames_in_state": self.frames_in_current_state,
            "total_observed_frames": self.total_observed_frames,
            "consecutive_no_helmet": self.consecutive_no_helmet_frames,
            "consecutive_helmet": self.consecutive_helmet_frames,
            "consecutive_occluded": self.consecutive_occluded_frames,
            "alert_latched": self.alert_latched,
            "transition_count": len(self.transition_log),
        }
