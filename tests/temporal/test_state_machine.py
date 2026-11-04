"""Unit Tests for Six-State Temporal Finite State Machine."""
import unittest
from src.temporal.state_machine import StateMachineConfig, TemporalState, TemporalStateMachine


class TestTemporalStateMachine(unittest.TestCase):
    def setUp(self):
        self.config = StateMachineConfig(
            min_observation_frames=3,
            confirm_frames=5,
            max_occlusion_frames=4,
            track_lost_frames=10,
            suspect_threshold=0.50,
            confirm_threshold=0.70,
            clear_threshold=0.40,
            helmet_threshold=0.65,
        )
        self.sm = TemporalStateMachine(track_id=1, config=self.config)

    def test_initial_state_is_unknown(self):
        self.assertEqual(self.sm.current_state, TemporalState.UNKNOWN)
        self.assertEqual(self.sm.total_observed_frames, 0)
        self.assertFalse(self.sm.alert_latched)

    def test_transition_to_helmet_confirmed(self):
        # Feed 3 compliant helmet frames
        for f in range(3):
            state, _ = self.sm.update(
                frame_index=f,
                aggregated_conf={"helmet": 0.85, "no_helmet": 0.15},
                is_detected=True,
            )
        self.assertEqual(state, TemporalState.HELMET_CONFIRMED)
        self.assertEqual(self.sm.total_observed_frames, 3)

    def test_transition_to_no_helmet_suspected_and_confirmed(self):
        # First 3 frames: suspect violation
        for f in range(3):
            state, _ = self.sm.update(
                frame_index=f,
                aggregated_conf={"helmet": 0.20, "no_helmet": 0.80},
                is_detected=True,
            )
        self.assertEqual(state, TemporalState.NO_HELMET_SUSPECTED)

        # Feed 2 more violation frames (reaching 5 consecutive confirm_frames)
        for f in range(3, 5):
            state, trans = self.sm.update(
                frame_index=f,
                aggregated_conf={"helmet": 0.10, "no_helmet": 0.90},
                is_detected=True,
            )
        self.assertEqual(state, TemporalState.NO_HELMET_CONFIRMED)
        self.assertIsNotNone(trans)

    def test_hysteresis_prevents_premature_deescalation(self):
        # Reach NO_HELMET_CONFIRMED
        for f in range(5):
            self.sm.update(
                frame_index=f,
                aggregated_conf={"helmet": 0.10, "no_helmet": 0.90},
                is_detected=True,
            )
        self.assertEqual(self.sm.current_state, TemporalState.NO_HELMET_CONFIRMED)

        # Single frame of ambiguous/lower confidence (0.50) must NOT clear confirmed violation
        state, _ = self.sm.update(
            frame_index=5,
            aggregated_conf={"helmet": 0.50, "no_helmet": 0.50},
            is_detected=True,
        )
        self.assertEqual(state, TemporalState.NO_HELMET_CONFIRMED)

        # Only when evidence drops below clear_threshold (0.40) does it de-escalate
        state, trans = self.sm.update(
            frame_index=6,
            aggregated_conf={"helmet": 0.30, "no_helmet": 0.35},
            is_detected=True,
        )
        self.assertEqual(state, TemporalState.UNKNOWN)

    def test_occlusion_and_track_lost(self):
        # Establish active track
        for f in range(3):
            self.sm.update(
                frame_index=f,
                aggregated_conf={"helmet": 0.90, "no_helmet": 0.10},
                is_detected=True,
            )
        self.assertEqual(self.sm.current_state, TemporalState.HELMET_CONFIRMED)

        # Missing detection for 2 frames -> TEMPORARILY_OCCLUDED
        for f in range(3, 5):
            state, _ = self.sm.update(frame_index=f, aggregated_conf={}, is_detected=False)
        self.assertEqual(state, TemporalState.TEMPORARILY_OCCLUDED)

        # Missing detection for 11 frames -> TRACK_LOST
        for f in range(5, 16):
            state, _ = self.sm.update(frame_index=f, aggregated_conf={}, is_detected=False)
        self.assertEqual(state, TemporalState.TRACK_LOST)
        self.assertFalse(self.sm.is_active())


if __name__ == "__main__":
    unittest.main()
