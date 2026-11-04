"""Unit Tests for Violation Decision Logic and Alert Deduplication."""
import unittest
from src.temporal.rider_tracker import BoundingBox, RiderTracker
from src.temporal.state_machine import StateMachineConfig
from src.temporal.violation_decision import DecisionConfig, ViolationDecisionEngine


class TestViolationDecision(unittest.TestCase):
    def setUp(self):
        sm_config = StateMachineConfig(min_observation_frames=3, confirm_frames=4, confirm_threshold=0.70)
        self.tracker = RiderTracker(iou_threshold=0.30, sm_config=sm_config)
        self.decision_config = DecisionConfig(
            violation_alert_threshold=0.65,
            min_confirmation_frames=4,
            suppress_duplicates_per_track=True,
            cooldown_frames=60,
        )
        self.decision_engine = ViolationDecisionEngine(config=self.decision_config)

    def test_compliant_rider_generates_zero_alerts(self):
        # 10 consecutive helmet frames
        for f in range(10):
            box = BoundingBox(100.0, 100.0, 160.0, 180.0, 0.95, 1, "helmet")
            tracks = self.tracker.update(frame_index=f, detections=[box])
            alerts = self.decision_engine.process_tracks(frame_index=f, tracks=tracks)
            self.assertEqual(len(alerts), 0)

        self.assertEqual(len(self.decision_engine.alerts_log), 0)

    def test_violation_rider_generates_exactly_one_authoritative_alert(self):
        all_emitted_alerts = []

        # Feed 15 consecutive violation frames
        for f in range(15):
            box = BoundingBox(100.0, 100.0, 160.0, 180.0, 0.92, 0, "no_helmet")
            tracks = self.tracker.update(frame_index=f, detections=[box])
            new_alerts = self.decision_engine.process_tracks(frame_index=f, tracks=tracks)
            all_emitted_alerts.extend(new_alerts)

        # Must generate exactly 1 authoritative alert across the entire trajectory
        self.assertEqual(len(all_emitted_alerts), 1)
        alert = all_emitted_alerts[0]
        self.assertEqual(alert.track_id, 1)
        self.assertGreaterEqual(alert.violation_confidence, 0.65)
        self.assertIn("VIOLATION-1-", alert.alert_id)

    def test_suppression_prevents_alert_flooding(self):
        # Reach confirmed violation
        for f in range(5):
            box = BoundingBox(100.0, 100.0, 160.0, 180.0, 0.90, 0, "no_helmet")
            tracks = self.tracker.update(frame_index=f, detections=[box])
            alerts = self.decision_engine.process_tracks(frame_index=f, tracks=tracks)

        self.assertEqual(len(self.decision_engine.alerts_log), 1)

        # Continue feeding 20 more frames: alerts_log must remain 1
        for f in range(5, 25):
            box = BoundingBox(100.0, 100.0, 160.0, 180.0, 0.90, 0, "no_helmet")
            tracks = self.tracker.update(frame_index=f, detections=[box])
            alerts = self.decision_engine.process_tracks(frame_index=f, tracks=tracks)
            self.assertEqual(len(alerts), 0)

        self.assertEqual(len(self.decision_engine.alerts_log), 1)


if __name__ == "__main__":
    unittest.main()
