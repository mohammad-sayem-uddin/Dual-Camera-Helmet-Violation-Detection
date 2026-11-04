"""Unit Tests for Multi-Target Association and Track Recovery."""
import unittest
from src.temporal.rider_tracker import BoundingBox, RiderTracker
from src.temporal.state_machine import TemporalState


class TestTrackRecovery(unittest.TestCase):
    def test_multi_rider_independent_association(self):
        tracker = RiderTracker(iou_threshold=0.30)

        # Frame 0: Rider A (left) and Rider B (right)
        box_a = BoundingBox(50.0, 100.0, 100.0, 180.0, 0.95, 1, "helmet")
        box_b = BoundingBox(300.0, 100.0, 350.0, 180.0, 0.92, 0, "no_helmet")

        tracks0 = tracker.update(frame_index=0, detections=[box_a, box_b])
        self.assertEqual(len(tracks0), 2)
        tid_a = tracks0[0].track_id
        tid_b = tracks0[1].track_id

        # Frame 1: Both move forward smoothly
        box_a1 = BoundingBox(55.0, 100.0, 105.0, 180.0, 0.94, 1, "helmet")
        box_b1 = BoundingBox(305.0, 100.0, 355.0, 180.0, 0.91, 0, "no_helmet")

        tracks1 = tracker.update(frame_index=1, detections=[box_a1, box_b1])
        self.assertEqual(len(tracks1), 2)

        # Verify identities preserved without swapping
        t_map = {t.track_id: t for t in tracks1}
        self.assertIn(tid_a, t_map)
        self.assertIn(tid_b, t_map)
        self.assertEqual(t_map[tid_a].current_bbox.class_name, "helmet")
        self.assertEqual(t_map[tid_b].current_bbox.class_name, "no_helmet")

    def test_velocity_projection_aids_recovery(self):
        tracker = RiderTracker(iou_threshold=0.20, max_age=5)

        # Rider moving at dx = +15 px/frame
        b0 = BoundingBox(100.0, 200.0, 150.0, 270.0, 0.90, 1, "helmet")
        b1 = BoundingBox(115.0, 200.0, 165.0, 270.0, 0.90, 1, "helmet")
        tracker.update(frame_index=0, detections=[b0])
        tracks = tracker.update(frame_index=1, detections=[b1])
        tid = tracks[0].track_id

        # Occluded for 1 frame (frame 2)
        tracker.update(frame_index=2, detections=[])

        # Frame 3: Rider appears at extrapolated position (~145 px)
        b3 = BoundingBox(145.0, 200.0, 195.0, 270.0, 0.91, 1, "helmet")
        tracks3 = tracker.update(frame_index=3, detections=[b3])

        self.assertEqual(len(tracks3), 1)
        self.assertEqual(tracks3[0].track_id, tid)


if __name__ == "__main__":
    unittest.main()
