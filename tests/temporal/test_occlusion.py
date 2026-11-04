"""Unit Tests for Occlusion Handling and Trajectory Extrapolation."""
import unittest
from src.temporal.rider_tracker import BoundingBox, RiderTracker
from src.temporal.state_machine import TemporalState


class TestOcclusionHandling(unittest.TestCase):
    def setUp(self):
        self.tracker = RiderTracker(iou_threshold=0.30, max_age=5)

    def test_short_occlusion_maintains_track_identity(self):
        # Frame 0: Detect rider
        b0 = BoundingBox(100.0, 100.0, 150.0, 150.0, 0.90, 1, "helmet")
        tracks = self.tracker.update(frame_index=0, detections=[b0])
        self.assertEqual(len(tracks), 1)
        orig_tid = tracks[0].track_id

        # Frames 1-3: Rider occluded (no detections)
        for f in range(1, 4):
            tracks = self.tracker.update(frame_index=f, detections=[])
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].track_id, orig_tid)
            self.assertEqual(tracks[0].state, TemporalState.TEMPORARILY_OCCLUDED)

        # Frame 4: Rider reappears near extrapolated position
        b4 = BoundingBox(105.0, 105.0, 155.0, 155.0, 0.92, 1, "helmet")
        tracks = self.tracker.update(frame_index=4, detections=[b4])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, orig_tid)
        self.assertEqual(tracks[0].state, TemporalState.HELMET_CONFIRMED)

    def test_long_occlusion_expires_track(self):
        # Frame 0: Detect rider
        b0 = BoundingBox(100.0, 100.0, 150.0, 150.0, 0.90, 0, "no_helmet")
        self.tracker.update(frame_index=0, detections=[b0])

        # Exceed max_age (6 empty frames)
        for f in range(1, 8):
            self.tracker.update(frame_index=f, detections=[])

        # Track should be purged from active tracks
        self.assertEqual(len(self.tracker.active_tracks), 0)
        self.assertEqual(len(self.tracker.dead_tracks), 1)


if __name__ == "__main__":
    unittest.main()
