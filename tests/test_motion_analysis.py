import tempfile
import unittest
from pathlib import Path

from scene_analytics import SceneAnalytics


class MotionAnalysisTests(unittest.TestCase):
    def test_update_tracks_estimates_speed_and_distance(self) -> None:
        analytics = SceneAnalytics()
        analytics.set_calibration(
            pixel_to_meter_ratio=0.01,
            reference_object_size_m=0.2,
            reference_object_height_px=100,
        )

        first_frame = [{
            "track_id": 1,
            "class_name": "person",
            "center_point": [100, 100],
            "bounding_box": [90, 90, 110, 110],
            "object_name": "Alice",
            "recognition_status": "Unknown",
        }]
        second_frame = [{
            "track_id": 1,
            "class_name": "person",
            "center_point": [110, 100],
            "bounding_box": [100, 90, 120, 110],
            "object_name": "Alice",
            "recognition_status": "Unknown",
        }]

        analytics.update_tracks(first_frame, (480, 640, 3))
        analytics.update_tracks(second_frame, (480, 640, 3))

        track = analytics.tracks[1]
        self.assertGreaterEqual(track.current_speed_mps, 0.0)
        self.assertGreaterEqual(track.estimated_distance_m, 0.0)
        self.assertGreaterEqual(len(track.speed_history), 1)

    def test_calibration_settings_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            analytics = SceneAnalytics(settings_path=Path(temp_dir) / "motion.json")
            analytics.set_calibration(pixel_to_meter_ratio=0.015, minimum_safe_distance=1.8)
            self.assertAlmostEqual(analytics.pixel_to_meter_ratio, 0.015)
            self.assertAlmostEqual(analytics.minimum_safe_distance, 1.8)
            self.assertTrue((Path(temp_dir) / "motion.json").exists())


if __name__ == "__main__":
    unittest.main()
