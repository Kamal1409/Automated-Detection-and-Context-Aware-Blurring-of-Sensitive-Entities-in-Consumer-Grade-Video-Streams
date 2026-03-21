import unittest

from job_manager import JobManager
from options import normalize_options


class OptionsTests(unittest.TestCase):
    def test_normalize_options_clamps_ranges(self):
        normalized = normalize_options(
            {
                "primary_subject_count": 999,
                "trusted_face_threshold": 2,
                "nudity_threshold": -1,
                "nudity_sample_stride": 0,
                "temporal_smoothing_alpha": 2,
                "temporal_blur_threshold": -5,
            }
        )

        self.assertEqual(normalized["primary_subject_count"], 5)
        self.assertEqual(normalized["trusted_face_threshold"], 0.98)
        self.assertEqual(normalized["nudity_threshold"], 0.2)
        self.assertEqual(normalized["nudity_sample_stride"], 1)
        self.assertEqual(normalized["temporal_smoothing_alpha"], 0.95)
        self.assertEqual(normalized["temporal_blur_threshold"], 0.2)


class JobManagerTests(unittest.TestCase):
    def test_create_and_get_job(self):
        manager = JobManager(max_workers=1)
        manager.create("job_1")

        job = manager.get("job_1")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["progress"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
