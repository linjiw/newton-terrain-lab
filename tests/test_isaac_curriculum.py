import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_terrain_curriculum import TerrainCurriculumState


class EpisodeMetricsTests(unittest.TestCase):
    def test_selected_reset_preserves_other_episode(self):
        c = TerrainCurriculumState(2)
        c.record(0, 0.03, 0.1, 0.02, 0.02)
        c.record(1, 0.04, 0.2, 0, 0.02)
        r = c.finish([0])
        self.assertAlmostEqual(r[0]["root_rmse"], 0.03)
        self.assertEqual(len(c.samples[0]), 0)
        self.assertEqual(len(c.samples[1]), 1)
        self.assertEqual(c.finish([0]), [])
        c.scheduler.levels[1] = 1
        c.finish([1], fell=True)
        self.assertEqual(c.scheduler.levels[1], 0)

    def test_reject_bad_exposure(self):
        c = TerrainCurriculumState(1)
        with self.assertRaises(ValueError):
            c.record(0, 0.1, 0.1, 0.03, 0.02)
        with self.assertRaises(ValueError):
            c.record(0, float("nan"), 0.1, 0, 0.02)


if __name__ == "__main__":
    unittest.main()
