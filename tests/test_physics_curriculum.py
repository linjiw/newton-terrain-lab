import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_audit import particle_radius, audit
from course import make_course
from large_map import Curriculum, make_large_map, require_training_ready


class PhysicsCurriculumTests(unittest.TestCase):
    def test_anisotropic_sample_volume(self):
        spacing = np.array([0.02, 0.025, 0.0175])
        self.assertAlmostEqual((2 * particle_radius(spacing)) ** 3, np.prod(spacing))
        self.assertGreater(np.prod(spacing) / min(spacing) ** 3, 1.5)
        with self.assertRaises(ValueError):
            particle_radius([1, 0, 1])

    def test_old_layer_unresolved(self):
        result = audit(make_course(), 0.08)
        self.assertTrue(all(not p["resolution_screen_pass"] for p in result["patches"]))
        self.assertFalse(result["training_ready"])

    def test_curriculum_exposure_and_isolation(self):
        c = Curriculum(2)
        outcome = dict(
            fell=False,
            root_rmse=0.05,
            joint_rmse=0.1,
            exposure_seconds=0,
            episode_seconds=6,
            numerically_valid=True,
        )
        for _ in range(3):
            c.update(0, **outcome)
        self.assertEqual(c.levels, [0, 0])
        outcome["exposure_seconds"] = 3
        for _ in range(3):
            c.update(0, **outcome)
        self.assertEqual(c.levels, [1, 0])
        outcome["fell"] = True
        c.update(0, **outcome)
        self.assertEqual(c.levels, [0, 0])
        outcome["numerically_valid"] = False
        with self.assertRaises(ValueError):
            c.update(0, **outcome)
        with self.assertRaises(RuntimeError):
            require_training_ready({})

    def test_large_map_unique_cells(self):
        course = make_large_map()
        self.assertEqual(len(course["cells"]), 256)
        self.assertEqual(len({tuple(c["origin"]) for c in course["cells"]}), 256)
        self.assertEqual(len({c["name"] for c in course["cells"]}), 16)


if __name__ == "__main__":
    unittest.main()
