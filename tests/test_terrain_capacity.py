import unittest
import numpy as np
from dry_course import make_dry_course
from terrain_capacity import particle_count_lower_bound, capacity


class TerrainCapacityTests(unittest.TestCase):
    def test_overlap_subtraction_is_conservative(self):
        rock = dict(lo=[0, 0, 0], hi=[0.025, 0.025, 0.025])
        cell = dict(
            patches=[dict(material="sand", lo=[0, 0, 0], hi=[0.04, 0.04, 0.04])],
            rocks=[rock, rock],
        )
        xyz = np.stack(
            np.meshgrid([0.01, 0.03], [0.01, 0.03], [0.01, 0.03]), axis=-1
        ).reshape(-1, 3)
        actual = int((~np.all(xyz <= 0.025, axis=1)).sum())
        self.assertLessEqual(particle_count_lower_bound(cell, 0.04), actual)

    def test_1024_requires_larger_map_and_exposes_memory_bound(self):
        with self.assertRaises(ValueError):
            capacity(make_dry_course(), 1024, 0.04)
        result = capacity(make_dry_course(64, 64), 1024, 0.04)
        self.assertEqual(
            result["reachable_level_cases"]["0"]["particles_lower_bound"], 0
        )
        self.assertGreater(result["maximum_case_particle_state_gib_lower_bound"], 13)
