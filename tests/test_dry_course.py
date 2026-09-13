import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dry_course import make_dry_course, TileAllocator
from course import solids
from isaac_terrain_curriculum import TerrainCurriculumState


class DryCourseTests(unittest.TestCase):
    def test_water_excluded_and_rocks_match_shared_geometry(self):
        c = make_dry_course()
        self.assertEqual(len(c["cells"]), 256)
        self.assertNotIn("water", c["material_presets"])
        self.assertEqual(len({cell["name"] for cell in c["cells"]}), 15)
        for cell in c["cells"]:
            self.assertNotIn("water", [p["material"] for p in cell["patches"]])
            names = {name for name, *_ in solids(cell)}
            self.assertTrue(all(r["name"] in names for r in cell["rocks"]))

    def test_exclusive_seeded_allocation_and_selective_reset(self):
        course = make_dry_course()
        a = TileAllocator(course, 8, 11)
        b = TileAllocator(course, 8, 11)
        levels = [i % 4 for i in range(8)]
        first = a.assign(range(8), levels)
        self.assertEqual(first, b.assign(range(8), levels))
        self.assertEqual(len(set(first.values())), 8)
        a.assign([0], levels)
        self.assertEqual(
            {i: a.assignments[i] for i in range(1, 8)},
            {i: first[i] for i in range(1, 8)},
        )
        self.assertEqual(len(set(a.assignments.values())), 8)
        with self.assertRaises(ValueError):
            TileAllocator(course, 65)

    def test_ground_curriculum_can_promote_without_material(self):
        c = TerrainCurriculumState(1)
        for _ in range(3):
            c.record(0, 0.03, 0.1, 0, 6)
            c.finish([0])
        self.assertEqual(c.scheduler.levels[0], 1)
        for _ in range(3):
            c.record(0, 0.03, 0.1, 0, 6)
            c.finish([0])
        self.assertEqual(c.scheduler.levels[0], 1)


if __name__ == "__main__":
    unittest.main()
