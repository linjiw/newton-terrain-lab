import sys
from pathlib import Path
import unittest
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course import make_course, add_mujoco_solids, export


class CourseTests(unittest.TestCase):
    def test_all_material_subsets_and_layers(self):
        course = make_course()
        groups = {
            frozenset(p["material"] for p in c["patches"]) for c in course["cells"][:15]
        }
        self.assertEqual(len(groups), 15)
        for c in course["cells"]:
            for p in c["patches"]:
                self.assertTrue(np.all(np.array(p["hi"]) > p["lo"]))
        layered = course["cells"][15]["patches"]
        self.assertEqual(layered[0]["hi"][2], layered[1]["lo"][2])

    def test_usd_and_mujoco_rigid_geometry_agree(self):
        course = make_course()
        tree = ET.Element("mujoco")
        ET.SubElement(tree, "worldbody")
        add_mujoco_solids(tree, course)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        export(Path(temp.name), course)
        stage = Usd.Stage.Open(str(Path(temp.name) / "course.usda"))
        boxes = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Cube) and prim.HasAPI(UsdPhysics.CollisionAPI):
                bbox = (
                    UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
                    .ComputeWorldBound(prim)
                    .ComputeAlignedBox()
                )
                boxes.append(np.r_[np.array(bbox.GetMin()), np.array(bbox.GetMax())])
        expected = []
        for g in tree.find("worldbody").findall("geom"):
            center = np.fromstring(g.get("pos"), sep=" ")
            half = np.fromstring(g.get("size"), sep=" ")
            expected.append(np.r_[center - half, center + half])
        self.assertEqual(len(boxes), len(expected))
        for box in expected:
            self.assertLess(min(np.max(abs(box - b)) for b in boxes), 1e-5)


if __name__ == "__main__":
    unittest.main()
