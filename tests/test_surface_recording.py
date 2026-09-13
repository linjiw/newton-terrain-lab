import tempfile
from pathlib import Path
import unittest
import numpy as np
from surface_recording import save_surface_frame


class SurfaceRecordingTests(unittest.TestCase):
    def test_numeric_mesh_retains_episode_identity_and_time(self):
        mesh = {
            "material": "sand",
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "indices": [0, 1, 2],
        }
        with tempfile.TemporaryDirectory() as folder:
            save_surface_frame(folder, 20, 0.005, {"0": [mesh]}, {0: 7}, [3])
            with np.load(
                Path(folder) / "frame-0000020.npz", allow_pickle=False
            ) as data:
                self.assertEqual(int(data["env0_tile"]), 7)
                self.assertEqual(int(data["env0_generation"]), 3)
                self.assertAlmostEqual(float(data["time_s"]), 0.1)
                self.assertEqual(data["env0_sand_faces"].shape, (1, 3))

    def test_rejects_nonfinite_mesh_and_out_of_range_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            for mesh in [
                dict(material="mud", vertices=[[float("nan"), 0, 0]], indices=[]),
                dict(material="mud", vertices=[[0, 0, 0]], indices=[0, 1, 2]),
            ]:
                with self.assertRaises(ValueError):
                    save_surface_frame(folder, 20, 0.005, {"0": [mesh]}, {0: 1}, [1])
