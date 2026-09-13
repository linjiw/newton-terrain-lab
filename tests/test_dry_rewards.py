import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    raise unittest.SkipTest(
        "Torch is optional; run this test in the native Isaac environment"
    )
from dry_rewards import undesired_contacts


class ContactFusionTests(unittest.TestCase):
    def test_name_mapping_and_union_avoid_double_count(self):
        force = torch.zeros(1, 2, 3, 3)
        force[0, 0, 0, 2] = 2
        sensor = NS(
            body_names=["knee", "ankle", "hip"], data=NS(net_forces_w_history=force)
        )
        runtime = NS(
            robot=NS(body_names=["ankle", "hip", "knee"]),
            material_force_current=torch.tensor([[0.0, 2.0, 3.0]]),
        )
        env = NS(scene={"contacts": sensor}, dry_terrain=runtime)
        cfg = NS(name="contacts", body_ids=[0, 2])
        self.assertEqual(undesired_contacts(env, 1, cfg).item(), 2)
        del env.dry_terrain
        self.assertEqual(undesired_contacts(env, 1, cfg).item(), 1)


if __name__ == "__main__":
    unittest.main()
