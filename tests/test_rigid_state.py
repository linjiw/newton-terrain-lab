import unittest
import numpy as np
from rigid_state import rotation, com_to_link_velocity


class RigidStateTests(unittest.TestCase):
    def test_rotating_offset_com_has_stationary_link_origin(self):
        # A 90-degree Z rotation carries the local X COM offset to world Y.
        r = rotation([2**-0.5, 0, 0, 2**-0.5])
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        velocity = com_to_link_velocity([-2, 0, 0], [0, 0, 2], r, [1, 0, 0])
        np.testing.assert_allclose(velocity, [0, 0, 0], atol=1e-12)
