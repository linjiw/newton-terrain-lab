"""Host state conventions: wxyz quaternion, world COM velocity."""

import numpy as np
import mujoco


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def com_to_link_velocity(v_com_world, omega_world, world_from_link, com_offset_link):
    offset_world = world_from_link @ np.asarray(com_offset_link)
    return np.asarray(v_com_world) - np.cross(omega_world, offset_world)
