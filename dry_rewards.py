"""Contact penalty includes external MPM forces, which PhysX sensors omit."""

import torch


def undesired_contacts(env, threshold, sensor_cfg):
    sensor = env.scene[sensor_cfg.name]
    ids = (
        list(range(len(sensor.body_names)))[sensor_cfg.body_ids]
        if isinstance(sensor_cfg.body_ids, slice)
        else list(sensor_cfg.body_ids)
    )
    native = (
        torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, :, ids, :], dim=-1)
        .max(dim=1)
        .values
        > threshold
    )
    runtime = getattr(env, "dry_terrain", None)
    if runtime is None:
        return native.sum(dim=1)
    robot_ids = [runtime.robot.body_names.index(sensor.body_names[i]) for i in ids]
    material = runtime.material_force_current[:, robot_ids] > threshold
    return (native | material).sum(dim=1)
