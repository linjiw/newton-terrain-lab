"""Interactive measured-motion playback: Space pauses, R restarts, Esc exits."""

import argparse
import json
import time
from pathlib import Path
import teacher_adapter  # noqa: F401 - native dependency bootstrap
import numpy as np
import mujoco
import mujoco.viewer
from render_course import build_model, visible_particles

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--speed", type=float, default=1.0)
a = p.parse_args()
if a.speed <= 0:
    p.error("--speed must be positive")
r = json.loads((a.folder / "result.json").read_text())
f = np.load(a.folder / "rollout.npz")
source = mujoco.MjModel.from_xml_path(str(a.folder / "model.xml"))
model, course = build_model(r)
data = mujoco.MjData(model)
mapping = []
for j in range(1, source.njnt):
    dest = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, source.joint(j).name)
    if dest >= 0:
        mapping.append((int(source.jnt_qposadr[j]), int(model.jnt_qposadr[dest])))
particles = (
    np.load(a.folder / "particles.npz")
    if (a.folder / "particles.npz").exists()
    else None
)
ui = {"paused": False, "restart": False}


def key(k):
    if k == 32:
        ui["paused"] = not ui["paused"]
    if k in (82, 114):
        ui["restart"] = True


with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
    viewer.cam.lookat[:] = np.array(course["cells"][r["cell"]]["origin"]) + [
        0.85,
        0.5,
        0.55,
    ]
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22
    viewer.cam.distance = 4.8
    index = 0
    last = time.monotonic()
    while viewer.is_running():
        if ui["restart"]:
            index = 0
            ui["restart"] = False
        if not ui["paused"] and time.monotonic() - last >= 0.02 / a.speed:
            index = (index + 1) % len(f["qpos"])
            last = time.monotonic()
        q = f["qpos"][index]
        data.qpos[:7] = q[:7]
        for x, y in mapping:
            data.qpos[y] = q[x]
        mujoco.mj_forward(model, data)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            if particles is not None:
                k = max(
                    0,
                    int(
                        np.searchsorted(particles["time_s"], index * 0.02, side="right")
                        - 1
                    ),
                )
                for xyz, color in zip(
                    *visible_particles(
                        particles["positions"][k],
                        particles["colors"],
                        course["cells"][r["cell"]]["origin"],
                        budget=viewer.user_scn.maxgeom,
                    )
                ):
                    scene = viewer.user_scn
                    if scene.ngeom >= scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.022] * 3),
                        xyz,
                        np.eye(3).flatten(),
                        np.r_[color, 1.0],
                    )
                    scene.ngeom += 1
        viewer.sync()
        time.sleep(0.005)
