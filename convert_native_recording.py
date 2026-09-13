"""Convert measured native poses into the shared visualization schema."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import teacher_adapter  # noqa: F401 - native dependency bootstrap
import numpy as np
import mujoco
from course import make_course, add_mujoco_solids
from humanoid_run import MODEL

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
r = json.loads((a.folder / "result.json").read_text())
f = np.load(a.folder / "native-rollout.npz")
course = make_course()
origin = np.array(course["cells"][r["cell"]]["origin"])
tree = ET.parse(MODEL / "model.xml").getroot()
add_mujoco_solids(tree, course)
ET.ElementTree(tree).write(a.output / "model.xml")
model = mujoco.MjModel.from_xml_path(str(a.output / "model.xml"))
# Native first pose is t=.02; align the replay with first particle frame at .04.
qpos = np.zeros((len(f["qpos"]) - 1, model.nq))
qpos[:, :7] = f["qpos"][1:, :7]
qpos[:, :3] += origin
for i, name in enumerate(f["joint_names"]):
    qpos[:, model.jnt_qposadr[model.joint(str(name)).id]] = f["qpos"][1:, 7 + i]
np.savez_compressed(a.output / "rollout.npz", qpos=qpos, control_dt=0.02)
particles = np.load(a.folder / "native-particles.npz")
positions = particles["positions"] + origin
np.savez_compressed(
    a.output / "particles.npz",
    positions=positions,
    time_s=np.arange(len(positions)) * 0.04,
    colors=particles["colors"],
)
r.update(
    cell_name=course["cells"][r["cell"]]["name"],
    live_mpm=True,
    all_active=False,
    record_start_s=0.04,
    visualization_source=str(a.folder),
    pose_source="measured Isaac Lab states, reindexed by joint name",
    course=True,
)
(a.output / "result.json").write_text(json.dumps(r, indent=2))
print(a.output)
