"""Local Unix-socket MPM worker, isolating Newton from Isaac Sim dependencies."""

import argparse
import json
import hashlib
import warp as wp
import socket
import xml.etree.ElementTree as ET
from pathlib import Path
import teacher_adapter  # noqa: F401 - native dependency bootstrap
import numpy as np
import mujoco
from course import make_course, add_mujoco_solids
from mpm_bridge import MPMBridge
from humanoid_run import MODEL
from sonic_mujoco_forecast_v2 import rotation, com_to_link_velocity

p = argparse.ArgumentParser()
p.add_argument("--socket", required=True)
p.add_argument("--cell", type=int, required=True)
p.add_argument("--voxel-size", type=float, default=0.08)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
course = make_course()
offset = np.array(course["cells"][a.cell]["origin"])
for c in course["cells"]:
    c["origin"] = (np.array(c["origin"]) - offset).tolist()
for w in course.get("walkways", []):
    w["lo"] = (np.array(w["lo"]) - offset).tolist()
    w["hi"] = (np.array(w["hi"]) - offset).tolist()
tree = ET.parse(MODEL / "model.xml").getroot()
add_mujoco_solids(tree, course)
model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
data = mujoco.MjData(model)
bridge = None
frames = []
steps = 0
peak = 0.0
reset_receipts = []
initial_digest = None
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(a.socket)
listener.listen(1)
conn, _ = listener.accept()
stream = conn.makefile("rwb")
try:
    while True:
        line = stream.readline()
        if not line:
            break
        message = json.loads(line)
        if message.get("close"):
            break
        if message.get("reset"):
            wp.synchronize()
            bridge = None  # Rebuild all particle, plastic, grid and warmstart state.
            reset_receipts.append(
                {"step": steps, "method": "full_bridge_reconstruction"}
            )
            stream.write(
                (
                    json.dumps({"reset": True, "generation": len(reset_receipts)})
                    + "\n"
                ).encode()
            )
            stream.flush()
            continue
        state = np.asarray(message["root_state"])
        q = state[3:7]
        r = rotation(q)
        data.qpos[:3] = state[:3]
        data.qpos[3:7] = q
        data.qvel[:3] = com_to_link_velocity(
            state[7:10], state[10:13], r, model.body_ipos[1]
        )
        data.qvel[3:6] = r.T @ state[10:13]
        for name, pos, vel in zip(
            message["joint_names"], message["joint_pos"], message["joint_vel"]
        ):
            j = model.joint(name).id
            data.qpos[model.jnt_qposadr[j]] = pos
            data.qvel[model.jnt_dofadr[j]] = vel
        mujoco.mj_forward(model, data)
        if bridge is None:
            bridge = MPMBridge(
                model, data, [course["cells"][a.cell]], voxel_size=a.voxel_size
            )
            digest = hashlib.sha256(
                bridge.state.particle_q.numpy().tobytes()
            ).hexdigest()
            if initial_digest is None:
                initial_digest = digest
            elif digest != initial_digest:
                raise RuntimeError("Reset initial particle geometry differs")
            if np.any(bridge.previous_wrench):
                raise RuntimeError("Reset retained coupling wrench")
            if reset_receipts:
                reset_receipts[-1]["initial_geometry_matches"] = True
                reset_receipts[-1]["previous_wrench_zero"] = True
        wrench = bridge.step_mujoco(model, data, message["dt"])
        names = message["body_names"]
        target_com = np.asarray(message["body_com_pos"])
        out = np.zeros((len(names), 6))
        for b in range(1, model.nbody):
            ancestor = b
            while ancestor and model.body(ancestor).name not in names:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == 0:
                if np.linalg.norm(wrench[b]) > 1e-8:
                    raise ValueError("Unmapped force-bearing proxy body")
                continue
            index = names.index(model.body(ancestor).name)
            out[index, :3] += wrench[b, :3]
            out[index, 3:] += wrench[b, 3:] + np.cross(
                data.xipos[b] - target_com[index], wrench[b, :3]
            )
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite mapped wrench")
        peak = max(peak, float(np.linalg.norm(out[:, :3], axis=1).max()))
        steps += 1
        if steps % 8 == 0:
            frames.append(bridge.state.particle_q.numpy()[::4])
        response = {"wrenches": out.tolist()}
        if steps % 8 == 0:
            response.update(
                particle_positions=frames[-1].tolist(),
                particle_colors=bridge.colors[::4].tolist(),
            )
        stream.write((json.dumps(response) + "\n").encode())
        stream.flush()
finally:
    if bridge is not None:
        np.savez_compressed(
            a.output / "native-particles.npz",
            positions=frames,
            colors=bridge.colors[::4],
            dt=0.04,
        )
    (a.output / "mpm-worker.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "reset_receipts": reset_receipts,
                "voxel_size_m": a.voxel_size,
                "peak_material_force_N": peak,
                "coupling": "host measured state to nominal collision proxies; world COM wrench feedback",
            },
            indent=2,
        )
    )
    stream.close()
    conn.close()
    listener.close()
    Path(a.socket).unlink(missing_ok=True)
