"""Batched IPC, independent local MPM worlds; selective resets leave others intact."""

import argparse
import copy
import hashlib
import json
import socket
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import warp as wp
from newton.geometry import ParticleSurface
from mpm_bridge import MPMBridge
from terrain_settings import setting

MODEL = Path(setting("model_dir"))
from course import add_mujoco_solids
from rigid_state import rotation, com_to_link_velocity


class World:
    def __init__(self, cell, h):
        self.cell = copy.deepcopy(cell)
        self.cell["origin"] = [0, 0, 0]
        self.h = h
        tree = ET.parse(MODEL / "model.xml").getroot()
        add_mujoco_solids(tree, dict(cells=[self.cell], walkways=[]))
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(tree, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.bridge = None
        self.steps = 0
        self.surfaces = {}

    def step(self, msg):
        model, data = self.model, self.data
        state = np.asarray(msg["root_state"])
        r = rotation(state[3:7])
        data.qpos[:7] = state[:7]
        data.qvel[:3] = com_to_link_velocity(
            state[7:10], state[10:13], r, model.body_ipos[1]
        )
        data.qvel[3:6] = r.T @ state[10:13]
        for name, pos, vel in zip(
            msg["joint_names"], msg["joint_pos"], msg["joint_vel"]
        ):
            j = model.joint(name).id
            data.qpos[model.jnt_qposadr[j]] = pos
            data.qvel[model.jnt_dofadr[j]] = vel
        mujoco.mj_forward(model, data)
        if self.bridge is None and any(
            p["material"] != "ground" for p in self.cell["patches"]
        ):
            self.bridge = MPMBridge(model, data, [self.cell], voxel_size=self.h)
        out = np.zeros((len(msg["body_names"]), 6))
        if self.bridge is not None:
            wrench = self.bridge.step_mujoco(model, data, msg["dt"])
            names = msg["body_names"]
            com = np.asarray(msg["body_com_pos"])
            for b in range(1, model.nbody):
                ancestor = b
                while ancestor and model.body(ancestor).name not in names:
                    ancestor = int(model.body_parentid[ancestor])
                if not ancestor:
                    if np.linalg.norm(wrench[b]) > 1e-8:
                        raise ValueError("Unmapped wrench")
                    continue
                i = names.index(model.body(ancestor).name)
                out[i, :3] += wrench[b, :3]
                out[i, 3:] += wrench[b, 3:] + np.cross(
                    data.xipos[b] - com[i], wrench[b, :3]
                )
        self.steps += 1
        if not np.isfinite(out).all():
            raise ValueError("Invalid force")
        return out.tolist()

    def surface_meshes(self):
        if self.bridge is None:
            return []
        meshes = []
        with wp.ScopedDevice(self.bridge.device):
            positions = self.bridge.state.particle_q.numpy()
            for name in ["sand", "mud"]:
                ids = (
                    np.concatenate(
                        [
                            np.arange(lo, hi)
                            for lo, hi, n in self.bridge.ranges
                            if n == name
                        ]
                    )
                    if any(n == name for _, _, n in self.bridge.ranges)
                    else np.array([], dtype=int)
                )
                if not len(ids):
                    continue
                if name not in self.surfaces:
                    self.surfaces[name] = ParticleSurface(
                        voxel_size=self.h * 0.35,
                        kernel_radius=self.h * 1.4,
                        threshold=0.25,
                        field_smooth_iterations=1,
                    )
                v, f, _ = (
                    self.surfaces[name]
                    .extract(
                        wp.array(positions[ids], dtype=wp.vec3),
                        wp.array(
                            self.bridge.model.particle_radius.numpy()[ids], dtype=float
                        ),
                    )
                    .to_arrays()
                )
                if v is not None:
                    meshes.append(
                        dict(
                            material=name,
                            vertices=v.numpy().tolist(),
                            indices=f.numpy().tolist(),
                        )
                    )
        return meshes

    def digest(self):
        if self.bridge is None:
            return "uninitialized"
        state = self.bridge.state
        arrays = [
            state.particle_q,
            state.particle_qd,
            state.mpm.particle_qd_grad,
            state.mpm.particle_elastic_strain,
            state.mpm.particle_Jp,
            state.mpm.particle_stress,
        ]
        return hashlib.sha256(
            b"".join(a.numpy().tobytes() for a in arrays)
            + self.bridge.previous_wrench.tobytes()
        ).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--course", type=Path, required=True)
    p.add_argument("--voxel-size", type=float, default=0.04)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    course = json.loads(a.course.read_text())
    worlds = {}
    events = []
    ticks = 0
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(a.socket)
    listener.listen(1)
    conn, _ = listener.accept()
    stream = conn.makefile("rwb")
    try:
        while line := stream.readline():
            msg = json.loads(line)
            if msg.get("close"):
                break
            if "reset" in msg:
                replacements = {int(k): int(v) for k, v in msg["reset"].items()}
                before = {
                    k: w.digest() for k, w in worlds.items() if k not in replacements
                }
                for k, tile in replacements.items():
                    worlds[k] = World(course["cells"][tile], a.voxel_size)
                preserved = all(worlds[k].digest() == v for k, v in before.items())
                if not preserved:
                    raise RuntimeError("Selective reset changed another world")
                events.append(
                    dict(
                        reset=replacements,
                        other_worlds_unchanged=preserved,
                        checked_worlds=list(before),
                    )
                )
                response = dict(reset=True, other_worlds_unchanged=preserved)
            else:
                response = dict(
                    wrenches=[worlds[int(m["env_id"])].step(m) for m in msg["batch"]]
                )
                ticks += 1
                if msg.get("render") and ticks % 20 == 0:
                    response["surfaces"] = {
                        i: w.surface_meshes() for i, w in worlds.items()
                    }
            stream.write((json.dumps(response) + "\n").encode())
            stream.flush()
    finally:
        a.receipt.write_text(
            json.dumps(
                dict(
                    batch_physics_ticks=ticks,
                    resets=events,
                    worlds={
                        k: dict(
                            steps=w.steps,
                            particles=0
                            if w.bridge is None
                            else w.bridge.model.particle_count,
                        )
                        for k, w in worlds.items()
                    },
                ),
                indent=2,
            )
        )
        stream.close()
        conn.close()
        listener.close()
        Path(a.socket).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
