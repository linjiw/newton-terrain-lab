from terrain_settings import setting

"""Frozen motion2scene teacher evaluation in a named MuJoCo plant."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
from pathlib import Path
import time
import shutil
import xml.etree.ElementTree as ET
from course import make_course, add_mujoco_solids
import teacher_adapter
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from sonic_mujoco_forecast_v2 import NominalPlant, clean_actor_history, reference_inputs
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER

ROOT = Path(__file__).resolve().parent
MODEL = Path(setting("model_dir"))
REFERENCES = Path(setting("reference_dir"))


def reference(path):
    f = np.load(path)
    old = f["time_s"]
    old = old - old[0]
    t = np.arange(0, old[-1] + 1e-7, 0.02)
    q = f["qpos"]
    names = list(f["joint_names"])
    joints = np.column_stack(
        [np.interp(t, old, q[:, 7 + names.index(n)]) for n in G1_ISAACLab_ORDER]
    )
    pos = np.column_stack([np.interp(t, old, q[:, i]) for i in range(3)])
    rotations = Slerp(old, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(t)
    quat = rotations.as_quat()[:, [3, 0, 1, 2]]
    return dict(
        dof_pos=joints,
        dof_vel=np.gradient(joints, 0.02, axis=0),
        body_quat_w_full=quat[:, None],
        root_pos=pos,
        quat=quat,
        t=t,
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    source_hashes = {
        path.name: teacher_adapter.sonic.sha(path) for path in snapshot.glob("*.py")
    }
    ref_path = REFERENCES / args.motion / "reference.npz"
    ref = reference(ref_path)
    course = make_course()
    cell = course["cells"][args.cell]
    tree = ET.parse(MODEL / "model.xml").getroot()
    if args.course:
        add_mujoco_solids(tree, course)
        offset = np.array(cell["origin"])
        ref["root_pos"] += offset
    model_path = args.output / "model.xml"
    ET.ElementTree(tree).write(model_path)
    plant = NominalPlant(model_path, MODEL / "drives.json", G1_ISAACLab_ORDER)
    actor = teacher_adapter.load_teacher()
    zero = np.zeros(29)
    initial = dict(
        state_time_s=0.0,
        action_start_time_s=-0.02,
        action_end_time_s=0.0,
        root_pos_w=ref["root_pos"][0],
        root_quat_w=ref["quat"][0],
        root_lin_vel_w=np.gradient(ref["root_pos"], 0.02, axis=0)[0],
        root_ang_vel_w=np.zeros(3),
        dof_pos=ref["dof_pos"][0],
        dof_vel=ref["dof_vel"][0],
        applied_joint_action=zero,
    )
    history = {
        k: np.repeat(np.asarray(v)[None], 10, axis=0) for k, v in initial.items()
    }
    plant.initialize(history)
    bridge = None
    particle_frames = []
    particle_times = []
    peak_force = 0.0
    work = 0.0
    saturation = 0
    substeps = 0
    if args.mpm:
        if not args.course:
            raise ValueError("--mpm requires --course")
        from mpm_bridge import MPMBridge

        active = course["cells"] if args.all_active else [cell]
        if any(p["material"] != "ground" for c in active for p in c["patches"]):
            bridge = MPMBridge(
                plant.model, plant.data, active, voxel_size=args.voxel_size
            )
    if bridge:
        particle_frames.append(bridge.state.particle_q.numpy()[:: args.particle_stride])
        particle_times.append(0.0)
    rows = []
    poses = [plant.data.qpos.copy()]
    actions = []
    reason = "horizon"
    for tick in range(min(int(args.seconds / 0.02), len(ref["t"]) - 1)):
        obs = clean_actor_history(history, plant.default_pos, plant.default_vel)
        cmd, ori, _ = reference_inputs(
            ref, tick, history["root_quat_w"][-1], G1_ISAACLab_ORDER
        )
        action = actor.infer(obs, cmd, ori)["environment_action"][0].numpy()
        if bridge:
            target = action * plant.scale + plant.default_pos
            torque = np.zeros(29)
            ratio = np.zeros(29)
            for _ in range(4):
                wrench = bridge.step_mujoco(plant.model, plant.data, 0.005)
                peak_force = max(
                    peak_force, float(np.linalg.norm(wrench[:, :3], axis=1).max())
                )
                plant.data.xfrc_applied[:] = wrench
                raw = (
                    plant.kp * (target - plant.data.qpos[plant.qadr])
                    - plant.kd * plant.data.qvel[plant.vadr]
                )
                applied = np.clip(raw, -plant.effort, plant.effort)
                plant.data.ctrl[plant.motor] = applied
                work += float(
                    np.abs(applied * plant.data.qvel[plant.vadr]).sum() * 0.005
                )
                saturation += int((abs(raw) > plant.effort).sum())
                substeps += 1
                torque = np.maximum(torque, abs(applied))
                mujoco.mj_step(plant.model, plant.data)
                ratio = np.maximum(
                    ratio, abs(plant.data.qvel[plant.vadr]) / plant.velocity_limit
                )
            if tick % 2 == 1:
                particle_frames.append(
                    bridge.state.particle_q.numpy()[:: args.particle_stride]
                )
                particle_times.append((tick + 1) * 0.02)
        else:
            target, torque, ratio = plant.advance(action)
        state = plant.snapshot(tick + 1, action)
        if (
            not np.isfinite(plant.data.qpos).all()
            or not np.isfinite(plant.data.qvel).all()
        ):
            raise FloatingPointError("Nonfinite host dynamics")
        err = float(np.linalg.norm(state["root_pos_w"] - ref["root_pos"][tick + 1]))
        rows.append(
            dict(
                time_s=(tick + 1) * 0.02,
                root_error_m=err,
                root_height_m=float(state["root_pos_w"][2]),
                joint_rmse_rad=float(
                    np.sqrt(np.mean((state["dof_pos"] - ref["dof_pos"][tick + 1]) ** 2))
                ),
                max_torque_Nm=float(torque.max()),
                max_velocity_limit_ratio=float(ratio.max()),
            )
        )
        poses.append(plant.data.qpos.copy())
        actions.append(action)
        history = {
            k: np.concatenate((v[1:], np.asarray(state[k])[None]))
            for k, v in history.items()
        }
        if state["root_pos_w"][2] < 0.3:
            reason = "fall"
            break
    np.savez_compressed(
        args.output / "rollout.npz",
        qpos=poses,
        actions=actions,
        joint_names=G1_ISAACLab_ORDER,
        reference_root=ref["root_pos"][: len(poses)],
        control_dt=0.02,
    )
    if bridge:
        np.savez_compressed(
            args.output / "particles.npz",
            positions=particle_frames,
            time_s=particle_times,
            colors=bridge.colors[:: args.particle_stride],
            voxel_size=args.voxel_size,
        )
    result = dict(
        source_sha256=source_hashes,
        all_active=args.all_active,
        voxel_size_m=args.voxel_size if bridge else None,
        physics_dt_s=0.005,
        control_dt_s=0.02,
        cell=args.cell,
        cell_name=cell["name"],
        course=args.course,
        live_mpm=bridge is not None,
        particle_count=bridge.model.particle_count if bridge else 0,
        peak_material_force_N=peak_force,
        positive_absolute_joint_work_J=work if bridge else None,
        torque_saturation_fraction=saturation / (substeps * 29) if substeps else None,
        status="executed",
        backend="mujoco+newton_mpm" if bridge else "mujoco",
        teacher_sha256=teacher_adapter.CHECKPOINT_SHA,
        motion=args.motion,
        reference_sha256=teacher_adapter.sonic.sha(ref_path),
        stop_reason=reason,
        steps=len(rows),
        simulated_seconds=len(rows) * 0.02,
        wall_seconds=time.monotonic() - start,
        root_rmse_m=float(np.sqrt(np.mean([r["root_error_m"] ** 2 for r in rows]))),
        max_root_error_m=max(r["root_error_m"] for r in rows),
        final_root_height_m=rows[-1]["root_height_m"],
        limitations=[
            "Approximate imported plant, not PhysX parity",
            "Repeated initial observation history",
            "Nominal drives, no training-time randomization",
            "Reference finite-difference velocities",
            "Uncalibrated MPM materials and coarse grid",
            "Explicit lagged coupling with per-link effective inertia",
            "Only selected cell has live MPM unless --all-active",
        ],
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--motion", default="00265")
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--course", action="store_true")
    p.add_argument("--mpm", action="store_true")
    p.add_argument("--cell", type=int, choices=range(16), default=0)
    p.add_argument("--voxel-size", type=float, default=0.08)
    p.add_argument("--all-active", action="store_true")
    p.add_argument("--particle-stride", type=int, default=4)
    run(p.parse_args())
