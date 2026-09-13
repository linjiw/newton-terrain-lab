from terrain_settings import setting

"""Attach independent material worlds to native Isaac Lab training/reset paths."""

import json
import socket
import subprocess
import time
import os
from pathlib import Path
import numpy as np
import torch
from dry_course import TileAllocator
from isaac_terrain_curriculum import TerrainCurriculumState
from course import COLORS


class DryTerrainRuntime:
    def __init__(self, wrapper, config):
        self.wrapper = wrapper
        self.env = wrapper.env
        self.config = config
        self.env.dry_terrain = self
        self.surface_updates = 0
        self.physics_ticks = 0
        self.course = json.loads(Path(config["course"]).read_text())
        self.allocator = TileAllocator(self.course, self.env.num_envs, config["seed"])
        self.metrics = TerrainCurriculumState(self.env.num_envs)
        self.metrics.scheduler.levels = [
            config.get("initial_level", 0)
        ] * self.env.num_envs
        self.robot = self.env.scene["robot"]
        self.command = wrapper.motion_command
        self.material_force_current = torch.zeros(
            (self.env.num_envs, len(self.robot.body_names)), device=self.env.device
        )
        self.material_force_accumulator = torch.zeros_like(self.material_force_current)
        self.exposure = np.zeros(self.env.num_envs)
        self.generations = np.zeros(self.env.num_envs, dtype=int)
        self.reset_events = []
        self.closed = False
        out = Path(config["output"])
        out.mkdir(parents=True, exist_ok=True)
        sock = str(out / "dry-mpm.sock")
        self.log = (out / "worker.log").open("x")
        self.worker = subprocess.Popen(
            [
                setting("newton_python"),
                str(Path(__file__).with_name("dry_mpm_worker.py")),
                "--socket",
                sock,
                "--course",
                config["course"],
                "--voxel-size",
                str(config["voxel_size"]),
                "--receipt",
                str(out / "worker.json"),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "2"},
        )
        deadline = time.monotonic() + 90
        while not Path(sock).exists():
            if self.worker.poll() is not None:
                raise RuntimeError("Dry MPM worker failed to start")
            if time.monotonic() > deadline:
                raise TimeoutError("Dry worker startup")
            time.sleep(0.1)
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(180)
        self.conn.connect(sock)
        self.stream = self.conn.makefile("rwb")
        self.original_step = self.env.sim.step
        self.original_reset = self.env._reset_idx
        self.original_compute = self.env.termination_manager.compute
        self.env.sim.step = self.step
        self.env._reset_idx = self.reset
        self.env.termination_manager.compute = self.compute
        self.assign(list(range(self.env.num_envs)))

    def request(self, msg):
        self.stream.write((json.dumps(msg) + "\n").encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError("Dry worker disconnected; inspect worker.log")
        return json.loads(line)

    def assign(self, ids):
        previous = {i: self.allocator.assignments.get(i) for i in ids}
        if self.config.get("eval_tiles") is not None:
            chosen = self.allocator.assign_tiles(
                {i: self.config["eval_tiles"][i] for i in ids}
            )
            for i in ids:
                self.metrics.scheduler.levels[i] = self.course["cells"][chosen[i]][
                    "level"
                ]
        else:
            chosen = self.allocator.assign(ids, self.metrics.scheduler.levels)
        origins = []
        for i in ids:
            origin = np.array(self.course["cells"][chosen[i]]["origin"], dtype=float)
            # Random spawn only in the known clear starting strip.
            spawn = origin + np.array(
                [
                    self.allocator.rng.uniform(-0.08, 0.05),
                    self.allocator.rng.uniform(-0.1, 0.1),
                    0,
                ]
            )
            origins.append(spawn)
            self.generations[i] += 1
        self.env.scene.env_origins[ids] = torch.tensor(
            np.array(origins), device=self.env.device, dtype=torch.float32
        )
        if self.config.get("visualize"):
            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for i in ids:
                stage.RemovePrim(f"/World/DryLive/env_{i}")
            visibility = {tile: True for tile in previous.values() if tile is not None}
            visibility.update(
                {tile: False for tile in self.allocator.assignments.values()}
            )
            for tile, visible in visibility.items():
                for j in range(len(self.course["cells"][tile]["patches"])):
                    prim = stage.GetPrimAtPath(
                        f"/World/ground/terrain/Cell_{tile:02d}/material_{j}"
                    )
                    if prim:
                        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                            "inherited" if visible else "invisible"
                        )
            if 0 in ids:
                origin = self.env.scene.env_origins[0].detach().cpu().numpy()
                self.env.sim.set_camera_view(
                    origin + [-3, -4, 3], origin + [1, 0.5, 0.5]
                )
        ack = self.request({"reset": chosen})
        self.reset_events.append(
            dict(
                env_ids=ids,
                tiles=chosen,
                origins=np.array(origins).tolist(),
                other_worlds_unchanged=ack["other_worlds_unchanged"],
            )
        )

    def reset(self, env_ids):
        ids = [int(i) for i in env_ids]
        for i in ids:
            self.metrics.finish(
                [i], fell=bool(self.env.termination_manager.terminated[i])
            )
        reasons = {
            i: [
                name
                for name in self.env.termination_manager.active_terms
                if bool(self.env.termination_manager.get_term(name)[i])
            ]
            for i in ids
        }
        self.assign(ids)
        self.reset_events[-1]["termination_terms"] = reasons
        zeros = torch.zeros(
            (len(ids), len(self.robot.body_names), 3), device=self.env.device
        )
        self.robot.set_external_force_and_torque(
            zeros,
            zeros,
            env_ids=torch.tensor(ids, device=self.env.device),
            is_global=True,
        )
        self.exposure[ids] = 0
        self.material_force_current[ids] = 0
        self.material_force_accumulator[ids] = 0
        return self.original_reset(env_ids)

    def step(self, *args, **kwargs):
        roots = self.robot.data.root_state_w.detach().cpu().numpy()
        jp = self.robot.data.joint_pos.detach().cpu().numpy()
        jv = self.robot.data.joint_vel.detach().cpu().numpy()
        com = self.robot.data.body_com_pos_w.detach().cpu().numpy()
        batch = []
        for i in range(self.env.num_envs):
            origin = np.array(
                self.course["cells"][self.allocator.assignments[i]]["origin"]
            )
            state = roots[i].copy()
            state[:3] -= origin
            batch.append(
                dict(
                    env_id=i,
                    root_state=state.tolist(),
                    joint_names=list(self.robot.joint_names),
                    joint_pos=jp[i].tolist(),
                    joint_vel=jv[i].tolist(),
                    body_names=list(self.robot.body_names),
                    body_com_pos=(com[i] - origin).tolist(),
                    dt=self.env.physics_dt,
                )
            )
        payload = self.request(
            {"batch": batch, "render": self.config.get("visualize", False)}
        )
        self.physics_ticks += 1
        if "surfaces" in payload:
            if self.config.get("record_surfaces"):
                from surface_recording import save_surface_frame

                save_surface_frame(
                    Path(self.config["output"]) / "surfaces",
                    self.physics_ticks,
                    self.env.physics_dt,
                    payload["surfaces"],
                    self.allocator.assignments,
                    self.generations,
                )
            self.surface_updates += 1
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for i, meshes in payload["surfaces"].items():
                origin = np.array(
                    self.course["cells"][self.allocator.assignments[int(i)]]["origin"]
                )
                for item in meshes:
                    mesh = UsdGeom.Mesh.Define(
                        stage, f"/World/DryLive/env_{i}/{item['material']}"
                    )
                    mesh.CreateSubdivisionSchemeAttr("none")
                    mesh.CreatePointsAttr(
                        np.asarray(item["vertices"], dtype=np.float32)
                        + origin.astype(np.float32)
                    )
                    mesh.CreateFaceVertexIndicesAttr(item["indices"])
                    mesh.CreateFaceVertexCountsAttr([3] * (len(item["indices"]) // 3))
                    mesh.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[item["material"]])])
        forces = np.asarray(payload["wrenches"])
        foot = [
            i
            for i, n in enumerate(self.robot.body_names)
            if "ankle" in n or "foot" in n
        ]
        self.exposure += (
            np.linalg.norm(forces[:, foot, :3], axis=-1).max(axis=1) > 5
        ) * self.env.physics_dt
        wrench = torch.tensor(forces, device=self.env.device, dtype=torch.float32)
        self.material_force_accumulator = torch.maximum(
            self.material_force_accumulator,
            torch.linalg.vector_norm(wrench[:, :, :3], dim=-1),
        )
        self.robot.set_external_force_and_torque(
            wrench[:, :, :3], wrench[:, :, 3:], is_global=True
        )
        self.robot.write_data_to_sim()
        return self.original_step(*args, **kwargs)

    def compute(self, *args, **kwargs):
        self.original_compute(*args, **kwargs)
        self.material_force_current.copy_(self.material_force_accumulator)
        self.material_force_accumulator.zero_()
        command = self.command
        ref = (
            command.anchor_pos_w
            if hasattr(command, "anchor_pos_w")
            else command.motion_anchor_pos_w
        )
        root_error = (
            torch.linalg.vector_norm(command.robot_anchor_pos_w - ref, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        joint_error = (
            torch.sqrt(
                torch.mean((command.robot_joint_pos - command.joint_pos) ** 2, dim=-1)
            )
            .detach()
            .cpu()
            .numpy()
        )
        roots = self.robot.data.root_pos_w.detach().cpu().numpy()
        for i in range(self.env.num_envs):
            origin = self.course["cells"][self.allocator.assignments[i]]["origin"]
            local = roots[i] - origin
            bounds = self.course["cells"][self.allocator.assignments[i]].get(
                "bounds", [[-0.45, -0.7, 0], [2.45, 1.7, 0]]
            )
            outside = not (
                bounds[0][0] + 0.15 < local[0] < bounds[1][0] - 0.15
                and bounds[0][1] + 0.15 < local[1] < bounds[1][1] - 0.15
            )
            if outside:
                self.env.termination_manager._terminated_buf[i] = True
            self.metrics.record(
                i,
                float(root_error[i]),
                float(joint_error[i]),
                min(self.exposure[i], self.env.step_dt),
                self.env.step_dt,
            )
        self.exposure[:] = 0
        return (
            self.env.termination_manager._terminated_buf
            | self.env.termination_manager._truncated_buf
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.env.sim.step = self.original_step
        self.env._reset_idx = self.original_reset
        self.env.termination_manager.compute = self.original_compute
        try:
            self.stream.write(b'{"close":true}\n')
            self.stream.flush()
        except (BrokenPipeError, ConnectionError):
            pass
        self.stream.close()
        self.conn.close()
        self.worker.wait(timeout=30)
        self.log.close()
        Path(self.config["output"], "runtime.json").write_text(
            json.dumps(
                dict(
                    resets=self.reset_events,
                    episodes=self.metrics.receipts,
                    levels=self.metrics.scheduler.levels,
                    worker_exit_code=self.worker.returncode,
                    num_envs=self.env.num_envs,
                    surface_updates=self.surface_updates,
                ),
                indent=2,
            )
        )
