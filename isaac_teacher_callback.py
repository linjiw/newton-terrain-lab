"""Native Isaac Lab teacher loop with optional external Newton MPM feedback."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from isaac_terrain_curriculum import TerrainCurriculumState


class TerrainTeacherCallback:
    def __init__(self, stage_config, **kwargs):
        self.config = json.loads(Path(stage_config).read_text())

    def on_step_end(self, args, state, control, **kwargs):
        c = self.config
        out = Path(c["output"])
        out.mkdir(parents=True, exist_ok=False)
        env = kwargs["env"]
        teacher = kwargs["model"].policy
        teacher.eval()
        teacher.eval_mode()
        env.set_is_evaluating(True)
        observation = env.reset_all()
        teacher.init_rollout()
        robot = env.env.scene["robot"]
        command = env.motion_command
        manager = env.env.termination_manager
        original_compute = manager.compute
        original_step = env.env.sim.step
        worker = None
        stream = None
        conn = None
        measured = {}
        trace = []
        qposes = []
        body_poses = []
        reset_events = []
        curriculum = TerrainCurriculumState(1)
        env.env.terrain_curriculum_state = curriculum
        exposure = [0.0]

        def compute(*a, **kw):
            original_compute(*a, **kw)
            measured["root"] = (
                command.robot_anchor_pos_w[0].detach().cpu().numpy().copy()
            )
            measured["reference"] = (
                command.anchor_pos_w[0].detach().cpu().numpy().copy()
                if hasattr(command, "anchor_pos_w")
                else command.motion_anchor_pos_w[0].detach().cpu().numpy().copy()
            )
            measured["joint_rmse"] = float(
                torch.sqrt(
                    torch.mean((command.robot_joint_pos[0] - command.joint_pos[0]) ** 2)
                ).item()
            )
            manager._terminated_buf.zero_()
            manager._truncated_buf.zero_()
            return manager._terminated_buf | manager._truncated_buf

        manager.compute = compute
        if c["mpm"]:
            sock = str(out / "mpm.sock")
            log = (out / "mpm-worker.log").open("x")
            worker = subprocess.Popen(
                [
                    c["newton_python"],
                    str(Path(__file__).with_name("mpm_server.py")),
                    "--socket",
                    sock,
                    "--cell",
                    str(c["cell"]),
                    "--voxel-size",
                    str(c.get("voxel_size", 0.08)),
                    "--output",
                    str(out),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "2"},
            )
            deadline = time.monotonic() + 90
            while not Path(sock).exists():
                if worker.poll() is not None:
                    raise RuntimeError("MPM worker failed to start")
                if time.monotonic() > deadline:
                    raise TimeoutError("MPM startup")
                time.sleep(0.1)
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(180)
            conn.connect(sock)
            stream = conn.makefile("rwb")

            def sim_step(*a, **kw):
                # Native scene.update refreshes data once per physics substep.
                message = dict(
                    root_state=robot.data.root_state_w[0].detach().cpu().tolist(),
                    joint_names=list(robot.joint_names),
                    joint_pos=robot.data.joint_pos[0].detach().cpu().tolist(),
                    joint_vel=robot.data.joint_vel[0].detach().cpu().tolist(),
                    body_names=list(robot.body_names),
                    body_com_pos=robot.data.body_com_pos_w[0].detach().cpu().tolist(),
                    dt=0.005,
                )
                stream.write((json.dumps(message) + "\n").encode())
                stream.flush()
                response = stream.readline()
                if not response:
                    raise RuntimeError("MPM worker disconnected")
                payload = json.loads(response)
                if "particle_positions" in payload:
                    from pxr import UsdGeom
                    import omni.usd

                    cloud = UsdGeom.Points.Define(
                        omni.usd.get_context().get_stage(),
                        "/World/LiveTerrainParticles",
                    )
                    cloud.CreatePointsAttr(
                        np.asarray(payload["particle_positions"], dtype=np.float32)
                    )
                    cloud.CreateWidthsAttr([0.04] * len(payload["particle_positions"]))
                    cloud.CreateDisplayColorAttr(
                        np.asarray(payload["particle_colors"], dtype=np.float32)
                    )
                wrench = torch.tensor(
                    payload["wrenches"], device=env.device, dtype=torch.float32
                )[None]
                feet = [
                    i
                    for i, name in enumerate(robot.body_names)
                    if "ankle" in name or "foot" in name
                ]
                if (
                    feet
                    and torch.linalg.vector_norm(wrench[0, feet, :3], dim=-1)
                    .max()
                    .item()
                    > 5.0
                ):
                    exposure[0] += 0.005
                robot.set_external_force_and_torque(
                    wrench[:, :, :3], wrench[:, :, 3:], is_global=True
                )
                robot.write_data_to_sim()
                return original_step(*a, **kw)

            env.env.sim.step = sim_step
        reason = "horizon"
        try:
            with torch.no_grad():
                for tick in range(c["steps"]):
                    if tick and c.get("reset_every") and tick % c["reset_every"] == 0:
                        curriculum.finish([0])
                        robot.set_external_force_and_torque(
                            torch.zeros(
                                (1, len(robot.body_names), 3), device=env.device
                            ),
                            torch.zeros(
                                (1, len(robot.body_names), 3), device=env.device
                            ),
                            is_global=True,
                        )
                        robot.write_data_to_sim()
                        observation = env.reset_all()
                        teacher.init_rollout()
                        if stream:
                            stream.write(b'{"reset":true}\n')
                            stream.flush()
                            acknowledgement = json.loads(stream.readline())
                            if acknowledgement.get("reset") is not True:
                                raise RuntimeError("Material reset not acknowledged")
                        reset_events.append(
                            {"tick": tick, "robot_reference_policy_reset": True}
                        )

                    last = env._motion_lib.get_time_step_total(command.motion_ids) - 2
                    command.time_steps.copy_(
                        torch.minimum(
                            command.time_steps, last - command.motion_start_time_steps
                        )
                    )
                    exposure[0] = 0.0
                    action = teacher.act_inference(
                        obs_dict=observation, skip_episode_attnmask=True
                    )
                    observation, _, done, _ = env.step({"actions": action})
                    if done.any():
                        raise RuntimeError("Unexpected native reset")
                    root = robot.data.root_state_w[0].detach().cpu().numpy()
                    trace.append(
                        dict(
                            time_s=(tick + 1) * 0.02,
                            root_height_m=float(root[2]),
                            root_error_m=float(
                                np.linalg.norm(measured["root"] - measured["reference"])
                            ),
                        )
                    )
                    curriculum.record(
                        0,
                        trace[-1]["root_error_m"],
                        measured["joint_rmse"],
                        exposure[0],
                        0.02,
                    )
                    qposes.append(
                        np.r_[root[:7], robot.data.joint_pos[0].detach().cpu().numpy()]
                    )
                    body_poses.append(
                        robot.data.body_state_w[0, :, :7].detach().cpu().numpy()
                    )
                    if root[2] < 0.3:
                        reason = "fall"
                        break
            curriculum.finish([0], fell=reason == "fall")
            (out / "curriculum-metrics.json").write_text(
                json.dumps(curriculum.receipts, indent=2)
            )
            np.savez_compressed(
                out / "native-rollout.npz",
                qpos=qposes,
                joint_names=list(robot.joint_names),
                body_poses=body_poses,
                body_names=list(robot.body_names),
                control_dt=0.02,
            )
            result = dict(
                status="executed",
                backend="isaaclab_physx+external_newton_mpm"
                if c["mpm"]
                else "isaaclab_physx",
                steps=len(trace),
                reset_events=reset_events,
                simulated_seconds=len(trace) * 0.02,
                stop_reason=reason,
                root_rmse_m=float(
                    np.sqrt(np.mean([r["root_error_m"] ** 2 for r in trace]))
                ),
                teacher_checkpoint=c["teacher_checkpoint"],
                cell=c["cell"],
            )
            (out / "result.json").write_text(json.dumps(result, indent=2))
            (out / "metrics.json").write_text(json.dumps(trace, indent=2))
        finally:
            manager.compute = original_compute
            env.env.sim.step = original_step
            if stream:
                try:
                    stream.write(b'{"close":true}\n')
                    stream.flush()
                except (BrokenPipeError, ConnectionError):
                    pass  # Preserve the original worker failure.
                finally:
                    try:
                        stream.close()
                    except (BrokenPipeError, ConnectionError):
                        pass
                    conn.close()
            if worker:
                worker.wait(timeout=30)
                log.close()
        if control is not None:
            control.should_training_stop = True
        # Match existing standalone evaluation callbacks' process lifecycle.
        return
