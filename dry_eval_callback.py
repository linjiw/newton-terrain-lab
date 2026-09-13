"""Native batched teacher test with selective reset and real termination handling."""

import json
from pathlib import Path
import torch
import numpy as np


class DryEvaluationCallback:
    def __init__(self, stage_config, **kwargs):
        self.config = json.loads(Path(stage_config).read_text())

    def on_step_end(self, args, state, control, **kwargs):
        wrapper = kwargs["env"]
        policy = kwargs["model"].policy
        runtime = wrapper.dry_terrain
        policy.eval()
        policy.eval_mode()
        wrapper.set_is_evaluating(True)
        obs = wrapper.reset_all()
        policy.init_rollout()
        dones = torch.zeros(wrapper.num_envs, device=wrapper.device, dtype=torch.bool)
        actions = []
        poses = []
        joint_poses = []
        tile_ids = []
        generations = []
        done_rows = []
        rewards = []
        isolation = []
        with torch.no_grad():
            for tick in range(self.config["steps"]):
                if tick == self.config.get("selective_reset_at", -1):
                    before = runtime.robot.data.root_state_w[1:].clone()
                    raw, _ = wrapper.env.reset(
                        env_ids=torch.tensor([0], device=wrapper.device)
                    )
                    obs = wrapper.process_raw_obs(raw, True)
                    dones[0] = True
                    unchanged = bool(
                        torch.equal(before, runtime.robot.data.root_state_w[1:])
                    )
                    isolation.append(
                        dict(tick=tick, other_robot_states_unchanged=unchanged)
                    )
                    if not unchanged:
                        raise RuntimeError("Selective reset moved another robot")
                action = policy.act_inference(
                    obs_dict=obs, cur_dones=dones, skip_episode_attnmask=True
                )
                obs, reward, dones, _ = wrapper.step({"actions": action})
                if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
                    raise ValueError("Invalid training transition")
                if (
                    not torch.isfinite(runtime.robot.data.root_state_w).all()
                    or not torch.isfinite(runtime.robot.data.joint_pos).all()
                ):
                    raise ValueError("Nonfinite measured robot pose")
                actions.append(action.cpu().numpy())
                rewards.append(reward.cpu().numpy())
                poses.append(runtime.robot.data.root_state_w.cpu().numpy())
                joint_poses.append(runtime.robot.data.joint_pos.cpu().numpy())
                tile_ids.append(
                    [runtime.allocator.assignments[i] for i in range(wrapper.num_envs)]
                )
                generations.append(runtime.generations.copy())
                done_rows.append(dones.cpu().numpy())
        if self.config.get("visualize") and runtime.surface_updates == 0:
            raise RuntimeError("No live surfaces produced")
        out = Path(self.config["output"])
        np.savez_compressed(
            out / "rollout.npz",
            actions=actions,
            root_state=poses,
            rewards=rewards,
            dones=done_rows,
            joint_pos=joint_poses,
            joint_names=np.asarray(runtime.robot.joint_names),
            tile_ids=tile_ids,
            generations=generations,
            time_s=(np.arange(len(poses)) + 1) * wrapper.env.step_dt,
            control_dt=wrapper.env.step_dt,
        )
        (out / "evaluation.json").write_text(
            json.dumps(
                dict(
                    num_envs=wrapper.num_envs,
                    steps=len(actions),
                    finite=True,
                    selective_reset_checks=isolation,
                    automatic_done_count=int(np.asarray(done_rows).sum()),
                    action_shape=list(np.asarray(actions).shape),
                    reward_shape=list(np.asarray(rewards).shape),
                ),
                indent=2,
            )
        )
        runtime.close()
        if control is not None:
            control.should_training_stop = True
