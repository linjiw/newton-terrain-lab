"""Atomic progress receipts for long native SONIC terrain training runs."""

import json
import time
from pathlib import Path
import torch
from transformers import TrainerCallback


class DryTrainingProgress(TrainerCallback):
    def __init__(self, output, target_iterations=2000, rollout_steps=24):
        self.output = Path(output)
        self.target_iterations = int(target_iterations)
        self.rollout_steps = int(rollout_steps)
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        env = kwargs["env"]
        runtime = env.dry_terrain
        iteration = int(state.global_step)
        finite = bool(
            torch.isfinite(runtime.robot.data.root_state_w).all()
            and torch.isfinite(runtime.robot.data.joint_pos).all()
        )
        receipt = dict(
            iteration=iteration,
            target_iterations=self.target_iterations,
            num_envs=env.num_envs,
            transitions=iteration * env.num_envs * self.rollout_steps,
            elapsed_seconds=time.monotonic() - self.started,
            finite_robot_states=finite,
            curriculum_levels=list(runtime.metrics.scheduler.levels),
            completed_episodes=len(runtime.metrics.receipts),
            recent_episodes=runtime.metrics.receipts[-8:],
            status="completed" if iteration >= self.target_iterations else "running",
            updated_unix=time.time(),
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(self.output)
        if not finite:
            raise FloatingPointError("Nonfinite robot state in terrain training")
        return control
