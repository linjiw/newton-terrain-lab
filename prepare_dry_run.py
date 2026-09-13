from terrain_settings import setting

"""Prepare native evaluation and future SONIC PPO configuration without launch."""

import argparse
import json
import hashlib
from pathlib import Path
import yaml
from pxr import Usd

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--course-dir", type=Path, default=ROOT / "assets/dry-map")
p.add_argument("--num-envs", type=int, default=2)
p.add_argument("--surfaces", action="store_true")
p.add_argument("--initial-level", type=int, choices=range(4), default=1)
p.add_argument("--selective-reset-at", type=int, default=-1)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--voxel-size", type=float, default=0.04)
p.add_argument("--training-iterations", type=int, default=100)
p.add_argument("--ppo-epochs", type=int, default=1)
p.add_argument("--rollout-steps", type=int, default=24)
p.add_argument("--seed", type=int, default=23)
a = p.parse_args()
if (
    min(
        a.num_envs,
        a.steps,
        a.voxel_size,
        a.training_iterations,
        a.rollout_steps,
        a.ppo_epochs,
    )
    <= 0
):
    raise ValueError("Counts and voxel size must be positive")
from dry_course import TileAllocator

course = a.course_dir / "course.json"
TileAllocator(json.loads(course.read_text()), a.num_envs)
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
(out / "course.json").write_bytes(course.read_bytes())
course = out / "course.json"
stage = Usd.Stage.Open(str(a.course_dir / "course.usda"))
stage.RemovePrim("/World/PhysicsScene")
stage.GetRootLayer().Export(str(out / "course.usda"))
checkpoint = setting("checkpoint")
runtime = dict(
    visualize=a.surfaces,
    course=str(course),
    checkpoint=checkpoint,
    output=str(out / "run"),
    seed=a.seed,
    initial_level=a.initial_level,
    voxel_size=a.voxel_size,
    steps=a.steps,
    selective_reset_at=a.selective_reset_at,
    num_envs=a.num_envs,
)
(out / "runtime.json").write_text(json.dumps(runtime, indent=2))
(out / "train-runtime.json").write_text(
    json.dumps(
        {
            **runtime,
            "output": str(out / "train-runtime"),
            "initial_level": 0,
            "visualize": False,
        },
        indent=2,
    )
)
from sonic_command import evaluation_command

base = evaluation_command()
replace = {
    "++num_envs": str(a.num_envs),
    "++seed": str(a.seed),
    "++eval_output_dir": str(out / "metrics"),
    "++eval_base_dir": str(out / "hydra"),
    "++manager_env._target_": "dry_native_config.DryTrackingEnvCfg",
    "++manager_env.config.scene_usd_path": str(out / "course.usda"),
    "++callbacks.im_eval._target_": "dry_eval_callback.DryEvaluationCallback",
    "++callbacks.im_eval.stage_config": str(out / "runtime.json"),
}
cmd = [x for x in base if x.split("=", 1)[0] not in replace]
cmd.extend(key + "=" + value for key, value in replace.items())
cmd[1] = str(ROOT / "dry_entry.py")
(out / "eval-command.json").write_text(json.dumps(cmd, indent=2))
# Future training uses the user-selected 84-clip screened pool, not diagnostic motion 00265.
manifest = json.loads(Path(setting("dataset_manifest")).read_text())
for split in ["train", "reserved"]:
    if manifest[split]["count"] != len(manifest[split]["ids"]) or len(
        set(manifest[split]["ids"])
    ) != len(manifest[split]["ids"]):
        raise ValueError("Dataset count/IDs mismatch")
    if (
        hashlib.sha256(Path(manifest[split]["path"]).read_bytes()).hexdigest()
        != manifest[split]["sha256"]
    ):
        raise ValueError("Dataset hash mismatch")
assert not set(manifest["train"]["ids"]) & set(manifest["reserved"]["ids"])
cfg = yaml.safe_load(Path(checkpoint).with_name("config.yaml").read_text())
cfg.update(
    num_envs=a.num_envs,
    seed=a.seed,
    checkpoint=checkpoint,
    force_flat_terrain=False,
    use_wandb=False,
    experiment_dir=str(out / "training"),
    save_dir=str(out / "training/.hydra"),
    output_dir=str(out / "training/output"),
    project_name="sonic_dry_terrain",
    experiment_name="sand_mud_rocks",
    headless=True,
    eval_overrides=None,
)
cfg["trainer"]["_target_"] = "dry_trainer.DryPPOTrainer"
cfg["callbacks"].pop("hindsight", None)
cfg["algo"]["config"].pop("hindsight_run_dir", None)
cfg["manager_env"]["_target_"] = "dry_native_config.DryTrackingEnvCfg"
cfg["manager_env"]["config"].update(
    terrain_type="scene_usd",
    scene_usd_path=str(out / "course.usda"),
    render_results=False,
    render_ego=False,
)
mc = cfg["manager_env"]["commands"]["motion"]
mc["motion_lib_cfg"]["motion_file"] = manifest["train"]["path"]
mc["motion_lib_cfg"]["multi_thread"] = False
mc["motion_lib_cfg"]["override_num_motions_to_load"] = manifest["train"]["count"]
cfg["algo"]["config"]["num_mini_batches"] = 1
cfg["algo"]["config"]["num_steps_per_env"] = a.rollout_steps
cfg["algo"]["config"]["num_learning_iterations"] = a.training_iterations
cfg["algo"]["config"]["num_learning_epochs"] = a.ppo_epochs
cfg["algo"]["trl"]["output_dir"] = str(out / "training")
(out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
(out / "dataset.json").write_text(json.dumps(manifest, indent=2))
(out / "train-command.json").write_text(
    json.dumps(
        [cmd[0], cmd[1], "--config-path", str(out), "--config-name", "train"], indent=2
    )
)
print(out)
