"""Evaluation options for the tested SONIC TRL fork; no archived run dependency."""

from pathlib import Path
from terrain_settings import setting


def evaluation_command():
    root = Path(__file__).resolve().parent
    return [
        setting("isaac_python"),
        str(root / "dry_entry.py"),
        "checkpoint=" + setting("checkpoint"),
        "++manager_env.commands.motion.motion_lib_cfg.motion_file="
        + setting("eval_motion_file"),
        *[
            "++headless=true",
            "++num_envs=1",
            "++seed=91260",
            "++use_wandb=false",
            "++use_encoder=g1",
            "++eval_callbacks=[im_eval]",
            "++run_eval_loop=false",
            "++trainer.schedule_dict=null",
            "++manager_env.config.terrain_type=scene_usd",
            "++manager_env.config.render_results=false",
            "++manager_env.config.render_ego=false",
            "++manager_env.commands.motion.debug_vis=false",
            "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=1",
            "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
            "++manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
            "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
            "++manager_env.commands.motion.cat_upper_body_poses=false",
            "++manager_env.commands.motion.freeze_frame_aug=false",
        ],
    ]
