from terrain_settings import setting

"""Prepare a fresh native Isaac Lab teacher run using the existing environment."""

import argparse
import json
from pathlib import Path
from pxr import Usd, UsdGeom, Gf
from course import make_course

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--cell", type=int, default=0)
p.add_argument("--steps", type=int, default=150)
p.add_argument("--mpm", action="store_true")
p.add_argument("--diagnostic-buffers", action="store_true")
p.add_argument("--reset-every", type=int, default=0)
p.add_argument("--voxel-size", type=float, default=0.08)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
stage = Usd.Stage.Open(str(ROOT / "assets/course.usda"))
stage.RemovePrim("/World/PhysicsScene")
if a.mpm:
    cell = make_course()["cells"][a.cell]
    for i in range(len(cell["patches"])):
        stage.RemovePrim(f"/World/Cell_{a.cell:02d}/material_{i}")
origin = make_course()["cells"][a.cell]["origin"]
UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp().Set(
    Gf.Vec3d(*[-v for v in origin])
)
stage.GetRootLayer().Export(str(a.output / "course.usda"))
c = dict(
    output=str(a.output / "run"),
    cell=a.cell,
    steps=a.steps,
    reset_every=a.reset_every,
    voxel_size=a.voxel_size,
    mpm=a.mpm,
    newton_python=setting("newton_python"),
    teacher_checkpoint=setting("checkpoint"),
)
(a.output / "config.json").write_text(json.dumps(c, indent=2))
base = json.loads(Path(setting("legacy_eval_command")).read_text())
replace = {
    "++eval_output_dir": str(a.output / "unused_metrics"),
    "++eval_base_dir": str(a.output / "hydra"),
    "++callbacks.im_eval._target_": "isaac_teacher_callback.TerrainTeacherCallback",
    "++callbacks.im_eval.stage_config": str(a.output / "config.json"),
    "++manager_env._target_": "native_terrain_config.TerrainTrackingEnvCfg"
    if a.diagnostic_buffers
    else "gear_sonic.envs.manager_env.modular_tracking_env_cfg.ModularTrackingEnvCfg",
    "++manager_env.config.scene_usd_path": str(a.output / "course.usda"),
}
command = [
    x.split("=", 1)[0] + "=" + replace[x.split("=", 1)[0]]
    if x.split("=", 1)[0] in replace
    else x
    for x in base
    if not x.startswith("++manager_env.config.navigation_task_path=")
]
command[1] = str(ROOT / "native_entry.py")
(a.output / "command.json").write_text(json.dumps(command, indent=2))
print(a.output)
