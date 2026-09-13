"""Check the exported replay with Isaac Sim's actual USD runtime."""

import json
from pathlib import Path
from isaaclab.app import AppLauncher
import filelock

with filelock.FileLock("/tmp/isaaclab_app_launcher.lock"):
    launcher = AppLauncher(
        headless=True, enable_cameras=False, device="cuda:0", kit_args="--no-window"
    )
app = launcher.app
try:
    from pxr import Usd, UsdGeom

    path = (
        Path(__file__).resolve().parent
        / "outputs/render-course-hires/isaac-replay.usdc"
    )
    stage = Usd.Stage.Open(str(path))
    assert stage and stage.GetEndTimeCode() > 0
    points = UsdGeom.Points(stage.GetPrimAtPath("/World/LiveMaterialReplay"))
    count = len(points.GetPointsAttr().Get(0))
    assert count == 80133
    receipt = {
        "status": "passed",
        "runtime": "Isaac Sim native USD",
        "particles": count,
        "end_time_code": stage.GetEndTimeCode(),
        "simulation_steps": 0,
    }
    (path.parent / "isaac-usd-validation.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(receipt, flush=True)
finally:
    app.close()
