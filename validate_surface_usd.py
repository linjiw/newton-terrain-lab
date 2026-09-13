"""Validate surface topology/time samples, optionally using native Isaac USD."""

import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("path", type=Path)
p.add_argument("--native", action="store_true")
a = p.parse_args()
app = None
if a.native:
    from isaaclab.app import AppLauncher
    import filelock

    with filelock.FileLock("/tmp/isaaclab_app_launcher.lock"):
        launcher = AppLauncher(
            headless=True, enable_cameras=False, device="cuda:0", kit_args="--no-window"
        )
    app = launcher.app
try:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(a.path))
    checks = []
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith(
            "/World/SurfaceReplay/"
        ) or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        times = mesh.GetPointsAttr().GetTimeSamples()
        assert len(times) > 1
        for t in [times[0], times[len(times) // 2], times[-1]]:
            v = np.asarray(mesh.GetPointsAttr().Get(t))
            f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(t))
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(t))
            assert (
                np.isfinite(v).all() and len(f) == counts.sum() and np.all(counts == 3)
            )
            assert f.min() >= 0 and f.max() < len(v)
        checks.append(dict(material=prim.GetName(), time_samples=len(times)))
    assert len(checks) == 2 and not stage.GetPrimAtPath("/World/LiveMaterialReplay")
    receipt = dict(
        passed=True,
        runtime="native Isaac Sim USD" if a.native else "standalone USD",
        surfaces=checks,
        physics_steps=0,
    )
    a.path.with_suffix(".validation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(receipt, flush=True)
finally:
    if app:
        app.close()
