"""Isaac Sim-compatible USD timeline with measured robot and surface animation."""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import numpy as np
import warp as wp
from newton.geometry import ParticleSurface
from pxr import Usd, UsdGeom, Gf
from course import COLORS

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("export_replay.py")),
        str(a.folder),
        "--output",
        str(a.output),
    ],
    check=True,
)
stage = Usd.Stage.Open(str(a.output))
stage.RemovePrim("/World/LiveMaterialReplay")
data = np.load(a.folder / "particles.npz")
h = float(data["voxel_size"])
counts = {}
for name, color in COLORS.items():
    if name == "ground":
        continue
    mask = np.all(np.isclose(data["colors"], color, atol=1e-5), axis=1)
    if not mask.any():
        continue
    mesh = UsdGeom.Mesh.Define(stage, "/World/SurfaceReplay/" + name)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.GetPrim().SetCustomDataByKey(
        "terrain:role", "reconstructed_measured_samples_not_live_physics"
    )
    surface = ParticleSurface(
        voxel_size=h * 0.35,
        kernel_radius=h * 1.4,
        threshold=0.25,
        field_smooth_iterations=1,
        device="cuda:0",
    )
    rows = []
    with wp.ScopedDevice("cuda:0"):
        radius = wp.full(int(mask.sum()), h / 4, dtype=float)
        for xyz, t in zip(data["positions"], data["time_s"]):
            v, f, _ = surface.extract(
                wp.array(xyz[mask], dtype=wp.vec3), radius
            ).to_arrays()
            if v is None:
                raise RuntimeError("Empty material surface")
            v, f = v.numpy(), f.numpy()
            assert np.isfinite(v).all()
            mesh.GetPointsAttr().Set(v, float(t * 25))
            mesh.GetFaceVertexCountsAttr().Set(
                np.full(len(f) // 3, 3, dtype=np.int32), float(t * 25)
            )
            mesh.GetFaceVertexIndicesAttr().Set(f, float(t * 25))
            rows.append(len(f) // 3)
    counts[name] = dict(
        frames=len(rows), triangles_min=min(rows), triangles_max=max(rows)
    )
stage.GetRootLayer().Save()
reopened = Usd.Stage.Open(str(a.output))
assert not reopened.GetPrimAtPath("/World/LiveMaterialReplay")
assert reopened.GetPrimAtPath("/World/SurfaceReplay")
a.output.with_suffix(".json").write_text(
    json.dumps(
        dict(
            materials=counts,
            physics="recorded timeline playback",
            normals="computed by USD renderer",
            display_radius_m=h / 4,
            training_ready=False,
        ),
        indent=2,
    )
    + "\n"
)
print(counts)
