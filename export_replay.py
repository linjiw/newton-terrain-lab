"""USD mesh/particle animation for Isaac Sim timeline playback of measured states."""

import argparse
import json
from pathlib import Path
import teacher_adapter  # noqa: F401 - native dependency bootstrap
import numpy as np
import mujoco
from pxr import Usd, UsdGeom, Gf
from render_course import build_model

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
root = Path(__file__).resolve().parent
result = json.loads((a.folder / "result.json").read_text())
record = np.load(a.folder / "rollout.npz")
source = mujoco.MjModel.from_xml_path(str(a.folder / "model.xml"))
model, course = build_model(result)
data = mujoco.MjData(model)
base = Usd.Stage.Open(str(root / "assets/course.usda"))
base.GetRootLayer().Export(str(a.output))
stage = Usd.Stage.Open(str(a.output))
stage.RemovePrim("/World/PhysicsScene")
# This is explicitly a replay, without dynamic rigid-body schemas.
stage.GetDefaultPrim().SetCustomDataByKey(
    "terrain:role", "measured_policy_replay_not_live_physics"
)
stage.SetTimeCodesPerSecond(25)
stage.SetFramesPerSecond(25)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode((len(record["qpos"]) - 1) * 0.5)
UsdGeom.Xform.Define(stage, "/World/TeacherRobot")
ops = {}
for g in range(model.ngeom):
    if model.geom_bodyid[g] == 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = int(model.geom_dataid[g])
    v = int(model.mesh_vertadr[mid])
    nv = int(model.mesh_vertnum[mid])
    f = int(model.mesh_faceadr[mid])
    nf = int(model.mesh_facenum[mid])
    mesh = UsdGeom.Mesh.Define(stage, f"/World/TeacherRobot/geom_{g}")
    mesh.CreatePointsAttr(model.mesh_vert[v : v + nv].astype(np.float32))
    mesh.CreateFaceVertexCountsAttr([3] * nf)
    mesh.CreateFaceVertexIndicesAttr(model.mesh_face[f : f + nf].flatten())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, model.geom_rgba[g, :3]))])
    ops[g] = mesh.AddTransformOp()
mapq = []
for j in range(1, source.njnt):
    dest = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, source.joint(j).name)
    if dest >= 0:
        mapq.append((int(source.jnt_qposadr[j]), int(model.jnt_qposadr[dest])))
for k in range(0, len(record["qpos"]), 2):
    q = record["qpos"][k]
    data.qpos[:7] = q[:7]
    for i, j in mapq:
        data.qpos[j] = q[i]
    mujoco.mj_forward(model, data)
    for g, op in ops.items():
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[g].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[g]
        op.Set(Gf.Matrix4d(transform.T.tolist()), k * 0.5)
if (a.folder / "particles.npz").exists():
    for cell in (
        course["cells"]
        if result.get("all_active")
        else [course["cells"][result["cell"]]]
    ):
        for i in range(len(cell["patches"])):
            stage.RemovePrim(f"/World/Cell_{cell['id']:02d}/material_{i}")
    f = np.load(a.folder / "particles.npz")
    cloud = UsdGeom.Points.Define(stage, "/World/LiveMaterialReplay")
    cloud.CreateWidthsAttr([0.044] * len(f["colors"]))
    cloud.CreateDisplayColorAttr(f["colors"])
    for positions, t in zip(f["positions"], f["time_s"]):
        cloud.GetPointsAttr().Set(positions.astype(np.float32), float(t * 25))
stage.GetRootLayer().Save()
assert Usd.Stage.Open(str(a.output)).GetEndTimeCode() > 0
print(
    json.dumps(
        {
            "output": str(a.output),
            "robot_meshes": len(ops),
            "frames": len(range(0, len(record["qpos"]), 2)),
            "physics": "recorded playback",
        }
    )
)
