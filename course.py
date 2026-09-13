"""Shared 16-cell terrain course; metres, Z up, named material patches."""

from itertools import combinations
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np

COLORS = {
    "ground": (0.40, 0.47, 0.50),
    "sand": (0.76, 0.59, 0.32),
    "mud": (0.28, 0.15, 0.09),
    "water": (0.08, 0.40, 0.69),
}
MATERIALS = {
    "sand": dict(density=1600.0, friction=0.75, viscosity=0.0, yield_stress=0.0),
    "mud": dict(
        density=1500.0,
        friction=0.0,
        viscosity=100.0,
        yield_stress=300.0,
        yield_pressure=1.0e10,
        tensile_yield_ratio=1.0,
    ),
    "water": dict(
        density=1000.0,
        friction=0.0,
        viscosity=0.0,
        yield_stress=0.0,
        yield_pressure=1.0e10,
        tensile_yield_ratio=1.0,
    ),
}


def make_course():
    kinds = ["ground", "sand", "mud", "water"]
    groups = [list(c) for n in range(1, 5) for c in combinations(kinds, n)] + [
        ["sand", "water"]
    ]
    cells = []
    for index, names in enumerate(groups):
        origin = [(index % 4) * 3.4, (index // 4) * 3.0, 0.0]
        patches = []
        for j, name in enumerate(names):
            lo = [0.20 + 2.2 * j / len(names), -0.7, -0.14]
            hi = [0.20 + 2.2 * (j + 1) / len(names), 1.7, 0.0]
            if index == 15:
                lo = [0.20, -0.7, -0.14 if j == 0 else -0.05]
                hi = [2.4, 1.7, -0.05 if j == 0 else 0.0]
            patches.append(dict(material=name, lo=lo, hi=hi))
        cells.append(
            dict(
                id=index,
                name="+".join(names) if index < 15 else "water-over-sand",
                origin=origin,
                patches=patches,
            )
        )
    walkways = []
    for k in range(3):
        walkways.append(
            dict(
                name=f"vertical_{k}",
                lo=[2.45 + 3.4 * k, -0.75, -0.14],
                hi=[2.95 + 3.4 * k, 10.75, 0.0],
            )
        )
        walkways.append(
            dict(
                name=f"horizontal_{k}",
                lo=[-0.45, 1.75 + 3 * k, -0.14],
                hi=[12.65, 2.25 + 3 * k, 0.0],
            )
        )
    return dict(
        schema="terrain_course_v2",
        walkways=walkways,
        up_axis="Z",
        units="metres",
        cells=cells,
        material_presets=MATERIALS,
        meaning="Adjacent material strips and one layered example; no saturation law",
    )


def solids(cell):
    # floor, starting pad, side/end retaining walls; all below or level with z=0.
    result = [
        ("bottom", [-0.45, -0.75, -0.20], [2.45, 1.75, -0.14], "ground"),
        ("start", [-0.45, -0.7, -0.14], [0.20, 1.7, 0.0], "ground"),
        ("left", [-0.45, -0.75, -0.14], [2.45, -0.70, 0.0], "ground"),
        ("right", [-0.45, 1.70, -0.14], [2.45, 1.75, 0.0], "ground"),
        ("end", [2.4, -0.7, -0.14], [2.45, 1.7, 0.0], "ground"),
        ("back", [-0.45, -0.7, -0.14], [-0.40, 1.7, 0.0], "ground"),
    ]
    result += [
        (f"patch_{i}", p["lo"], p["hi"], "ground")
        for i, p in enumerate(cell["patches"])
        if p["material"] == "ground"
    ]
    result += [(r["name"], r["lo"], r["hi"], "ground") for r in cell.get("rocks", [])]
    result += [(r["name"], r["lo"], r["hi"], "ground") for r in cell.get("aprons", [])]
    return result


def add_mujoco_solids(tree, course):
    world = tree.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("type") == "plane":
            world.remove(geom)
    for cell in course["cells"]:
        for name, lo, hi, mat in solids(cell):
            lo = np.array(lo) + cell["origin"]
            hi = np.array(hi) + cell["origin"]
            ET.SubElement(
                world,
                "geom",
                name=f"terrain_{cell['id']}_{name}",
                type="box",
                pos=" ".join(map(str, (lo + hi) / 2)),
                size=" ".join(map(str, (hi - lo) / 2)),
                rgba=" ".join(map(str, (*COLORS[mat], 1))),
                friction=".8 .005 .0001",
            )

    for w in course.get("walkways", []):
        lo = np.array(w["lo"])
        hi = np.array(w["hi"])
        ET.SubElement(
            world,
            "geom",
            name="walkway_" + w["name"],
            type="box",
            pos=" ".join(map(str, (lo + hi) / 2)),
            size=" ".join(map(str, (hi - lo) / 2)),
            rgba=".35 .42 .45 1",
            friction=".8 .005 .0001",
        )


def export(root, course=None):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

    root = Path(root)
    root.mkdir(exist_ok=True)
    course = make_course() if course is None else course
    (root / "course.json").write_text(json.dumps(course, indent=2) + "\n")
    stage = Usd.Stage.CreateNew(str(root / "course.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(
        9.81
    )
    material = UsdShade.Material.Define(stage, "/World/GroundPhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.8)
    physics_material.CreateDynamicFrictionAttr(0.8)
    physics_material.CreateRestitutionAttr(0.0)

    def bind_physics(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, materialPurpose="physics"
        )

    for cell in course["cells"]:
        base = f"/World/Cell_{cell['id']:02d}"
        UsdGeom.Xform.Define(stage, base).GetPrim().SetCustomDataByKey(
            "terrain:name", cell["name"]
        )
        for name, lo, hi, mat in solids(cell):
            lo = np.array(lo) + cell["origin"]
            hi = np.array(hi) + cell["origin"]
            cube = UsdGeom.Cube.Define(stage, base + "/" + name)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
            cube.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[mat])])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            bind_physics(cube.GetPrim())
        for i, p in enumerate(cell["patches"]):
            if p["material"] == "ground":
                continue
            lo = np.array(p["lo"]) + cell["origin"]
            hi = np.array(p["hi"]) + cell["origin"]
            pts = (
                np.array(
                    np.meshgrid(
                        *[np.arange(lo[k] + 0.025, hi[k], 0.05) for k in range(3)],
                        indexing="ij",
                    )
                )
                .reshape(3, -1)
                .T
            )
            cloud = UsdGeom.Points.Define(stage, base + f"/material_{i}")
            cloud.CreatePointsAttr(pts.astype(np.float32))
            cloud.CreateWidthsAttr([0.035] * len(pts))
            cloud.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[p["material"]])])
            cloud.GetPrim().SetCustomDataByKey("terrain:material", p["material"])
            cloud.GetPrim().SetCustomDataByKey(
                "terrain:role", "preview_only_attach_live_MPM_adapter"
            )
    for w in course.get("walkways", []):
        lo = np.array(w["lo"])
        hi = np.array(w["hi"])
        cube = UsdGeom.Cube.Define(stage, "/World/Walkway_" + w["name"])
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        cube.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.42, 0.45)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind_physics(cube.GetPrim())
    stage.GetRootLayer().Save()
    return course


if __name__ == "__main__":
    c = export(Path(__file__).resolve().parent / "assets")
    print("Exported", len(c["cells"]), "terrain cells")
