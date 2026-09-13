from terrain_settings import setting

"""Render recorded physics poses, never substitute the commanded reference."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import teacher_adapter  # noqa: F401 - native dependency bootstrap
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from course import make_course, add_mujoco_solids, COLORS

ROBOT = Path(setting("robot_xml"))


def build_model(result):
    tree = ET.parse(ROBOT).getroot()
    tree.find("compiler").set("meshdir", str(ROBOT.parent / "meshes"))
    course = make_course()
    add_mujoco_solids(tree, course)
    world = tree.find("worldbody")
    asset = tree.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="terrain_sky",
        type="skybox",
        builtin="gradient",
        rgb1=".12 .18 .25",
        rgb2=".55 .64 .72",
        width="512",
        height="3072",
    )
    visual = tree.find("visual")
    if visual is None:
        visual = ET.SubElement(tree, "visual")
    existing = visual.find("global")
    if existing is not None:
        visual.remove(existing)
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".75 .75 .75")
    ET.SubElement(world, "light", pos="5 4 12", dir="0 0 -1", diffuse=".8 .8 .8")
    for cell in course["cells"]:
        if result.get("live_mpm") and (
            result.get("all_active") or cell["id"] == result["cell"]
        ):
            continue
        for i, p in enumerate(cell["patches"]):
            if p["material"] == "ground":
                continue
            lo = np.array(p["lo"]) + cell["origin"]
            hi = np.array(p["hi"]) + cell["origin"]
            ET.SubElement(
                world,
                "geom",
                name=f"preview_{cell['id']}_{i}",
                type="box",
                pos=" ".join(map(str, (lo + hi) / 2)),
                size=" ".join(map(str, (hi - lo) / 2)),
                rgba=" ".join(map(str, (*COLORS[p["material"]], 1))),
                contype="0",
                conaffinity="0",
            )
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode")), course


def visible_particles(positions, colors, origin, budget=14000):
    positions = np.asarray(positions)
    center = np.asarray(origin[:2]) + [1.0, 0.5]
    near = np.all(abs(positions[:, :2] - center) < [1.5, 1.3], axis=1)
    primary = np.flatnonzero(near)
    other = np.flatnonzero(~near)
    if len(primary) > budget:
        primary = primary[np.linspace(0, len(primary) - 1, budget, dtype=int)]
    remaining = budget - len(primary)
    if len(other) > remaining:
        other = other[np.linspace(0, len(other) - 1, remaining, dtype=int)]
    ids = np.r_[primary, other]
    return positions[ids], colors[ids]


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def draw_overlay(frame, result, t, course):
    t += result.get("record_start_s", 0.0)
    host_name = (
        "Isaac Lab / PhysX"
        if result.get("backend", "").startswith("isaaclab")
        else "MuJoCo"
    )
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1280, 92), fill=(15, 23, 32))
    d.text(
        (24, 12), "MOTION2SCENE  /  HUMANOID TERRAIN LAB", font=font(25), fill="white"
    )
    d.text(
        (24, 50),
        f"{result['cell_name'].upper()}  |  measured teacher execution  |  t = {t:.2f} s",
        font=font(19),
        fill=(183, 216, 232),
    )
    d.rectangle((0, 669, 1280, 720), fill=(15, 23, 32))
    d.text(
        (24, 681),
        f"{host_name} robot + {'live Newton MPM' if result['live_mpm'] else 'rigid ground'}  |  {result['stop_reason']}  |  root RMSE {result['root_rmse_m']:.3f} m",
        font=font(18),
        fill="white",
    )
    # Course map; inactive cells are scene previews, explicitly labelled.
    x0 = 1010
    d.rectangle((994, 98, 1260, 392), fill=(22, 31, 42))
    d.text((1010, 107), "16-CELL COURSE", font=font(16), fill="white")
    for c in course["cells"]:
        x = x0 + (c["id"] % 4) * 58
        y = 140 + (c["id"] // 4) * 53
        for j, p in enumerate(c["patches"]):
            n = len(c["patches"])
            color = tuple(int(255 * v) for v in COLORS[p["material"]])
            d.rectangle((x + j * 50 / n, y, x + (j + 1) * 50 / n, y + 42), fill=color)
        d.text((x + 4, y + 3), str(c["id"]), font=font(14), fill="white")
        if c["id"] == result["cell"]:
            d.rectangle((x - 2, y - 2, x + 52, y + 44), outline="white", width=3)
    d.text(
        (1010, 359),
        "All cells: live MPM"
        if result.get("all_active")
        else "Other cells: static preview",
        font=font(13),
        fill=(196, 206, 215),
    )
    return np.asarray(im)


def render(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((folder / "result.json").read_text())
    record = np.load(folder / "rollout.npz")
    source = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    model, course = build_model(result)
    data = mujoco.MjData(model)
    joint_map = []
    for j in range(1, source.njnt):
        name = source.joint(j).name
        dest = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if dest >= 0:
            joint_map.append((int(source.jnt_qposadr[j]), int(model.jnt_qposadr[dest])))
    particles = (
        np.load(folder / "particles.npz")
        if (folder / "particles.npz").exists()
        else None
    )
    renderer = mujoco.Renderer(model, height=720, width=1280, max_geom=16000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    origin = np.array(course["cells"][result["cell"]]["origin"])
    camera.azimuth = 135
    camera.elevation = -22
    camera.distance = 4.8
    indices = np.arange(0, len(record["qpos"]), 2)
    with imageio.get_writer(
        output / "motion.mp4", fps=25, codec="libx264", quality=8
    ) as video:
        for k in indices:
            q = record["qpos"][k]
            data.qpos[:7] = q[:7]
            for a, b in joint_map:
                data.qpos[b] = q[a]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = origin + [0.85, 0.5, 0.55]
            renderer.update_scene(data, camera)
            if particles is not None:
                frame = min(
                    int(
                        np.searchsorted(particles["time_s"], k * 0.02, side="right") - 1
                    ),
                    len(particles["positions"]) - 1,
                )
                for xyz, color in zip(
                    *visible_particles(
                        particles["positions"][max(frame, 0)],
                        particles["colors"],
                        origin,
                    )
                ):
                    if renderer.scene.ngeom >= renderer.scene.maxgeom:
                        break
                    g = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.022] * 3),
                        xyz,
                        np.eye(3).flatten(),
                        np.r_[color, 1.0],
                    )
                    renderer.scene.ngeom += 1
            pixels = renderer.render()
            overlay = draw_overlay(pixels, result, k * 0.02, course)
            video.append_data(overlay)
            if k == indices[len(indices) // 2]:
                Image.fromarray(overlay).save(output / "preview.png")
    # Whole-course overview, with the final measured robot pose.
    camera.lookat[:] = [5.6, 4.5, 0]
    camera.distance = 20.0
    camera.azimuth = 120
    camera.elevation = -65
    renderer.update_scene(data, camera)
    if particles is not None:
        for xyz, color in zip(
            *visible_particles(particles["positions"][-1], particles["colors"], origin)
        ):
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                renderer.scene.geoms[renderer.scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.022] * 3),
                xyz,
                np.eye(3).flatten(),
                np.r_[color, 1.0],
            )
            renderer.scene.ngeom += 1
    Image.fromarray(renderer.render()).save(output / "course-overview.png")
    renderer.close()
    (output / "render-receipt.json").write_text(
        json.dumps(
            dict(
                source=str(folder),
                pose_source="measured rollout qpos",
                static_preview_cells=[]
                if result.get("all_active")
                else [i for i in range(16) if i != result["cell"]],
                fps=25,
                frames=len(indices),
                physics_stepped_during_render=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.folder, a.output)
