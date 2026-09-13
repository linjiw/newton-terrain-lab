"""Render measured native dry rollouts and reconstructed surfaces, without physics."""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
import pyrender
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from course import COLORS, solids
from terrain_settings import setting


def camera_pose(eye, target):
    z = np.asarray(eye, dtype=float) - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    p = np.eye(4)
    p[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    p[:3, 3] = eye
    return p


def material(color, metallic=0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color[:3], 1], roughnessFactor=0.7, metallicFactor=metallic
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "folder", type=Path, help="Prepared run folder containing course.json and run/"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--model-label", default="SONIC / configured checkpoint")
    p.add_argument("--max-seconds", type=float, default=10)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    course = json.loads((a.folder / "course.json").read_text())
    roll = np.load(a.folder / "run/rollout.npz", allow_pickle=False)
    root_states, joints = roll["root_state"], roll["joint_pos"]
    if not 0 <= a.env < root_states.shape[1]:
        raise ValueError("Environment index out of range")
    source = Path(setting("robot_xml"))
    xml = ET.parse(source).getroot()
    xml.find("compiler").set("meshdir", str(source.parent / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    joint_map = [
        (i, model.joint(str(name)).qposadr[0])
        for i, name in enumerate(roll["joint_names"])
    ]
    scene = pyrender.Scene(
        bg_color=[0.035, 0.055, 0.075, 1], ambient_light=[0.38, 0.4, 0.44]
    )
    nodes = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        va, vn, fa, fn = (
            model.mesh_vertadr[mid],
            model.mesh_vertnum[mid],
            model.mesh_faceadr[mid],
            model.mesh_facenum[mid],
        )
        mesh = trimesh.Trimesh(
            model.mesh_vert[va : va + vn], model.mesh_face[fa : fa + fn], process=False
        )
        node = scene.add(
            pyrender.Mesh.from_trimesh(
                mesh, material=material(model.geom_rgba[g], 0.12), smooth=False
            )
        )
        nodes.append((g, node))
    camera = scene.add(
        pyrender.PerspectiveCamera(yfov=np.pi / 4),
        pose=camera_pose([-3, -4, 3], [0.8, 0.4, 0.4]),
    )
    scene.add(
        pyrender.DirectionalLight(color=[1, 0.94, 0.83], intensity=3),
        pose=camera_pose([-3, -4, 7], [0, 0, 0]),
    )
    scene.add(
        pyrender.DirectionalLight(color=[0.55, 0.72, 1], intensity=1.5),
        pose=camera_pose([3, 2, 4], [0, 0, 0]),
    )
    renderer = pyrender.OffscreenRenderer(a.width, a.height)
    snapshots = sorted((a.folder / "run/surfaces").glob("frame-*.npz"))
    if not snapshots:
        raise ValueError(
            "No surface snapshots; prepare this evaluation with --record-surfaces"
        )
    # Stored timestamps below are authoritative; filename ticks only determine ordering.
    snapshot_times = [
        float(np.load(f, allow_pickle=False)["time_s"]) for f in snapshots
    ]
    dynamic, terrain = [], []
    last_tile, last_surface = None, None
    used_tiles, frames, triangles = [], 0, []
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 18)
        small = ImageFont.truetype(font_path, 13)
    except OSError:
        font = small = ImageFont.load_default()
    fps = 1 / (2 * float(roll["control_dt"]))
    with imageio.get_writer(
        a.output / "motion.mp4",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as video:
        for k in range(
            0, min(len(root_states), int(a.max_seconds / float(roll["control_dt"]))), 2
        ):
            t = float(roll["time_s"][k])
            tile = int(roll["tile_ids"][k, a.env])
            generation = int(roll["generations"][k, a.env])
            cell = course["cells"][tile]
            origin = np.asarray(cell["origin"])
            data.qpos[:7] = root_states[k, a.env, :7]
            data.qpos[:3] -= origin
            for i, j in joint_map:
                data.qpos[j] = joints[k, a.env, i]
            mujoco.mj_forward(model, data)
            for g, node in nodes:
                pose = np.eye(4)
                pose[:3, :3] = data.geom_xmat[g].reshape(3, 3)
                pose[:3, 3] = data.geom_xpos[g]
                scene.set_pose(node, pose)
            if tile != last_tile:
                for node in terrain:
                    scene.remove_node(node)
                terrain = []
                for name, lo, hi, _ in solids(cell):
                    lo, hi = np.asarray(lo), np.asarray(hi)
                    color = (
                        [0.20, 0.25, 0.28]
                        if not name.startswith("rock")
                        else [0.29, 0.33, 0.35]
                    )
                    mesh = trimesh.creation.box(extents=hi - lo)
                    pose = np.eye(4)
                    pose[:3, 3] = (lo + hi) / 2
                    terrain.append(
                        scene.add(
                            pyrender.Mesh.from_trimesh(mesh, material=material(color)),
                            pose=pose,
                        )
                    )
                last_tile = tile
                used_tiles.append(tile)
            index = np.searchsorted(snapshot_times, t, side="right") - 1
            surface_key = (index, generation, tile)
            if surface_key != last_surface:
                for node in dynamic:
                    scene.remove_node(node)
                dynamic = []
                if index >= 0:
                    with np.load(snapshots[index], allow_pickle=False) as snap:
                        prefix = f"env{a.env}"
                        if (
                            int(snap[prefix + "_generation"]) == generation
                            and int(snap[prefix + "_tile"]) == tile
                        ):
                            for name in ["sand", "mud"]:
                                key = f"{prefix}_{name}"
                                if key + "_vertices" not in snap:
                                    continue
                                v, f = snap[key + "_vertices"], snap[key + "_faces"]
                                if not len(f):
                                    continue
                                mesh = trimesh.Trimesh(v, f, process=False)
                                dynamic.append(
                                    scene.add(
                                        pyrender.Mesh.from_trimesh(
                                            mesh,
                                            material=material(COLORS[name]),
                                            smooth=True,
                                        )
                                    )
                                )
                                triangles.append(len(f))
                last_surface = surface_key
            angle = -2.05 + 0.30 * np.sin(t * 0.3)
            eye = np.array([0.8 + 4.5 * np.cos(angle), 0.5 + 4.5 * np.sin(angle), 2.6])
            scene.set_pose(camera, camera_pose(eye, [0.9, 0.5, 0.4]))
            pixels, _ = renderer.render(scene)
            im = Image.fromarray(pixels)
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, a.width, 69), fill=(10, 17, 23))
            draw.text(
                (24, 12),
                "NEWTON TERRAIN LAB   /   " + cell["name"].upper(),
                font=font,
                fill=(239, 244, 244),
            )
            draw.text(
                (24, 39),
                a.model_label + "   |   Isaac Lab + Newton MPM   |   measured replay",
                font=small,
                fill=(151, 174, 184),
            )
            draw.rectangle((0, a.height - 37, a.width, a.height), fill=(10, 17, 23))
            text = f"{t:05.2f} s  /  env {a.env}  /  tile {tile:03d}  /  episode {generation}     •     4 cm grid  ·  uncalibrated soil"
            draw.text((24, a.height - 27), text, font=small, fill=(195, 213, 218))
            video.append_data(np.asarray(im))
            if frames in [15, 60]:
                im.save(a.output / "poster.jpg", quality=92)
            frames += 1
    renderer.delete()
    receipt = dict(
        frames=frames,
        fps=fps,
        duration_s=frames / fps,
        env_id=a.env,
        tiles=used_tiles,
        model_label=a.model_label,
        checkpoint_sha256=hashlib.sha256(
            Path(
                json.loads((a.folder / "runtime.json").read_text())["checkpoint"]
            ).read_bytes()
        ).hexdigest(),
        rollout_sha256=hashlib.sha256(
            (a.folder / "run/rollout.npz").read_bytes()
        ).hexdigest(),
        surface_snapshots=len(snapshots),
        triangle_count_range=[min(triangles), max(triangles)] if triangles else [0, 0],
        wall_seconds=time.monotonic() - started,
        physics_stepped_during_render=False,
        robot_pose_source="measured native root and joint states",
        surface_source="measured Newton ParticleSurface at 10 Hz; latest sample within same episode",
        physics_surface_lag_s=0.005,
        resets_visible=True,
    )
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
