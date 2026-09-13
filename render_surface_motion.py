"""Continuous surface reconstruction of measured MPM and humanoid motion."""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse
import json
import time
from pathlib import Path
import teacher_adapter  # noqa: F401
import numpy as np
import warp as wp
import mujoco
import trimesh
import pyrender
import imageio.v2 as imageio
from PIL import Image
from newton.geometry import ParticleSurface
from render_course import build_model, draw_overlay
from course import COLORS


def camera_pose(eye, target):
    z = np.asarray(eye) - target
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = eye
    return pose


def render(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = json.loads((folder / "result.json").read_text())
    rollout = np.load(folder / "rollout.npz")
    particles = np.load(folder / "particles.npz")
    model, course = build_model(result)
    data = mujoco.MjData(model)
    source = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    joint_map = [
        (source.jnt_qposadr[j], model.joint(source.joint(j).name).qposadr[0])
        for j in range(1, source.njnt)
    ]
    scene = pyrender.Scene(
        bg_color=[0.17, 0.22, 0.28, 1], ambient_light=[0.5, 0.5, 0.5]
    )
    nodes = []
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            va = model.mesh_vertadr[mid]
            vn = model.mesh_vertnum[mid]
            fa = model.mesh_faceadr[mid]
            fn = model.mesh_facenum[mid]
            mesh = trimesh.Trimesh(
                model.mesh_vert[va : va + vn],
                model.mesh_face[fa : fa + fn],
                process=False,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * model.geom_size[g])
        else:
            continue
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=model.geom_rgba[g], roughnessFactor=0.7
        )
        node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False))
        nodes.append((g, node))
    origin = np.array(course["cells"][result["cell"]]["origin"])
    pose = camera_pose(origin + [-2.4, -3.1, 2.4], origin + [0.9, 0.5, 0.35])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3), pose=pose)
    renderer = pyrender.OffscreenRenderer(1280, 720)
    h = float(particles["voxel_size"])
    masks = {
        name: np.all(np.isclose(particles["colors"], color, atol=1e-5), axis=1)
        for name, color in COLORS.items()
        if name != "ground"
    }
    # Rendering radius is an estimate for older recordings without stored radii.
    surfaces = {
        name: ParticleSurface(
            voxel_size=h * 0.35,
            kernel_radius=h * 1.4,
            threshold=0.25,
            field_smooth_iterations=1,
            device="cuda:0",
        )
        for name, mask in masks.items()
        if mask.any()
    }
    dynamic = []
    counts = []
    extraction_seconds = 0
    with imageio.get_writer(
        output / "motion.mp4", fps=25, codec="libx264", quality=8
    ) as video:
        for k in range(0, len(rollout["qpos"]), 2):
            q = rollout["qpos"][k]
            data.qpos[:7] = q[:7]
            for a, b in joint_map:
                data.qpos[b] = q[a]
            mujoco.mj_forward(model, data)
            for g, node in nodes:
                pose = np.eye(4)
                pose[:3, :3] = data.geom_xmat[g].reshape(3, 3)
                pose[:3, 3] = data.geom_xpos[g]
                scene.set_pose(node, pose)
            for node in dynamic:
                scene.remove_node(node)
            dynamic = []
            total = 0
            frame = max(
                0,
                min(
                    np.searchsorted(particles["time_s"], k * 0.02, side="right") - 1,
                    len(particles["positions"]) - 1,
                ),
            )
            for name, surface in surfaces.items():
                p = particles["positions"][frame, masks[name]]
                t = time.monotonic()
                with wp.ScopedDevice("cuda:0"):
                    v, f, _ = surface.extract(
                        wp.array(p, dtype=wp.vec3), wp.full(len(p), h / 4, dtype=float)
                    ).to_arrays()
                    if v is None:
                        continue
                    v, f = v.numpy(), f.numpy().reshape(-1, 3)
                extraction_seconds += time.monotonic() - t
                if not np.isfinite(v).all():
                    raise ValueError("Nonfinite surface")
                total += len(f)
                mesh = trimesh.Trimesh(v, f, process=False)
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[*COLORS[name], 1],
                    metallicFactor=0.1 if name == "water" else 0,
                    roughnessFactor=0.18 if name == "water" else 0.75,
                )
                dynamic.append(
                    scene.add(
                        pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
                    )
                )
            counts.append(total)
            pixels, _ = renderer.render(scene)
            pixels = draw_overlay(pixels, result, k * 0.02, course)
            video.append_data(pixels)
            if k == 2 * (len(rollout["qpos"]) // 4):
                Image.fromarray(pixels).save(output / "preview.png")
    renderer.delete()
    receipt = dict(
        source=str(folder.resolve()),
        frames=len(counts),
        fps=25,
        wall_seconds=time.monotonic() - started,
        surface_extraction_seconds=extraction_seconds,
        triangle_count_range=[min(counts), max(counts)],
        physics_stepped_during_render=False,
        radius_source="display estimate h/4",
        surface_voxel_m=h * 0.35,
        limitations=[
            "Surface smoothing is cosmetic",
            "Inactive tiles are static boxes",
            "No physical transparency/refraction model",
        ],
        stop_reason=result["stop_reason"],
        root_rmse_m=result["root_rmse_m"],
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.folder, a.output)
