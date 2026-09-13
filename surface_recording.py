"""Numeric surface snapshots with episode identity; no pickle or policy assets."""

from pathlib import Path
import numpy as np


def save_surface_frame(output, tick, dt, surfaces, assignments, generations):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"time_s": np.asarray(tick * dt)}
    for key, meshes in surfaces.items():
        env_id = int(key)
        prefix = f"env{env_id}"
        arrays[prefix + "_tile"] = np.asarray(assignments[env_id])
        arrays[prefix + "_generation"] = np.asarray(generations[env_id])
        for mesh in meshes:
            material = mesh["material"]
            if material not in {"sand", "mud"}:
                raise ValueError("Unexpected material in dry surface recording")
            vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
            faces = np.asarray(mesh["indices"], dtype=np.int32).reshape(-1, 3)
            if not np.isfinite(vertices).all() or (
                faces.size and (faces.min() < 0 or faces.max() >= len(vertices))
            ):
                raise ValueError("Invalid reconstructed mesh")
            arrays[f"{prefix}_{material}_vertices"] = vertices
            arrays[f"{prefix}_{material}_faces"] = faces
    np.savez_compressed(output / f"frame-{tick:07d}.npz", **arrays)
