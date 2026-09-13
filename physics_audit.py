"""Units/resolution audit. Passing these checks is not material calibration."""

import json
from pathlib import Path
import numpy as np
from course import make_course, MATERIALS


def particle_radius(spacing):
    """Newton 1.6 implicit MPM uses V=(2r)^3, not sphere volume."""
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Expected three positive finite sample spacings")
    return float(np.cbrt(np.prod(spacing)) / 2)


def audit(course, voxel_size, minimum_cells=4):
    """Four cells is a screening heuristic; convergence still must be measured."""
    rows = []
    for cell in course["cells"]:
        for patch in cell["patches"]:
            name = patch["material"]
            if name == "ground":
                continue
            size = np.array(patch["hi"]) - patch["lo"]
            dims = np.maximum(1, np.ceil(size / (voxel_size / 2)).astype(int))
            spacing = size / dims
            density = MATERIALS[name]["density"]
            rows.append(
                dict(
                    cell=cell["id"],
                    material=name,
                    particles=int(np.prod(dims)),
                    mass_kg=float(np.prod(size) * density),
                    minimum_cells=float(min(size) / voxel_size),
                    resolution_screen_pass=bool(
                        min(size) / voxel_size >= minimum_cells
                    ),
                    old_effective_density_kg_m3=float(
                        np.prod(spacing) * density / min(spacing) ** 3
                    ),
                    corrected_effective_density_kg_m3=float(
                        np.prod(spacing) * density / (2 * particle_radius(spacing)) ** 3
                    ),
                )
            )
    return dict(
        voxel_size_m=voxel_size,
        minimum_cells_screen=minimum_cells,
        total_particles=sum(r["particles"] for r in rows),
        patches=rows,
        training_ready=False,
        material_calibrated=False,
        pending=[
            "grid/time convergence",
            "hydrostatic pressure and buoyancy",
            "sand shear/repose",
            "mud rheometry",
            "articulated coupling energy",
            "isolated environment reset and batched training integration",
        ],
    )


if __name__ == "__main__":
    output = Path(__file__).parent / "outputs/physics-audit"
    output.mkdir(exist_ok=True)
    result = {str(h): audit(make_course(), h) for h in [0.1, 0.08, 0.02]}
    (output / "resolution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                h: dict(
                    particles=r["total_particles"],
                    unresolved=sum(
                        not p["resolution_screen_pass"] for p in r["patches"]
                    ),
                    maximum_old_density_ratio=max(
                        p["old_effective_density_kg_m3"]
                        / p["corrected_effective_density_kg_m3"]
                        for p in r["patches"]
                    ),
                )
                for h, r in result.items()
            },
            indent=2,
        )
    )
