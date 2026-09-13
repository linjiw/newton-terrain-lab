"""Independent GPU check: solver density, mass and unconstrained gravity motion."""

import json
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from course import MATERIALS
from physics_audit import particle_radius


def run():
    rows = []
    for name, material in MATERIALS.items():
        for dt in [0.005, 0.0025]:
            with wp.ScopedDevice("cuda:0"):
                builder = newton.ModelBuilder()
                SolverImplicitMPM.register_custom_attributes(builder)
                spacing = np.array([0.02, 0.025, 0.0175])
                n = 8
                builder.add_particle_grid(
                    pos=wp.vec3(0, 0, 2),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0),
                    dim_x=n,
                    dim_y=n,
                    dim_z=n,
                    cell_x=float(spacing[0]),
                    cell_y=float(spacing[1]),
                    cell_z=float(spacing[2]),
                    mass=float(np.prod(spacing) * material["density"]),
                    radius_mean=particle_radius(spacing),
                    jitter=0,
                    custom_attributes={
                        "mpm:" + k: v for k, v in material.items() if k != "density"
                    },
                )
                model = builder.finalize()
                cfg = SolverImplicitMPM.Config(
                    voxel_size=0.04,
                    max_iterations=100,
                    tolerance=1e-5,
                    strain_basis="P0",
                    critical_fraction=0,
                )
                solver = SolverImplicitMPM(model, config=cfg)
                state = model.state()
                initial = state.particle_q.numpy().mean(axis=0)
                steps = round(0.1 / dt)
                for _ in range(steps):
                    solver.step(state, state, contacts=None, control=None, dt=dt)
                density = solver._mpm_model.particle_density.numpy()
                volume = solver._mpm_model.particle_volume.numpy()
                mass = model.particle_mass.numpy()
                vz = state.particle_qd.numpy()[:, 2].mean()
                dz = state.particle_q.numpy()[:, 2].mean() - initial[2]
                expected_dz = -9.81 * dt * dt * steps * (steps + 1) / 2
                row = dict(
                    material=name,
                    dt_s=dt,
                    effective_density_kg_m3=float(density.mean()),
                    volume_m3=float(volume.sum()),
                    mass_kg=float(mass.sum()),
                    vz_m_s=float(vz),
                    expected_vz_m_s=-0.981,
                    dz_m=float(dz),
                    expected_semiimplicit_dz_m=expected_dz,
                )
                row["passed"] = bool(
                    np.allclose(density, material["density"], rtol=1e-5)
                    and np.isclose(volume.sum(), n**3 * np.prod(spacing), rtol=1e-5)
                    and abs(vz + 0.981) < 0.002
                    and abs(dz - expected_dz) < 0.001
                )
                rows.append(row)
    out = Path(__file__).parent / "outputs/physics-audit"
    out.mkdir(exist_ok=True)
    (out / "gpu-basics.json").write_text(
        json.dumps(
            dict(
                checks=rows,
                passed=all(r["passed"] for r in rows),
                scope="Density/volume and unconstrained gravity only; no material response validation",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))
    assert all(r["passed"] for r in rows)


if __name__ == "__main__":
    run()
