"""Fully submerged fixed cube: measured MPM force versus rho*g*V."""

import json
import argparse
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from course import MATERIALS
from physics_audit import particle_radius


def run(h, dt, projection=False, ground_plane=False, seconds=1.5):
    with wp.ScopedDevice("cuda:0"):
        b = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0)
        body = b.add_body(
            xform=wp.transform((0, 0, 0.12), wp.quat_identity()),
            mass=1,
            is_kinematic=True,
        )
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        for pos, half in [
            ((0, 0, -0.03), (0.24, 0.24, 0.03)),
            ((-0.23, 0, 0.15), (0.03, 0.2, 0.15)),
            ((0.23, 0, 0.15), (0.03, 0.2, 0.15)),
            ((0, -0.23, 0.15), (0.2, 0.03, 0.15)),
            ((0, 0.23, 0.15), (0.2, 0.03, 0.15)),
        ]:
            b.add_shape_box(
                -1,
                xform=wp.transform(pos, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
            )
        if ground_plane:
            b.add_ground_plane(cfg=cfg)
        collider = b.finalize()
        b = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(b)
        spacing = h / 2
        axes = [np.arange(-0.2 + spacing / 2, 0.2, spacing)] * 2 + [
            np.arange(spacing / 2, 0.24, spacing)
        ]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        pts = pts[~np.all(abs(pts - [0, 0, 0.12]) < 0.05, axis=1)]
        mat = MATERIALS["water"]
        b.add_particles(
            pos=pts,
            vel=np.zeros_like(pts),
            mass=np.full(len(pts), 1000 * spacing**3),
            radius=np.full(len(pts), particle_radius([spacing] * 3)),
            custom_attributes={"mpm:" + k: v for k, v in mat.items() if k != "density"},
        )
        model = b.finalize()
        state = model.state()
        solver = SolverImplicitMPM(
            model,
            config=SolverImplicitMPM.Config(
                voxel_size=h,
                strain_basis="P0",
                critical_fraction=0,
                max_iterations=250,
                tolerance=1e-5,
            ),
        )
        solver.setup_collider(model=collider)
        state.body_q = wp.clone(collider.body_q)
        state.body_qd = wp.zeros(1, dtype=wp.spatial_vector)
        state.body_f = wp.zeros_like(state.body_qd)
        forces = []
        speeds = []
        for k in range(round(seconds / dt)):
            state.clear_forces()
            solver.step(state, state, contacts=None, control=None, dt=dt)
            if projection:
                solver.project_outside(state, state, dt)
            imp, _, ids = solver.collect_collider_impulses(state)
            ids = ids.numpy()
            imp = imp.numpy()
            mapping = solver.collider_body_index.numpy()
            good = (ids >= 0) & (ids < len(mapping))
            ids = ids[good]
            imp = imp[good]
            forces.append(float(imp[mapping[ids] == 0, 2].sum() / dt))
            speeds.append(float(np.sqrt(np.mean(state.particle_qd.numpy() ** 2))))
        tail = np.array(forces[-round(0.5 / dt) :])
        expected = 1000 * 9.81 * 0.1**3
        relative = abs(tail.mean() - expected) / expected
        return dict(
            voxel_m=h,
            dt_s=dt,
            particles=len(pts),
            mean_upward_force_N=float(tail.mean()),
            force_std_N=float(tail.std()),
            expected_archimedes_N=expected,
            relative_error=float(relative),
            final_velocity_component_rms_m_s=speeds[-1],
            force_check_passed=bool(relative < 0.15 and tail.std() < expected * 0.15),
            passed=bool(
                relative < 0.15
                and tail.std() < expected * 0.15
                and speeds[-1] < 0.1
                and state.particle_q.numpy()[:, 2].min() > -0.01
            ),
            forces_N=forces,
            final_z_min=float(state.particle_q.numpy()[:, 2].min()),
            particles_below_floor=int((state.particle_q.numpy()[:, 2] < -0.03).sum()),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--ground-plane", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.0025)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for h, dt in [(args.voxel_size, args.dt)]:
        row = run(h, dt, args.projection, args.ground_plane)
        rows.append(row)
        (out / "results.json").write_text(
            json.dumps(
                dict(
                    checks=rows,
                    training_ready=False,
                    projection=args.projection,
                    ground_plane=args.ground_plane,
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({k: v for k, v in row.items() if k != "forces_N"}), flush=True)
