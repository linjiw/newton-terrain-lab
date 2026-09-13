"""Exploratory terrain probes using Newton 1.6's two-way MPM example.

The upstream example supplies the lagged-impulse coupling implementation.
Material constants are demo starting points, not measured soil parameters.
"""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import warp as wp
import newton
from physics_audit import particle_radius
import newton.examples
from newton.examples.mpm.example_mpm_twoway_coupling import Example as CoupledExample

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
COLORS = {"sand": (0.7, 0.6, 0.4), "mud": (0.3, 0.18, 0.1), "water": (0.1, 0.4, 0.9)}


class TerrainProbe(CoupledExample):
    def __init__(self, viewer, args):
        self.kind = args.material
        self.ranges = []
        self.rows = []
        self.args = args
        self.started = time.monotonic()
        if self.kind == "ground":
            self.viewer = viewer
            self.fps = 100
            self.frame_dt = 0.01
            self.sim_dt = 0.0025
            self.sim_time = 0.0
            b = newton.ModelBuilder()
            self._emit_rigid_bodies(b)
            b.add_ground_plane()
            self.model = b.finalize()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_contacts=False
            )
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
            newton.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, self.state_0
            )
            viewer.set_model(self.model)
        else:
            super().__init__(viewer, args)
            colors = np.zeros((self.sand_model.particle_count, 3), np.float32)
            for lo, hi, name in self.ranges:
                colors[lo:hi] = COLORS[name]
            self.particle_render_colors.assign(colors)
        self.record()

    def capture(self):
        # Keep initialization free of simulated steps; inspectable eager coupling first.
        self.graph = None

    def _emit_rigid_bodies(self, builder):
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 0.50), wp.quat_identity()), label="foot_probe"
        )
        cfg = newton.ModelBuilder.ShapeConfig(
            density=5.0 / (0.24 * 0.12 * 0.06), mu=0.5
        )
        builder.add_shape_box(body, hx=0.12, hy=0.06, hz=0.03, cfg=cfg)
        # A shallow tray confines fluids; identical rigid geometry in every condition.
        for p, h in [
            ((-0.42, 0.0, 0.16), (0.02, 0.44, 0.16)),
            ((0.42, 0.0, 0.16), (0.02, 0.44, 0.16)),
            ((0.0, -0.42, 0.16), (0.4, 0.02, 0.16)),
            ((0.0, 0.42, 0.16), (0.4, 0.02, 0.16)),
        ]:
            builder.add_shape_box(
                -1, xform=wp.transform(p, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2]
            )

    def _emit_particles(self, builder, voxel_size):
        names = ["sand", "mud", "water"] if self.kind == "mixed" else [self.kind]
        spacing = voxel_size / 2.0
        for i, name in enumerate(names):
            lo = np.array([-0.375 + 0.75 * i / len(names), -0.375, 0.025])
            hi = np.array([-0.375 + 0.75 * (i + 1) / len(names), 0.375, 0.225])
            dims = np.maximum(1, np.floor((hi - lo) / spacing).astype(int))
            cell = (hi - lo) / dims
            mat = MATERIALS[name]
            start = len(builder.particle_q)
            builder.add_particle_grid(
                pos=wp.vec3(lo + cell / 2),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=int(dims[0]),
                dim_y=int(dims[1]),
                dim_z=int(dims[2]),
                cell_x=float(cell[0]),
                cell_y=float(cell[1]),
                cell_z=float(cell[2]),
                mass=float(np.prod(cell) * mat["density"]),
                radius_mean=particle_radius(cell),
                jitter=0.0,
                custom_attributes={
                    "mpm:" + k: v for k, v in mat.items() if k != "density"
                },
            )
            self.ranges.append((start, len(builder.particle_q), name))

    def step(self):
        if self.kind == "ground":
            for _ in range(4):
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.frame_dt
        else:
            super().step()
        self.record()

    def record(self):
        pose = self.state_0.body_q.numpy()[0]
        velocity = self.state_0.body_qd.numpy()[0]
        if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
            raise RuntimeError("Nonfinite rigid state")
        force_z = None
        if self.kind != "ground":
            ids = self.collider_impulse_ids.numpy()
            impulses = self.collider_impulses.numpy()
            bodies = self.collider_body_id.numpy()
            valid = (ids >= 0) & (ids < len(bodies))
            selected = np.flatnonzero(valid)
            selected = selected[bodies[ids[selected]] == 0]
            force_z = float(impulses[selected, 2].sum() / self.frame_dt)
            if not np.isfinite(self.sand_state_0.particle_q.numpy()).all():
                raise RuntimeError("Nonfinite particle state")
        self.rows.append(
            dict(
                time_s=self.sim_time,
                x_m=float(pose[0]),
                y_m=float(pose[1]),
                z_m=float(pose[2]),
                vz_m_s=float(velocity[2]),
                terrain_force_z_N=force_z,
            )
        )

    def render(self):
        if self.kind != "ground":
            return super().render()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q.numpy()[0, 2] < -0.03:
            raise RuntimeError("Probe escaped through floor")

    def save(self):
        out = self.args.receipt_dir
        with (out / "trajectory.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        receipt = dict(
            status="exploratory_smoke_pass",
            material=self.kind,
            simulated_time_s=self.sim_time,
            frames=len(self.rows) - 1,
            elapsed_wall_s=time.monotonic() - self.started,
            particle_count=0
            if self.kind == "ground"
            else self.sand_model.particle_count,
            probe_mass_kg=float(self.model.body_mass.numpy()[0]),
            final_probe_z_m=self.rows[-1]["z_m"],
            peak_abs_terrain_force_z_N=max(
                abs(r["terrain_force_z_N"] or 0) for r in self.rows
            ),
            device=str(self.model.device),
            material_presets=MATERIALS,
            versions={
                n: importlib.metadata.version(n)
                for n in ["newton", "warp-lang", "mujoco", "mujoco-warp"]
            },
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Uncalibrated demo materials",
                "No humanoid policy",
                "Mixed means adjacent phases, no saturation or seepage model",
                "Ground reaction force not instrumented; null in CSV",
                "5 cm grid is too coarse for validated humanoid foot contact",
            ],
        )
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--material",
        choices=["ground", "sand", "mud", "water", "mixed"],
        default="sand",
    )
    parser.add_argument("--receipt-dir", type=Path, required=True)
    viewer, args = newton.examples.init(parser)
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    try:
        example = TerrainProbe(viewer, args)
        newton.examples.run(example, args)
        example.test_final()
        example.save()
    except Exception as exc:
        (args.receipt_dir / "failure.json").write_text(
            json.dumps({"status": "failed", "error": repr(exc)}, indent=2)
        )
        raise
