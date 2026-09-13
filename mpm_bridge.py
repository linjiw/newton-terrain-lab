"""Live MPM bridge: measured body poses/COM velocities in, world wrenches out.

Host engine owns articulation integration. Coupling is explicit, lagged by one
physics step; per-link MPM effective masses approximate articulated inertia.
"""

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from course import MATERIALS, COLORS
from physics_audit import particle_radius


class MPMBridge:
    def __init__(self, host_model, host_data, cells, voxel_size=0.08, device="cuda:0"):
        import mujoco

        self.device = device
        self.voxel_size = voxel_size
        self.ranges = []
        with wp.ScopedDevice(device):
            b = newton.ModelBuilder()
            # Exact host link COM/inertia, without letting collision geometry add mass.
            for i in range(1, host_model.nbody):
                q = host_model.body_iquat[i]
                r = np.empty(9)
                mujoco.mju_quat2Mat(r, q)
                r = r.reshape(3, 3)
                inertia = r @ np.diag(host_model.body_inertia[i]) @ r.T
                b.add_body(
                    label=host_model.body(i).name,
                    mass=float(host_model.body_mass[i]),
                    com=wp.vec3(host_model.body_ipos[i]),
                    inertia=wp.mat33(inertia.astype(np.float32)),
                    lock_inertia=True,
                    xform=wp.transform(
                        host_data.xpos[i], host_data.xquat[i][[1, 2, 3, 0]]
                    ),
                )
            for g in range(host_model.ngeom):
                if not (host_model.geom_contype[g] or host_model.geom_conaffinity[g]):
                    continue
                bid = int(host_model.geom_bodyid[g]) - 1
                kw = dict(
                    xform=wp.transform(
                        host_model.geom_pos[g], host_model.geom_quat[g][[1, 2, 3, 0]]
                    ),
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=0.0, mu=float(host_model.geom_friction[g, 0])
                    ),
                )
                size = host_model.geom_size[g]
                kind = int(host_model.geom_type[g])
                if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    b.add_shape_sphere(bid, radius=float(size[0]), **kw)
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    b.add_shape_capsule(
                        bid, radius=float(size[0]), half_height=float(size[1]), **kw
                    )
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    b.add_shape_box(
                        bid,
                        hx=float(size[0]),
                        hy=float(size[1]),
                        hz=float(size[2]),
                        **kw,
                    )
                elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
                    b.add_ground_plane()
                elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(host_model.geom_dataid[g])
                    v = int(host_model.mesh_vertadr[mid])
                    nv = int(host_model.mesh_vertnum[mid])
                    f = int(host_model.mesh_faceadr[mid])
                    nf = int(host_model.mesh_facenum[mid])
                    mesh = newton.Mesh(
                        host_model.mesh_vert[v : v + nv].copy(),
                        host_model.mesh_face[f : f + nf].reshape(-1).copy(),
                    )
                    b.add_shape_mesh(bid, mesh=mesh, **kw)
                else:
                    raise ValueError(f"Unsupported collision geom {g}: {kind}")
            self.collider_model = b.finalize()
            particles = newton.ModelBuilder()
            SolverImplicitMPM.register_custom_attributes(particles)
            for cell in cells:
                for p in cell["patches"]:
                    name = p["material"]
                    if name == "ground":
                        continue
                    lo = np.array(p["lo"]) + cell["origin"]
                    hi = np.array(p["hi"]) + cell["origin"]
                    dims = np.maximum(
                        1, np.ceil((hi - lo) / (voxel_size / 2)).astype(int)
                    )
                    spacing = (hi - lo) / dims
                    mat = MATERIALS[name]
                    start = len(particles.particle_q)
                    if cell.get("rocks"):
                        grid = np.stack(
                            np.meshgrid(
                                *[
                                    lo[k] + spacing[k] * (np.arange(dims[k]) + 0.5)
                                    for k in range(3)
                                ],
                                indexing="ij",
                            ),
                            axis=-1,
                        ).reshape(-1, 3)
                        keep = np.ones(len(grid), dtype=bool)
                        for rock in cell["rocks"]:
                            rlo = np.array(rock["lo"]) + cell["origin"]
                            rhi = np.array(rock["hi"]) + cell["origin"]
                            keep &= ~np.all((grid >= rlo) & (grid <= rhi), axis=1)
                        grid = grid[keep]
                        particles.add_particles(
                            pos=grid,
                            vel=np.zeros_like(grid),
                            mass=np.full(len(grid), np.prod(spacing) * mat["density"]),
                            radius=np.full(len(grid), particle_radius(spacing)),
                            custom_attributes={
                                "mpm:" + k: v for k, v in mat.items() if k != "density"
                            },
                        )
                    else:
                        particles.add_particle_grid(
                            pos=wp.vec3(lo + spacing / 2),
                            rot=wp.quat_identity(),
                            vel=wp.vec3(0.0),
                            dim_x=int(dims[0]),
                            dim_y=int(dims[1]),
                            dim_z=int(dims[2]),
                            cell_x=float(spacing[0]),
                            cell_y=float(spacing[1]),
                            cell_z=float(spacing[2]),
                            mass=float(np.prod(spacing) * mat["density"]),
                            radius_mean=particle_radius(spacing),
                            jitter=0.0,
                            custom_attributes={
                                "mpm:" + k: v for k, v in mat.items() if k != "density"
                            },
                        )
                    self.ranges.append((start, len(particles.particle_q), name))
            self.model = particles.finalize()
            self.state = self.model.state()
            config = SolverImplicitMPM.Config()
            config.voxel_size = voxel_size
            config.max_iterations = 50
            config.tolerance = 1e-4
            config.strain_basis = "P0"
            config.critical_fraction = 0.0
            self.solver = SolverImplicitMPM(self.model, config=config)
            self.solver.setup_collider(model=self.collider_model)
            self.state.body_q = wp.clone(self.collider_model.body_q)
            self.state.body_qd = wp.zeros(
                self.collider_model.body_count, dtype=wp.spatial_vector
            )
            self.state.body_f = wp.zeros_like(self.state.body_qd)
            self.colors = np.zeros((self.model.particle_count, 3), np.float32)
            for lo, hi, name in self.ranges:
                self.colors[lo:hi] = COLORS[name]
        self.host_body_count = host_model.nbody
        self.previous_wrench = np.zeros((host_model.nbody, 6))

    def advance(self, positions, quaternions_xyzw, velocities_com, dt):
        """Inputs: [links,3], [links,4], [links,6] (linear then angular).

        Returns [links,6] world force then torque about each link COM, N/Nm.
        The caller applies this wrench during its next host physics step.
        """
        with wp.ScopedDevice(self.device):
            if not all(
                np.isfinite(x).all()
                for x in (positions, quaternions_xyzw, velocities_com)
            ):
                raise FloatingPointError("Nonfinite host state passed to MPM")
            poses = np.concatenate((positions, quaternions_xyzw), axis=1).astype(
                np.float32
            )
            self.state.body_q.assign(poses)
            self.state.body_qd.assign(np.asarray(velocities_com, np.float32))
            self.state.clear_forces()
            self.solver.step(self.state, self.state, contacts=None, control=None, dt=dt)
            impulses, points, collider_ids = self.solver.collect_collider_impulses(
                self.state
            )
            ids = collider_ids.numpy()
            imp = impulses.numpy()
            pts = points.numpy()
            body_ids = self.solver.collider_body_index.numpy()
            valid = (ids >= 0) & (ids < len(body_ids))
            ids = ids[valid]
            imp = imp[valid]
            pts = pts[valid]
            bodies = body_ids[ids]
            valid = bodies >= 0
            bodies = bodies[valid]
            imp = imp[valid]
            pts = pts[valid]
            com = self.collider_model.body_com.numpy()
            from scipy.spatial.transform import Rotation

            world_com = positions + Rotation.from_quat(quaternions_xyzw).apply(com)
            force = imp / dt
            torque = np.cross(pts - world_com[bodies], force)
            result = np.zeros((len(positions), 6), np.float64)
            np.add.at(result, bodies, np.concatenate((force, torque), axis=1))
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite MPM wrench")
            return result

    def step_mujoco(self, model, data, dt):
        import mujoco

        velocity = np.zeros((model.nbody - 1, 6))
        for i in range(1, model.nbody):
            v = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, i, v, 0)
            velocity[i - 1] = np.r_[v[3:], v[:3]]
        # Remove the previous impulse's velocity increment as in Newton's example,
        # using per-link effective inertia (an approximation for articulations).
        mass = model.body_mass[1:]
        velocity[:, :3] -= dt * np.divide(
            self.previous_wrench[1:, :3],
            mass[:, None],
            out=np.zeros_like(velocity[:, :3]),
            where=mass[:, None] > 0,
        )
        for i in range(1, model.nbody):
            r = data.ximat[i].reshape(3, 3)
            inertia = model.body_inertia[i]
            delta = np.divide(
                r.T @ self.previous_wrench[i, 3:],
                inertia,
                out=np.zeros(3),
                where=inertia > 0,
            )
            velocity[i - 1, 3:] -= dt * (r @ delta)
        wrench = self.advance(
            data.xpos[1:], data.xquat[1:][:, [1, 2, 3, 0]], velocity, dt
        )
        self.previous_wrench[1:] = wrench
        return self.previous_wrench.copy()
