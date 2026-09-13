# Physics assumptions and interpretation

All geometry uses metres, Z up, kilograms and seconds, with gravity 9.81 m/s². `course.py` defines material presets; `dry_course.py` excludes water from the default course. Rigid USD contact material uses static/dynamic friction 0.8 and restitution 0; MuJoCo boxes use friction `0.8 0.005 0.0001`. Engines have different contact solvers, so identical coefficients do not imply identical trajectories.

| Preset | Density kg/m³ | Friction | Viscosity Pa·s | Yield stress Pa |
|---|---:|---:|---:|---:|
| Sand | 1600 | 0.75 | 0 | 0 |
| Mud | 1500 | 0 | 100 | 300 |

Mud additionally uses yield pressure 1e10 Pa and tensile yield ratio 1. The solver's limiting stiffness defaults (Young's modulus 1e15 Pa, Poisson ratio 0.3), zero damping/hardening/dilatancy, and sand default yield pressure 1e15 Pa are not measured soil properties. This is not a validated wet granular mixture or saturation model. Random rocks are fixed axis-aligned blocks, not moving rubble or reconstructed geological surfaces.

Newton's implicit MPM advances continuum material carried by particles on a background grid. Rendering particles as balls shows sampling locations. Surface reconstruction uses `newton.geometry.ParticleSurface`; its smoothing and kernel settings affect the visible boundary, not the simulated constitutive response. Live worker surfaces are sampled at approximately 10 Hz. Recorded USD/mesh animations are playback only.

Particle mass is density times sample volume. Newton reference volume is `(2 * radius)^3`, so `physics_audit.particle_radius` computes half the cube root of the spacing product. An earlier implementation used half the minimum spacing and inflated effective density for anisotropic sampling; that was corrected. Tests verify volume consistency. Particle centers inside rock boxes are removed; this is not exact cut-cell volume integration.

The recommended experimental voxel size 0.04 m has only 3.5 cells through a 0.14 m layer. It is coarse and not grid-converged. MPM uses 50 maximum iterations, tolerance 1e-4 and the bridge's strain basis/settings; see `mpm_bridge.py` for the executable source of truth. A smaller voxel can greatly increase memory and runtime.

The host engine owns humanoid articulation and rigid contacts. The bridge feeds body poses/COM velocities and effective per-link inertias into Newton, obtains material impulses, converts them to world forces/torques about COM, and applies them on the host's next step. This is explicit, lagged two-way coupling. Per-link inertia approximates the articulated response; neither exact coupled integration nor energy conservation has been established. Collider friction is taken from host geometries. Validate time-step convergence and energy behavior before physical conclusions.

## What has and has not been checked

Density/reference-volume and basic free-fall screens passed for three materials at two time steps. They do not validate sand shear, mud rheology, foot penetration or friction against real experiments. The unit tests check geometry agreement, material volume, curriculum, placement, state conversion and reset bookkeeping. Native tests check finite execution and isolation.

Water failed the acceptance series. In one finer-grid variant, buoyancy was near the expected value but particles leaked through the floor. A projection/ground-plane variant reached 9.556 N against 9.81 N expected with no below-floor particles, but settling velocity RMS 0.187 m/s exceeded the 0.1 m/s screen. Passing one force metric was insufficient. Water remains in standalone exploratory probes for future work and is not in the recommended dry map.

Before claims about real soil, run controlled penetration, shear, slump and settling tests against measured data; calibrate density/friction/yield/viscosity; repeat grid/time-step studies; check boundary containment, articulation coupling and force/energy balance; then evaluate multiple held-out motions and seeds. Continuous-looking rendering is not physical validation.

Primary references: [Newton 1.6.0](https://github.com/newton-physics/newton/releases/tag/v1.6.0), [Implicit MPM API](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverImplicitMPM.html), [ParticleSurface API](https://newton-physics.github.io/newton/latest/api/_generated/newton.geometry.ParticleSurface.html), [coupling concepts](https://newton-physics.github.io/newton/latest/concepts/coupling.html). Online latest documentation may differ from the pinned release; inspect the installed API when changing versions.
