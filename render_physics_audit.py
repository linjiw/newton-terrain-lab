"""Diagnostic rendering: same recorded water samples as points and a surface."""

from pathlib import Path
import json
import numpy as np
import warp as wp
from newton.geometry import ParticleSurface
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from course import COLORS
from large_map import make_large_map

root = Path(__file__).parent
out = root / "outputs/physics-audit"
data = np.load(root / "outputs/course-all-live-v3/particles.npz")
p = data["positions"][25]
mask = np.all(np.isclose(data["colors"], COLORS["water"], atol=1e-5), axis=1)
mask &= (p[:, 0] > 10.2) & (p[:, 0] < 12.7) & (p[:, 1] < 1.9)
p = p[mask].copy()
p[:, 0] -= 10.2
with wp.ScopedDevice("cuda:0"):
    surface = ParticleSurface(
        voxel_size=0.025, kernel_radius=0.13, threshold=0.25, field_smooth_iterations=1
    )
    vertices, indices, _ = surface.extract(
        wp.array(p, dtype=wp.vec3), wp.full(len(p), 0.025, dtype=float)
    ).to_arrays()
    v, f = vertices.numpy(), indices.numpy().reshape(-1, 3)
    assert np.isfinite(v).all() and len(f) > 0
np.savez_compressed(out / "water-display-surface.npz", vertices=v, faces=f)
fig = plt.figure(figsize=(12, 5))
for i, title in enumerate(
    ["MPM sample-point display", "Surface reconstructed from the same samples"]
):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    if i == 0:
        ax.scatter(*p.T, s=7, c=[COLORS["water"]])
    else:
        mesh = Poly3DCollection(v[f], facecolor=COLORS["water"], edgecolor="none")
        ax.add_collection3d(mesh)
    ax.set(
        xlim=(0.1, 2.5),
        ylim=(-0.8, 1.8),
        zlim=(-0.2, 0.4),
        title=title,
        xlabel="x (m)",
        ylabel="y (m)",
    )
    ax.set_box_aspect((2.4, 2.6, 0.6))
fig.suptitle(
    "Historical unvalidated rollout, t=1 s — surface smoothing changes appearance only"
)
fig.tight_layout()
fig.savefig(out / "points-versus-surface.png", dpi=170)
plt.close(fig)
course = make_large_map()
fig, ax = plt.subplots(figsize=(12, 10))
for cell in course["cells"]:
    for patch in cell["patches"]:
        lo = np.array(patch["lo"]) + cell["origin"]
        hi = np.array(patch["hi"]) + cell["origin"]
        ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color=COLORS[patch["material"]]))
for w in course["walkways"]:
    ax.add_patch(
        Rectangle(
            w["lo"][:2], *(np.array(w["hi"]) - w["lo"])[:2], color=COLORS["ground"]
        )
    )
ax.set(
    xlim=(-1, 54),
    ylim=(-1, 48),
    aspect="equal",
    xlabel="metres",
    ylabel="metres",
    title="256-tile terrain map — geometry preview, training validation pending",
)
fig.savefig(out / "large-map.png", dpi=160)
(out / "surface-receipt.json").write_text(
    json.dumps(
        dict(
            source="course-all-live-v3",
            frame=25,
            points=len(p),
            triangles=len(f),
            physics_changed=False,
            radius_m=0.025,
            radius_note="display estimate from old 0.1 m voxel grid; not simulation volume",
        ),
        indent=2,
    )
    + "\n"
)
print("Rendered diagnostic surface and map", len(f), "triangles")
