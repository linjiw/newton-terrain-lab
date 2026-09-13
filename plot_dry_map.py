"""Water-free map layout and measured multi-environment spawn locations."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from course import COLORS

import argparse

p = argparse.ArgumentParser()
p.add_argument("--course", type=Path, default=Path("assets/dry-map/course.json"))
p.add_argument("--receipt", type=Path)
p.add_argument("--output", type=Path, default=Path("outputs/overview.png"))
a = p.parse_args()
course = json.loads(a.course.read_text())
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax in axes:
    for cell in course["cells"]:
        origin = np.array(cell["origin"])
        ax.add_patch(Rectangle(origin[:2] - 2, 10, 10, color="#cbd0d1"))
        for p in cell["patches"]:
            lo = np.array(p["lo"]) + origin
            hi = np.array(p["hi"]) + origin
            ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color=COLORS[p["material"]]))
        for rock in cell["rocks"]:
            lo = np.array(rock["lo"]) + origin
            hi = np.array(rock["hi"]) + origin
            ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color="#3e454a"))
    for w in course["walkways"]:
        lo = np.array(w["lo"])
        hi = np.array(w["hi"])
        ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], color="#cbd0d1"))
    ax.set_aspect("equal")
    ax.set_xlabel("metres")
    ax.set_ylabel("metres")
if a.receipt:
    record = json.loads((a.receipt).read_text())
    initial = next(e for e in reversed(record["resets"]) if len(e["env_ids"]) > 0)
    for i, origin in zip(initial["env_ids"], initial["origins"]):
        axes[0].scatter(*origin[:2], s=70, c="red", edgecolors="white")
        axes[0].annotate(
            f"env {i}", origin[:2], xytext=(5, 5), textcoords="offset points"
        )
axes[0].set(
    xlim=(-3, 167),
    ylim=(-3, 167),
    title="256 exclusive tiles • 167.5 × 167.5 m • optional measured spawn positions",
)
axes[1].set(
    xlim=(27.5, 40.5),
    ylim=(-3, 9),
    title="Mixed sand/mud/rock tile and surrounding walkable ground",
)
fig.tight_layout()
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=160)
print("Saved map overview")
