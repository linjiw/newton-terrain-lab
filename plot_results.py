"""Static figure from the recorded exploratory evaluation."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rows = json.loads((root / "outputs/course-eval-v2/results.json").read_text())
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharey=True, layout="constrained")
y = np.arange(16)
colors = ["#3c8d9e" if r["stop_reason"] == "horizon" else "#c96843" for r in rows]
ax[0].barh(y, [r["simulated_seconds"] for r in rows], color=colors)
ax[0].set_yticks(y, [f"{r['cell']:02d}  {r['cell_name']}" for r in rows])
ax[0].invert_yaxis()
ax[0].set_xlabel("Time before fall or horizon (s)")
ax[0].set_xlim(0, 6.5)
ax[1].barh(y, [r["root_rmse_m"] for r in rows], color=colors)
ax[1].set_xlabel("Root position RMSE over recorded duration (m)")
for a in ax:
    a.grid(axis="x", alpha=0.2)
    a.set_axisbelow(True)
fig.suptitle(
    "Frozen motion2scene teacher: one motion, 16 terrain conditions\nBlue: reached 6 s without falling · Orange: fell",
    fontsize=15,
)
fig.supxlabel(
    "Exploratory: coarse 8 cm MPM grid, uncalibrated materials, explicit coupling. Durations differ; no aggregate tracking ranking.",
    fontsize=9,
)
fig.savefig(root / "outputs/course-results.png", dpi=180)
print(root / "outputs/course-results.png")
