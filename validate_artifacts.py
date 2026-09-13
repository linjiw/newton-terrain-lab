"""Validate recorded state/force artifacts without re-running simulations."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "outputs/course-eval-v2/results.json").read_text())
assert [r["cell"] for r in rows] == list(range(16))
for r in rows:
    assert r["exit_code"] == 0
    f = ROOT / "outputs/course-eval-v2" / f"cell-{r['cell']:02d}"
    with np.load(f / "rollout.npz") as d:
        assert len(d["qpos"]) == r["steps"] + 1
        assert d["qpos"].shape[1] == 36
        assert np.isfinite(d["qpos"]).all() and np.isfinite(d["actions"]).all()
        np.testing.assert_allclose(
            np.linalg.norm(d["qpos"][:, 3:7], axis=1), 1, atol=1e-6
        )
        if r["stop_reason"] == "fall":
            assert d["qpos"][-1, 2] < 0.3
        else:
            assert r["steps"] == 300
    if r["live_mpm"]:
        assert r["peak_material_force_N"] > 0
        with np.load(f / "particles.npz") as d:
            assert np.isfinite(d["positions"]).all()
            assert np.all(np.diff(d["time_s"]) > 0)
            assert d["positions"].shape[1] == len(d["colors"])
assert rows[0]["stop_reason"] == "horizon"
receipt = dict(
    state="passed",
    episodes=len(rows),
    falls=sum(r["stop_reason"] == "fall" for r in rows),
    horizon_completions=sum(r["stop_reason"] == "horizon" for r in rows),
    checks=[
        "all episodes returned zero",
        "finite physical states/actions/particles",
        "unit root quaternions",
        "record lengths and chronological particle samples",
        "fall threshold agrees with final pose",
        "nonzero material feedback",
    ],
    limitation="Software/data checks, not material calibration or backend equivalence",
)
(ROOT / "outputs/artifact-validation.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt))
