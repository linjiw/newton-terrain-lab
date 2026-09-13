"""Sequential, fresh-output frozen-teacher course evaluation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cells", type=int, nargs="+", default=list(range(16)))
p.add_argument("--seconds", type=float, default=6.0)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parent
plan = dict(
    cells=a.cells,
    seconds=a.seconds,
    motion="00265",
    teacher="motion2scene repaired step1400",
    scope="exploratory single-motion backend/terrain test, no held-out performance claim",
)
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
rows = []
for cell in a.cells:
    out = a.output / f"cell-{cell:02d}"
    cmd = [
        sys.executable,
        str(root / "humanoid_run.py"),
        "--course",
        "--mpm",
        "--cell",
        str(cell),
        "--seconds",
        str(a.seconds),
        "--output",
        str(out),
    ]
    with (a.output / f"cell-{cell:02d}.log").open("x") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "2"},
            timeout=900,
        )
    row = {"cell": cell, "exit_code": proc.returncode}
    if (out / "result.json").exists():
        row.update(json.loads((out / "result.json").read_text()))
    rows.append(row)
    (a.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        cell,
        proc.returncode,
        row.get("stop_reason"),
        row.get("root_rmse_m"),
        flush=True,
    )
if any(r["exit_code"] for r in rows):
    sys.exit(1)
