from terrain_settings import setting

"""Run a prepared diagnostic or explicitly selected future training configuration."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--mode", choices=["eval", "train"], default="eval")
p.add_argument("--view", action="store_true")
a = p.parse_args()
folder = a.folder.resolve()
root = Path(__file__).parent.resolve()
command = json.loads((folder / f"{a.mode}-command.json").read_text())
runtime_path = folder / ("train-runtime.json" if a.mode == "train" else "runtime.json")
label = a.mode
if a.view:
    if a.mode != "eval":
        raise ValueError("--view is evaluation only")
    label = "view"
    runtime = json.loads(runtime_path.read_text())
    runtime.update(visualize=True, output=str(folder / "view"))
    runtime_path = folder / "view-runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2))
    command = [x.replace("++headless=true", "++headless=false") for x in command]
    command = [
        f"++callbacks.im_eval.stage_config={runtime_path}"
        if x.startswith("++callbacks.im_eval.stage_config=")
        else x
        for x in command
    ]
start = time.monotonic()
with (folder / f"{label}.log").open("x") as log:
    result = subprocess.run(
        command,
        cwd=setting("sonic_root"),
        env={
            **os.environ,
            "PYTHONPATH": str(root) + os.pathsep + setting("sonic_root"),
            "DRY_TERRAIN_CONFIG": str(runtime_path),
            "DRY_TERRAIN_MODE": a.mode,
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=900 if a.mode == "eval" else None,
    )
(folder / f"{label}-receipt.json").write_text(
    json.dumps(
        dict(exit_code=result.returncode, wall_seconds=time.monotonic() - start),
        indent=2,
    )
)
raise SystemExit(result.returncode)
