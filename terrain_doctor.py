"""Check a local terrain installation without launching physics or training."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from terrain_settings import setting


def inspect(native=False):
    checks = []
    for package in [
        "numpy",
        "usd-core",
        "newton",
        "warp-lang",
        "mujoco",
        "mujoco-warp",
    ]:
        try:
            version = importlib.metadata.version(package)
            checks.append(dict(check=package, ok=True, detail=version))
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                dict(check=package, ok=False, detail="Install requirements-sim.txt")
            )
    if native:
        targets = {
            "sonic_root": "gear_sonic/train_agent_trl.py",
            "model_dir": "model.xml",
            "checkpoint": "",
            "dataset_manifest": "",
            "isaac_python": "",
            "newton_python": "",
        }
        for key, suffix in targets.items():
            try:
                path = Path(setting(key))
                path = path / suffix if suffix else path
                exists = path.is_file()
                checks.append(
                    dict(
                        check=key,
                        ok=exists,
                        detail="Found"
                        if exists
                        else "File not found; check local.json",
                    )
                )
            except RuntimeError as error:
                checks.append(dict(check=key, ok=False, detail=str(error)))
        try:
            root = Path(setting("sonic_root"))
            for path in [
                "gear_sonic/eval_agent_trl.py",
                "gear_sonic/research/hindsight_training/runtime.py",
                "gear_sonic/trl/trainer/ppo_trainer_aux_loss.py",
            ]:
                checks.append(
                    dict(
                        check=path,
                        ok=(root / path).is_file(),
                        detail="Required fork adapter module",
                    )
                )
            code = 'import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n) is not None for n in ["torch","isaaclab","isaacsim"]}))'
            proc = subprocess.run(
                [setting("isaac_python"), "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            modules = json.loads(proc.stdout) if proc.returncode == 0 else {}
            checks.append(
                dict(
                    check="native Python modules",
                    ok=all(modules.values()) and len(modules) == 3,
                    detail=modules,
                )
            )
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            checks.append(
                dict(check="native Python modules", ok=False, detail=str(error))
            )
    return dict(
        ok=all(c["ok"] for c in checks),
        checks=checks,
        scope="Dependency/file check only; not GPU, task compatibility, calibration or training validation",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    result = inspect(a.native)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
