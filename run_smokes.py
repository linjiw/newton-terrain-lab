"""Run remaining material probes sequentially, retaining every failure."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
failed = []
for material in ["ground", "mud", "water", "mixed"]:
    output = root / "outputs" / (material + "-v1")
    with (root / (material + "-v1.log")).open("x") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "terrain_probe.py"),
                "--material",
                material,
                "--viewer",
                "null",
                "--device",
                "cuda:0",
                "--num-frames",
                "100",
                "--receipt-dir",
                str(output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(material, result.returncode, flush=True)
    if result.returncode:
        failed.append(material)
if failed:
    sys.exit("Failed: " + ", ".join(failed))
