from terrain_settings import setting

"""Launch a prepared native scene in the existing Isaac environment."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

folder = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parent
cmd = json.loads((folder / "command.json").read_text())
start = time.monotonic()
with (folder / "process.log").open("x") as log:
    p = subprocess.run(
        cmd,
        cwd=setting("sonic_root"),
        env={
            **os.environ,
            "PYTHONPATH": str(root) + os.pathsep + setting("sonic_root"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
receipt = dict(exit_code=p.returncode, wall_seconds=time.monotonic() - start)
(folder / "launch-receipt.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
sys.exit(p.returncode)
