"""Conservative particle-state memory screen; excludes solver grids and host policy."""

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from dry_course import TileAllocator

# Newton 1.6 State: q, qd, four mat33 fields, and Jp (float32).
PARTICLE_STATE_BYTES = (3 + 3 + 4 * 9 + 1) * 4


def particle_count_lower_bound(cell, voxel_size):
    count = 0
    for patch in cell["patches"]:
        if patch["material"] == "ground":
            continue
        lo, hi = np.array(patch["lo"]), np.array(patch["hi"])
        dims = np.maximum(1, np.ceil((hi - lo) / (voxel_size / 2)).astype(int))
        spacing = (hi - lo) / dims
        removed_upper = 0
        for rock in cell.get("rocks", []):
            inside = []
            for axis in range(3):
                samples = lo[axis] + spacing[axis] * (np.arange(dims[axis]) + 0.5)
                inside.append(
                    np.count_nonzero(
                        (samples >= rock["lo"][axis]) & (samples <= rock["hi"][axis])
                    )
                )
            removed_upper += int(np.prod(inside))
        count += max(0, int(np.prod(dims)) - removed_upper)
    return count


def capacity(course, num_envs, voxel_size):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    TileAllocator(course, num_envs)
    levels = {}
    for level in range(4):
        counts = sorted(
            (
                particle_count_lower_bound(c, voxel_size)
                for c in course["cells"]
                if c["level"] == level
            ),
            reverse=True,
        )
        levels[str(level)] = dict(
            particles_lower_bound=sum(counts[:num_envs]),
            particle_state_gib_lower_bound=sum(counts[:num_envs])
            * PARTICLE_STATE_BYTES
            / 2**30,
        )
    return dict(
        num_envs=num_envs,
        tiles=len(course["cells"]),
        voxel_size=voxel_size,
        bytes_per_particle_state=PARTICLE_STATE_BYTES,
        reachable_level_cases=levels,
        maximum_case_particle_state_gib_lower_bound=max(
            v["particle_state_gib_lower_bound"] for v in levels.values()
        ),
        excludes=[
            "Newton material model arrays",
            "Newton solver grids/workspaces",
            "Isaac/PhysX robots and contacts",
            "policy and PPO optimizer/storage",
            "worker CPU memory",
        ],
        scope="Lower bound for reachable exclusive same-level allocations; passing is not a feasibility or throughput guarantee",
    )


def screen_runtime(config):
    result = capacity(
        json.loads(Path(config["course"]).read_text()),
        config["num_envs"],
        config["voxel_size"],
    )
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    available = float(subprocess.check_output(command, text=True).strip()) / 1024
    result["available_gpu_gib_before_isaac"] = available
    result["particle_state_exceeds_available"] = (
        result["maximum_case_particle_state_gib_lower_bound"] > available
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runtime", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    r = screen_runtime(json.loads(a.runtime.read_text()))
    text = json.dumps(r, indent=2)
    print(text)
    if a.output:
        a.output.write_text(text + "\n")
