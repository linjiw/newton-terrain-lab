"""Water-free sand/mud/rock course and reproducible exclusive tile allocation."""

import copy
import json
from pathlib import Path
import numpy as np
from course import make_course, export, MATERIALS


def make_dry_course(rows=16, columns=16, seed=23):
    if rows < 1 or columns < 4:
        raise ValueError("Need positive rows and at least four columns")
    rng = np.random.default_rng(seed)
    base = make_course()
    course = copy.deepcopy(base)
    course.update(
        schema="dry_terrain_v1",
        seed=seed,
        material_presets={k: v for k, v in MATERIALS.items() if k != "water"},
    )
    course["cells"] = []
    for i in range(rows * columns):
        level = i % 4
        families = [
            [["ground"]],
            [["sand"], ["mud"], ["ground", "sand"], ["ground", "mud"]],
            [
                ["sand", "mud"],
                ["sand", "rocks"],
                ["mud", "rocks"],
                ["ground", "rocks"],
                ["ground", "sand", "mud"],
                ["ground", "sand", "rocks"],
                ["ground", "mud", "rocks"],
                ["rocks"],
            ],
            [["sand", "mud", "rocks"], ["ground", "sand", "mud", "rocks"]],
        ]
        family = families[level][(i // 4) % len(families[level])]
        cell = copy.deepcopy(base["cells"][0])
        cell.update(
            id=i,
            origin=[(i % columns) * 10.5, (i // columns) * 10.5, 0],
            level=level,
            rocks=[],
        )
        names = [n for n in family if n != "rocks"] or ["ground"]
        cell["name"] = "+".join(family)
        cell["patches"] = []
        for j, name in enumerate(names):
            cell["patches"].append(
                dict(
                    material=name,
                    lo=[0.2 + 2.2 * j / len(names), -0.7, -0.14],
                    hi=[0.2 + 2.2 * (j + 1) / len(names), 1.7, 0],
                )
            )
        for k in range(2 + level if "rocks" in family else 0):
            x, y = rng.uniform(0.6, 2.15), rng.uniform(-0.45, 1.45)
            hx, hy = rng.uniform(0.06, 0.14, 2)
            height = rng.uniform(0.025, 0.04 + level * 0.025)
            cell["rocks"].append(
                dict(
                    name=f"rock_{k}",
                    lo=[x - hx, y - hy, -0.14],
                    hi=[x + hx, y + hy, height],
                )
            )
        cell["bounds"] = [[-2.0, -2.0, 0.0], [8.0, 8.0, 0.0]]
        cell["aprons"] = [
            dict(name="apron_left", lo=[-2, -2, -0.14], hi=[-0.45, 8, 0]),
            dict(name="apron_right", lo=[2.45, -2, -0.14], hi=[8, 8, 0]),
            dict(name="apron_bottom", lo=[-0.45, -2, -0.14], hi=[2.45, -0.75, 0]),
            dict(name="apron_top", lo=[-0.45, 1.75, -0.14], hi=[2.45, 8, 0]),
        ]
        course["cells"].append(cell)
    course["walkways"] = [
        dict(
            name=f"v{k}",
            lo=[8 + 10.5 * k, -2, -0.14],
            hi=[8.5 + 10.5 * k, 10.5 * (rows - 1) + 8, 0],
        )
        for k in range(columns - 1)
    ]
    course["walkways"] += [
        dict(
            name=f"h{k}",
            lo=[-2, 8 + 10.5 * k, -0.14],
            hi=[10.5 * (columns - 1) + 8, 8.5 + 10.5 * k, 0],
        )
        for k in range(rows - 1)
    ]
    course["meaning"] = (
        "Dry means no separate water phase; mud remains uncalibrated viscoplastic material. Rocks are rigid block proxies."
    )
    return course


class TileAllocator:
    def __init__(self, course, num_envs, seed=23):
        self.course = course
        self.rng = np.random.default_rng(seed)
        self.assignments = {}
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        # Reserve enough tiles at every level so worst-case same-level allocation works.
        if (
            min(sum(c["level"] == level for c in course["cells"]) for level in range(4))
            < num_envs
        ):
            raise ValueError("Not enough exclusive tiles per curriculum level")

    def assign_tiles(self, mapping):
        """Explicit exclusive allocation for controlled evaluation conditions."""
        chosen = {int(env): int(tile) for env, tile in mapping.items()}
        if any(env < 0 or env >= self.num_envs for env in chosen):
            raise ValueError("Invalid environment IDs")
        if any(
            tile < 0 or tile >= len(self.course["cells"]) for tile in chosen.values()
        ):
            raise ValueError("Invalid tile IDs")
        updated = {**self.assignments, **chosen}
        if len(set(updated.values())) != len(updated):
            raise ValueError("Tile allocation must remain exclusive")
        self.assignments = updated
        return chosen

    def assign(self, env_ids, levels):
        ids = [int(i) for i in env_ids]
        if len(ids) != len(set(ids)) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("Invalid environment IDs")
        occupied = {v for k, v in self.assignments.items() if k not in ids}
        result = {}
        for i in ids:
            options = [
                c["id"]
                for c in self.course["cells"]
                if c["level"] == levels[i] and c["id"] not in occupied
            ]
            if not options:
                raise ValueError("No free tile at requested level")
            selected = int(self.rng.choice(options))
            occupied.add(selected)
            result[i] = selected
        self.assignments.update(result)
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "assets/dry-map"
    )
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--no-particle-previews",
        action="store_true",
        help="Omit static display points from large USD exports; live physics is unchanged",
    )
    args = parser.parse_args()
    if min(args.rows, args.columns) < 1:
        parser.error("rows and columns must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    course = make_dry_course(args.rows, args.columns, args.seed)
    export(args.output, course, preview_particles=not args.no_particle_previews)
    print(
        json.dumps(
            dict(output=str(args.output), cells=len(course["cells"]), water=False)
        )
    )
