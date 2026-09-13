"""Large shared geometry and curriculum contract; live batched MPM is gated."""

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from course import make_course, export, add_mujoco_solids
from physics_audit import audit


def make_large_map(rows=16, columns=16):
    if rows < 1 or columns < 1:
        raise ValueError("Map dimensions must be positive")
    template = make_course()
    result = copy.deepcopy(template)
    result["schema"] = "terrain_large_map_v1"
    result["cells"] = []
    for i in range(rows * columns):
        cell = copy.deepcopy(template["cells"][i % 16])
        cell.update(id=i, origin=[(i % columns) * 3.4, (i // columns) * 3.0, 0.0])
        cell["curriculum_family"] = cell["name"]
        result["cells"].append(cell)
    result["walkways"] = [
        dict(
            name=f"vertical_{k}",
            lo=[2.45 + 3.4 * k, -0.75, -0.14],
            hi=[2.95 + 3.4 * k, (rows - 1) * 3 + 1.75, 0],
        )
        for k in range(columns - 1)
    ] + [
        dict(
            name=f"horizontal_{k}",
            lo=[-0.45, 1.75 + 3 * k, -0.14],
            hi=[(columns - 1) * 3.4 + 2.45, 2.25 + 3 * k, 0],
        )
        for k in range(rows - 1)
    ]
    result["execution"] = dict(
        mode="geometry_preview",
        live_mpm="one selected tile per isolated environment",
        training_ready=False,
        note="Do not allocate the full map as fine-grid MPM",
    )
    return result


class Curriculum:
    """Per-environment episode scheduler; explicit exposure and tracking success."""

    def __init__(self, count, levels=4):
        if count < 1 or levels < 1:
            raise ValueError("Positive environment and level counts required")
        self.levels = [0] * count
        self.streaks = [0] * count
        self.maximum = levels - 1

    def update(
        self,
        env_id,
        *,
        fell,
        root_rmse,
        joint_rmse,
        exposure_seconds,
        episode_seconds,
        numerically_valid,
        require_material_exposure=True,
    ):
        import math

        if not 0 <= env_id < len(self.levels):
            raise IndexError(env_id)
        values = [root_rmse, joint_rmse, exposure_seconds, episode_seconds]
        if not numerically_valid or not all(
            math.isfinite(v) and v >= 0 for v in values
        ):
            raise ValueError("Invalid simulation episodes must not update curriculum")
        if exposure_seconds > episode_seconds:
            raise ValueError("Exposure exceeds episode duration")
        success = (
            not fell
            and root_rmse <= 0.10
            and joint_rmse <= 0.25
            and (not require_material_exposure or exposure_seconds >= 2)
            and episode_seconds >= 6
        )
        if fell:
            self.levels[env_id] = max(0, self.levels[env_id] - 1)
        self.streaks[env_id] = self.streaks[env_id] + 1 if success else 0
        if self.streaks[env_id] >= 3:
            self.levels[env_id] = min(self.maximum, self.levels[env_id] + 1)
            self.streaks[env_id] = 0
        return self.levels[env_id]


def require_training_ready(validation):
    required = [
        "material_response",
        "grid_time_convergence",
        "coupling_energy",
        "isolated_reset",
        "batched_policy_step",
    ]
    missing = [k for k in required if validation.get(k) is not True]
    if missing:
        raise RuntimeError("Training blocked pending validation: " + ", ".join(missing))


if __name__ == "__main__":
    root = Path(__file__).parent / "assets/large-map"
    root.mkdir(exist_ok=True)
    course = make_large_map()
    export(root, course)
    xml = ET.Element("mujoco", model="large_terrain")
    ET.SubElement(xml, "compiler", angle="radian")
    ET.SubElement(xml, "option", gravity="0 0 -9.81")
    ET.SubElement(xml, "worldbody")
    add_mujoco_solids(xml, course)
    ET.ElementTree(xml).write(root / "terrain.xml")
    contract = dict(
        status="scheduler_and_geometry_only",
        training_ready=False,
        levels=[
            "rigid ground baseline",
            "single material",
            "adjacent combinations",
            "layered experimental",
        ],
        thresholds=dict(
            root_rmse_m=0.10,
            joint_rmse_rad=0.25,
            exposure_s=2,
            episode_s=6,
            consecutive_successes=3,
        ),
        reset_required=[
            "robot pose/velocity",
            "policy and reference history",
            "particle positions/velocities",
            "MPM deformation/plastic state",
            "solver warmstart/grid",
            "previous coupling wrench",
        ],
        backend_status={
            "MuJoCo": "existing single-environment bridge",
            "IsaacSim_IsaacLab_SONIC": "existing single-environment evaluation callback; no training hook",
        },
        validation={},
        resolution=audit(course, 0.02),
    )
    (root / "curriculum.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("Exported 256 tiles: 53.9 x 47.5 metres; training gate closed")
