"""Convert a G1-retargeted LAFAN CSV into a single SONIC G1-encoder motion pool."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import joblib
from terrain_settings import setting

# Source dataset's documented left-leg, right-leg, waist, left-arm, right-arm order.
JOINTS = [
    *[
        f"{side}_{name}_joint"
        for side in ["left", "right"]
        for name in [
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        ]
    ],
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *[
        f"{side}_{name}_joint"
        for side in ["left", "right"]
        for name in [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    ],
]


def convert(csv, model_xml, output, name="lafan1_dance1_subject1"):
    values = np.loadtxt(csv, delimiter=",")
    if values.ndim != 2 or values.shape[1] != 36 or not np.isfinite(values).all():
        raise ValueError("Expected finite XYZ + XYZW quaternion + 29 joint columns")
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    if not np.allclose(norms, 1, atol=1e-4):
        raise ValueError("Source root quaternions are not normalized")
    tree = ET.parse(model_xml).getroot()
    joints = tree.find("worldbody").findall(".//joint")
    joints = [j for j in joints if j.get("name") in JOINTS]
    names = [j.get("name") for j in joints]
    if len(names) != 29 or set(names) != set(JOINTS):
        raise ValueError("Target MJCF must contain the documented G1 joints")
    dof = values[:, [7 + JOINTS.index(n) for n in names]].astype(np.float32)
    axes = np.array([np.fromstring(j.get("axis", "0 0 1"), sep=" ") for j in joints])
    pose = np.zeros((len(values), 30, 3), dtype=np.float32)
    pose[:, 0] = Rotation.from_quat(values[:, 3:7]).as_rotvec()
    pose[:, 1:] = dof[:, :, None] * axes[None, :, :]
    trans = values[:, :3].copy()
    # Translate the complete dance footprint into the 10 m tile; keep height and timing.
    offset = np.array([1.0, 1.0]) - (trans[:, :2].min(0) + trans[:, :2].max(0)) / 2
    trans[:, :2] += offset
    entry = dict(
        root_trans_offset=trans.astype(np.float32),
        pose_aa=pose,
        dof=dof,
        root_rot=values[:, 3:7].astype(np.float32),
        smpl_joints=np.zeros((len(values), 24, 3), np.float32),
        fps=30,
    )
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump({name: entry}, output / "motion_pool.pkl", compress=True)
    joblib.dump({}, output / "reserved_empty.pkl", compress=True)
    splits = {}
    for split, filename, ids in [
        ("train", "motion_pool.pkl", [name]),
        ("reserved", "reserved_empty.pkl", []),
    ]:
        path = (output / filename).resolve()
        splits[split] = dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            count=len(ids),
            ids=ids,
        )
    (output / "dataset.json").write_text(json.dumps(splits, indent=2) + "\n")
    violations = {}
    for i, j in enumerate(joints):
        limits = np.fromstring(j.get("range", "-inf inf"), sep=" ")
        count = int(
            ((dof[:, i] < limits[0] - 1e-4) | (dof[:, i] > limits[1] + 1e-4)).sum()
        )
        if count:
            violations[names[i]] = count
    receipt = dict(
        motion=name,
        frames=len(values),
        source_fps=30,
        duration_s=(len(values) - 1) / 30,
        source_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        target_mjcf_sha256=hashlib.sha256(model_xml.read_bytes()).hexdigest(),
        joint_names=names,
        root_xy_translation_m=offset.tolist(),
        root_bounds_m=[trans.min(0).tolist(), trans.max(0).tolist()],
        joint_limit_violation_frame_counts=violations,
        encoder="g1 only; no real SMPL observations supplied",
        single_motion=True,
        held_out_split=False,
        dynamics_validated=False,
    )
    (output / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="lafan1_dance1_subject1")
    p.add_argument(
        "--model-xml",
        type=Path,
        default=Path(setting("sonic_root"))
        / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
    )
    a = p.parse_args()
    convert(a.csv, a.model_xml, a.output, a.name)
