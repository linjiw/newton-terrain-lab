"""Export rigid terrain geometry; external MPM is required for live materials."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from course import add_mujoco_solids


def export_mujoco(course, output):
    root = ET.Element("mujoco", model="dry_terrain")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    ET.SubElement(root, "worldbody")
    add_mujoco_solids(root, course)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export_mujoco(json.loads(args.course.read_text()), args.output)
