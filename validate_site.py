"""Validate static Pages entrypoints, local references and simulation media."""

from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent / "site"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for name in ["src", "href", "poster"]:
            if name in attrs:
                self.links.append(attrs[name])


def main():
    p = References()
    p.feed((ROOT / "index.html").read_text())
    errors = []
    for link in p.links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc:
            continue
        if not parts.path:
            if parts.fragment and parts.fragment not in p.ids:
                errors.append("Missing anchor: " + link)
        elif not (ROOT / unquote(parts.path)).is_file():
            errors.append("Missing file: " + link)
    for name in ["mixed", "sand", "mud"]:
        path = ROOT / "media" / f"{name}.mp4"
        if not path.is_file() or b"ftyp" not in path.read_bytes()[:32]:
            errors.append("Invalid MP4: " + name)
        if not (ROOT / "media" / f"{name}-poster.jpg").is_file():
            errors.append("Missing poster: " + name)
    course = json.loads((ROOT / "course-map.json").read_text())
    if len(course["cells"]) != 256:
        errors.append("Interactive map expects 16x16 default cells")
    if any(
        patch["material"] == "water"
        for cell in course["cells"]
        for patch in cell["patches"]
    ):
        errors.append("Water in dry showcase")
    evidence = json.loads((ROOT / "evidence.json").read_text())
    if not all(run["finite"] for run in evidence["native_runs"]):
        errors.append("Nonfinite curated rollout")
    print(
        json.dumps(
            {
                "ok": not errors,
                "local_references": len(p.links),
                "map_tiles": len(course["cells"]),
                "errors": errors,
            },
            indent=2,
        )
    )
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
