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
    errors = []
    pages = {}
    for path in ROOT.rglob("*.html"):
        parser = References()
        parser.feed(path.read_text())
        pages[path.resolve()] = parser
    for path, parser in pages.items():
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing file: {link}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(ROOT)}: missing anchor: {link}")
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
                "html_pages": len(pages),
                "local_references": sum(len(p.links) for p in pages.values()),
                "map_tiles": len(course["cells"]),
                "errors": errors,
            },
            indent=2,
        )
    )
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
