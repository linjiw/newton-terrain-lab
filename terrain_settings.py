"""External integration paths; core generation never requires this configuration."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def setting(name, default=None):
    """Environment TERRAIN_<NAME> overrides TERRAIN_SETTINGS/local.json."""
    path = Path(os.environ.get("TERRAIN_SETTINGS", str(ROOT / "local.json")))
    values = json.loads(path.read_text()) if path.is_file() else {}
    value = os.environ.get("TERRAIN_" + name.upper(), values.get(name, default))
    if value is None or value == "":
        raise RuntimeError(
            f"Missing terrain setting {name!r}; configure local.json or TERRAIN_{name.upper()}. See docs/integration.md."
        )
    return str(value)
