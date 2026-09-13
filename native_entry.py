from terrain_settings import setting

"""Run the existing native evaluator with CPU checkpoint deserialization.

Only the exact selected checkpoint is affected. load_state_dict copies policy
weights onto the evaluator's chosen device; unused optimizer tensors stay CPU.
"""

from pathlib import Path
import runpy
import torch

checkpoint = Path(setting("checkpoint"))
original_load = torch.load


def load(*args, **kwargs):
    path = args[0] if args else kwargs.get("f")
    if isinstance(path, (str, Path)) and Path(path).resolve() == checkpoint:
        kwargs["map_location"] = "cpu"
    return original_load(*args, **kwargs)


torch.load = load
try:
    runpy.run_path(
        str(Path(setting("sonic_root")) / "gear_sonic/eval_agent_trl.py"),
        run_name="__main__",
    )
finally:
    torch.load = original_load
