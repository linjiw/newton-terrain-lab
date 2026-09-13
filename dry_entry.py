from terrain_settings import setting

"""Shared native SONIC train/evaluate entry; installs runtime before first reset."""

import atexit
import json
import os
from pathlib import Path
import runpy
import torch
from gear_sonic import train_agent_trl

runtime_config = json.loads(Path(os.environ["DRY_TERRAIN_CONFIG"]).read_text())
original_create = train_agent_trl.create_manager_env
runtimes = []


def create(*args, **kwargs):
    wrapper = original_create(*args, **kwargs)
    from dry_runtime import DryTerrainRuntime

    wrapper.dry_terrain = DryTerrainRuntime(wrapper, runtime_config)
    runtimes.append(wrapper.dry_terrain)
    return wrapper


train_agent_trl.create_manager_env = create
original_load = torch.load


def load(*args, **kwargs):
    path = args[0] if args else kwargs.get("f")
    if (
        isinstance(path, (str, Path))
        and Path(path).resolve() == Path(runtime_config["checkpoint"]).resolve()
    ):
        kwargs["map_location"] = "cpu"
    return original_load(*args, **kwargs)


torch.load = load


def close():
    for runtime in runtimes:
        runtime.close()


atexit.register(close)
try:
    if os.environ.get("DRY_TERRAIN_MODE", "eval") == "train":
        import sys
        from isaaclab.app import AppLauncher

        early_launcher = None
        if "--cfg" not in sys.argv:
            # Start Kit before TRL's heavy training dependency imports. Reuse the
            # same app when the original trainer reaches its launcher call.
            from filelock import FileLock

            hydra_argv = sys.argv[:]
            sys.argv = [sys.argv[0]]  # Kit must not consume Hydra's --config-path/name.
            try:
                with FileLock("/tmp/isaaclab_app_launcher.lock"):
                    early_launcher = AppLauncher(
                        headless=True,
                        enable_cameras=False,
                        device="cuda:0",
                        kit_args="--no-window",
                    )
            finally:
                sys.argv = hydra_argv

            def reuse_launcher(self, *args, **kwargs):
                self.__dict__.update(early_launcher.__dict__)

            AppLauncher.__init__ = reuse_launcher

        class TrainingOS:
            def __getattr__(self, name):
                return getattr(os, name)

            def _exit(self, status):
                close()
                # Preserve SONIC's immediate process exit after orderly worker shutdown.
                # Kit teardown can hang with the trainer's loaded GPU modules.
                os._exit(status)

        train_agent_trl.os = TrainingOS()
        train_agent_trl.main()
    else:
        runpy.run_path(
            str(Path(setting("sonic_root")) / "gear_sonic/eval_agent_trl.py"),
            run_name="__main__",
        )
finally:
    close()
    torch.load = original_load
