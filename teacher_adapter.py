from terrain_settings import setting

"""Frozen motion2scene repaired teacher, with the existing SONIC CPU interface."""

from pathlib import Path
import sys
import site

ROOT = Path(__file__).resolve().parent
NATIVE = Path(setting("native_site_packages"))
site.addsitedir(str(NATIVE))
RUNTIME = Path(setting("teacher_runtime"))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, setting("teacher_scripts"))
import torch
import sonic_cpu_inference as sonic

CHECKPOINT = Path(setting("checkpoint"))
CHECKPOINT_SHA = setting("checkpoint_sha256")


def load_teacher():
    torch.set_num_threads(2)
    teacher = sonic.SonicCPUInference()
    if sonic.sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Motion2scene teacher checkpoint hash mismatch")
    import pickle
    import types

    local_pickle = types.ModuleType("terrain_teacher_pickle")
    local_pickle.Unpickler = sonic._CheckpointUnpickler
    local_pickle.load = pickle.load
    saved = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False, pickle_module=local_pickle
    )
    state = saved.get("actor_model_state_dict", saved.get("policy_state_dict"))
    teacher.actor.load_state_dict(state, strict=True)
    teacher.actor.eval()
    teacher.actor.requires_grad_(False)
    return teacher


if __name__ == "__main__":
    import json

    teacher = load_teacher()
    obs = torch.zeros(1, 930)
    command = torch.zeros(1, 10, 58)
    ori = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(1, 1, 6).repeat(1, 10, 1)
    actual = teacher.infer(obs, command, ori)["action_mean"]
    expected = teacher.original_forward(teacher.full_observation(obs, command, ori))
    error = float((actual - expected).abs().max())
    assert error < 1e-5
    print(
        json.dumps(
            {
                "teacher_sha256": CHECKPOINT_SHA,
                "tensor_forward_max_error": error,
                "action_shape": list(actual.shape),
                "physics_steps": 0,
            }
        )
    )
