"""Batch-vs-independent inference parity for the exact frozen teacher."""

import json
from pathlib import Path
from teacher_adapter import load_teacher, CHECKPOINT_SHA, torch

torch.manual_seed(17)
t = load_teacher()
obs = torch.randn(4, 930) * 0.1
commands = torch.randn(4, 10, 58) * 0.1
ori = torch.tensor([1.0, 0, 0, 1, 0, 0]).reshape(1, 1, 6).repeat(4, 10, 1)
with torch.no_grad():
    batch = t.infer(obs, commands, ori)["action_mean"]
    serial = torch.cat(
        [
            t.infer(obs[i : i + 1], commands[i : i + 1], ori[i : i + 1])["action_mean"]
            for i in range(4)
        ]
    )
    err = float((batch - serial).abs().max())
assert batch.shape == (4, 29) and torch.isfinite(batch).all() and err < 1e-5
result = dict(
    teacher_sha256=CHECKPOINT_SHA,
    batch_size=4,
    action_shape=list(batch.shape),
    maximum_batch_serial_error=err,
    passed=True,
    scope="Tensor inference only; no batched physics or optimizer step",
)
(Path(__file__).parent / "outputs/teacher-batch.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(result)
