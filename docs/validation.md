# Validation record

The original development runs used Linux / RTX 5080 16 GB and the versions listed in the integration guide. Large videos, simulator state archives, checkpoints, private motion pools and machine-specific raw logs are deliberately not shipped. The summary below reports observed short-run results; it is not an independently reproduced benchmark.

| Run | Scope | Observation |
|---|---|---|
| dry-large-batch-v4 | 4 envs, 180 control steps, 720 physics batches, final 256-tile map | finite actions/rewards; two automatic terminations; selective reset preserved peer robot and MPM states; exit 0; 51.84 s |
| dry-contact-surfaces-v6 | 2 envs, 120 control steps | 24 live native surface updates; selective reset peer isolation; finite; exit 0; 48.92 s |
| dry-ppo-smoke-v6 | 2 envs, 8 rollout steps, 1 PPO update, initial mixed level 3 | 16 transitions; finite changed policy; 69 optimizer-state entries; exit 0; 20.86 s |

The source teacher checkpoint hash remained unchanged after the PPO smoke. Training used a disposable warm-start student and a fresh optimizer. Sixteen transitions do not demonstrate improved tracking. A four-env/100-update profile was prepared but not launched in the original development session.

The map overview contains measured four-environment initial placements from the first run above. It is a layout plot, not a rendered screenshot of material dynamics.

## Reproduction layers

1. CPU: run `python -m unittest discover -s tests -v` and `ruff check .`. Tests create temporary exports; no private assets are needed. Install Torch to include the contact-fusion case.
2. Generate the full dry map, export/compile its MuJoCo rigid XML and verify 256 tiles, 15 combinations, no water and 3,270 rigid geometries.
3. Newton GPU: run `python terrain_probe.py --material sand --viewer null --device cuda:0 --num-frames 100 --receipt-dir outputs/sand-probe`. `validate_material_basics.py` runs the basic material checks; its output is under `outputs/physics-audit`.
4. Native integration: follow `integration.md`, run short two/four-env diagnostics, inspect finite states and selective reset receipts; enable surfaces for a live mesh update check.
5. PPO: use a fresh prepared folder and run one small optimizer update; inspect `train-runtime/optimizer-receipt.json` and process exit. Then evaluate learning over longer runs and held-out motions.

Public-packaging checks are recorded in `packaging-validation.json`. Native receipt schemas may differ between evaluation callbacks; use the executable callback as the source of truth. CPU CI does not certify CUDA, Isaac Sim startup, solver calibration or policy quality.
