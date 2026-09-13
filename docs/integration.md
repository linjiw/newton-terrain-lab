# Integrating terrain with Isaac Lab, SONIC, Isaac Sim and MuJoCo

## Environments and external inputs

Use two environments: a Newton worker Python with `requirements-sim.txt`, and your existing native Isaac Lab / Isaac Sim / SONIC Python. Do not install Isaac into the worker environment. The checked development combination was Linux, Python 3.11, RTX 5080 16 GB, Newton 1.6.0, Warp 1.17.0, MuJoCo 3.12.0, Isaac Sim 5.1 and Isaac Lab extension 0.54.2.

The native adapter targets the SONIC TRL fork interface used during development:

- `gear_sonic.train_agent_trl.create_manager_env`, `main`, and `eval_agent_trl.py`;
- `gear_sonic.envs.manager_env.modular_tracking_env_cfg.ModularTrackingEnvCfg`;
- `gear_sonic.trl.trainer.ppo_trainer_aux_loss.TRLAuxLossPPOTrainer`;
- `gear_sonic.research.hindsight_training.runtime.load_release_checkpoint` for the training checkpoint format;
- wrapper `.env` and `.motion_command`, native robot/contact sensors, and the observation/actor contracts consumed by `dry_eval_callback.py`.

The source checkout HEAD was `8020b771ad6271f97fbfcd90a9cd0066f396825f`; Isaac Lab HEAD was `37ddf626871758333d6ed89cf64ad702aef127d0`. These identify local checkouts, not a claim that their working trees or all research extensions are publicly released. **This repository does not include that fork or make stock SONIC compatible automatically.** If your installation lacks these modules, port these adapter boundaries before launching. Core map generation and Newton probes remain usable independently.

Create `.venv` using the README and install `requirements-sim.txt`, then copy `local.example.json` to ignored `local.json`. All paths must be absolute. Each key can also be supplied as `TERRAIN_<UPPERCASE_KEY>`; `TERRAIN_SETTINGS` selects another JSON file.

| Setting | Required content |
|---|---|
| `sonic_root` | Your compatible SONIC source checkout |
| `isaac_python` | Its native Isaac Python executable |
| `newton_python` | Worker Python executable with requirements-sim installed |
| `model_dir` | Matching G1 MuJoCo `model.xml` and referenced assets; same body/joint names and inertial conventions as the native robot |
| `checkpoint` | Compatible trusted teacher checkpoint; adjacent `config.yaml` must describe its SONIC model/task |
| `eval_motion_file` | Motion library directory/file accepted by your SONIC motion loader |
| `dataset_manifest` | JSON identifying disjoint train and reserved pools (schema below) |

Only historical CPU teacher/replay tools additionally require `native_site_packages`, `teacher_runtime` (providing `sonic_cpu_inference`), `teacher_scripts` (providing `sonic_mujoco_forecast_v2` and its dependencies), `reference_dir`, `robot_xml` and `checkpoint_sha256`. The native dry worker no longer imports those teacher helpers. The historical single-cell `prepare_native.py` also requires `legacy_eval_command`; new integrations should use `prepare_dry_run.py`.

Weights/data are not downloaded or redistributed. Supply checkpoints you trust: the SONIC loader can deserialize Python objects. Preserve the model architecture, action scaling, joint ordering, history and motion sampling conventions. The tested G1 teacher had 29 actions, 930 actor observations, 10 history frames and 10 future motion targets. Changing the robot is an adapter task, not just replacing its visual mesh.

Dataset manifest example (use real pool hashes and IDs):

```json
{
  "train": {"path": "/absolute/path/train/motion_pool.pkl", "sha256": "POOL_SHA256", "count": 84, "ids": ["train_001"]},
  "reserved": {"path": "/absolute/path/reserved/motion_pool.pkl", "sha256": "POOL_SHA256", "count": 20, "ids": ["eval_001"]}
}
```

The abbreviated ID lists above illustrate the schema; supply the full lists matching `count`. Preparation checks file hashes and split-ID disjointness. The original experiment used an admitted 84-clip training pool and 20 reserved clips. Use your own licensed data and retain held-out evaluation; preparation uses your manifest count, not a hardcoded 84. Pool contents must independently agree with the manifest; hashing alone cannot validate semantic labels.

## Prepare, evaluate, train

Run these commands from the repository with the worker environment active and the map generated:

```bash
python prepare_dry_run.py --course-dir assets/dry-map --output outputs/eval-01 \
  --num-envs 2 --initial-level 3 --steps 120 --surfaces --selective-reset-at 60
python run_dry.py outputs/eval-01
```

Inspect `eval-receipt.json`, `run/runtime.json`, `run/worker.json`, and the evaluation callback outputs/logs. A diagnostic selective reset should leave peers unchanged. Surface updates should be positive when enabled. Native automatic terminations remain active. CLI logs use exclusive creation: create a fresh prepared folder to repeat a run.

For native GUI rendering, `python run_dry.py outputs/eval-01 --view` uses a separate output/log directory. This launches native live surface updates and follows environment 0. Headless mesh updates were tested; GUI behavior depends on the display setup and was not validated in the original receipt.

Prepare a minimal training smoke:

```bash
python prepare_dry_run.py --output outputs/ppo-smoke --num-envs 2 \
  --training-iterations 1 --rollout-steps 8 --ppo-epochs 1
python run_dry.py outputs/ppo-smoke --mode train
```

Training starts at ground level 0; to specifically smoke-test mixed-material PPO, set `initial_level` to 3 in the generated `train-runtime.json` before launching. The tested mixed-material smoke used one clip loaded from the admitted pool; normal preparation loads the full declared training pool. The trainer strictly loads policy/value weights into a fresh optimizer; it rejects resume. It records initialization and optimizer receipts under `train-runtime/`, leaving the supplied checkpoint untouched. It removes the inherited `hindsight` callback and its historical run directory, but review other callbacks in your own checkpoint configuration before running a new fork.

For a future small curriculum experiment:

```bash
python prepare_dry_run.py --output outputs/curriculum-01 --num-envs 4 \
  --training-iterations 100 --rollout-steps 24 --ppo-epochs 1
python run_dry.py outputs/curriculum-01 --mode train
```

This is an experiment profile, not a validated convergence recipe. Save/evaluation cadence remains inherited from your teacher configuration; review `train.yaml` and set an appropriate cadence for longer work. Evaluation does not run PPO. Training is launched only by `--mode train`.

## Scene, origins, reset and curriculum contracts

Load the USD once as a shared world. `DryTrackingEnvCfg` works around the tested source configuration's single-environment scene restriction and replicates the robots. `DryTerrainRuntime` attaches before the first reset and sets `scene.env_origins` to allocated world positions. The worker uses tile-local coordinates and independent MPM state for each environment. Other robots should remain collision-isolated using your task's collision filtering.

`TileAllocator` samples unused tiles at each environment's current level. Every environment starts on a safe rigid pad with x jitter [-0.08, 0.05] m and y jitter [-0.10, 0.10] m. Selective resets consume only that environment's metrics, assign a tile, clear its material forces, reset its worker state, and reset its native robot and policy history. Peers are preserved. Out-of-tile motion is terminated; this is not a seamless shared deformable world for robots to roam across arbitrary tiles.

Curriculum levels: 0 ground; 1 individual sand/mud and ground combinations; 2 paired mixtures/rocks; 3 sand+mud+rocks and ground+sand+mud+rocks. Promotion requires three consecutive qualifying episodes: root RMSE ≤0.10 m, joint RMSE ≤0.25 rad, duration ≥6 s, no native failure, and material exposure ≥2 s except on ground. Exposure uses foot/ankle external force above 5 N. Failure demotes. Review these thresholds for your motions; they are starter criteria, not universal success metrics. The historical scheduler field `fell` receives native failure and can include non-fall terminations; runtime receipts retain actual termination names.

Material contacts are external to PhysX and do not automatically appear in native contact sensors. `dry_rewards.undesired_contacts` unions native contact history with per-control-step material forces by body name and avoids double counting. Audit other rewards/observations that depend on contact; they do not automatically inherit this fusion. The optional source single-Floor contact filters are disabled because this terrain has many colliders.

To adapt a different Isaac Lab task, implement equivalent hooks for root/joint/COM states, world-frame body wrenches at each physics substep, selective reset, motion-reference origin/history changes, termination metrics and contact reward fusion. Preserve physics stepping order; validate state/force frame conventions before policy evaluation. The runtime currently monkey-patches simulator/reset methods and is version-sensitive.

## Direct MuJoCo and Isaac Sim use

`export_mujoco.py` writes only rigid geometry. To add it to a robot model, call `course.add_mujoco_solids(xml_root, course)` before compiling. It removes world planes and inserts named terrain geometry. Resolve robot mesh references with the correct asset paths before compiling from an XML string.

For live materials, construct `MPMBridge(model, data, selected_cells, voxel_size=0.04)`. At each host physics step call `step_mujoco(model, data, dt)`, assign its world-frame force/torque array to `data.xfrc_applied`, and advance MuJoCo. Rebuild/reset material state when episodes reset. If your application has other external forces, compose them deliberately instead of overwriting them.

For Isaac Sim, load the USD for geometry and attach a bridge that converts native body COM states to worker coordinates and applies returned wrenches. The shipped `DryTerrainRuntime` does this through Isaac Lab; there is no general standalone Isaac Sim extension installer. A USDA import or surface replay alone never activates material physics.

## Record videos and hold evaluation conditions fixed

`python terrain_doctor.py --native` checks installed packages, configured files and native Python modules without launching physics. It is a setup check, not a task/GPU compatibility guarantee.

Pass `--record-surfaces` to `prepare_dry_run.py` to save native joint/root states and numeric surface snapshots for `render_dry_showcase.py`. See [the recording guide](showcase.md). Training has recording disabled by default.

For controlled material evaluation, `--eval-tiles 1,5 --num-envs 2` binds environments to sand and mud respectively on the default seed-23 map. Explicit tiles remain exclusive and unchanged across resets. Native failures and policy-history resets still occur; the assigned condition's level overrides curriculum changes for these evaluation environments. The generated training runtime always clears explicit tiles and retains random curriculum allocation. Without this flag, evaluation continues using the normal curriculum, including demotion after failure.
