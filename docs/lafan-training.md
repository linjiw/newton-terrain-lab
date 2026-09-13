# Single-motion LAFAN dance training

The requested experiment is **1,024 environments × 2,000 SONIC training iterations**, with 24 rollout steps per environment and one PPO optimization epoch per iteration (49,152,000 total transitions). Here “2,000 epochs” is interpreted as 2,000 outer training iterations. The prepared profile uses eight minibatches and periodic checkpoint saves.

**Status: prepared, not launched at the requested scale.** A two-environment, one-update compatibility smoke on live terrain passed with finite changed policy parameters. This is not evidence that the full dance is learned. The 1,024-environment live MPM curriculum does not fit the available GPU memory; a physics/batch/hardware choice is required before the long run.

## Motion input

Source: [G1-retargeted LAFAN1 `dance1_subject1.csv`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/blob/ce1572906efe6157840e8474d5a0d7aa87481e74/g1/dance1_subject1.csv). Revision `ce1572906efe6157840e8474d5a0d7aa87481e74`; CSV SHA256 `e2a369e92e5ad076c5acffff7bce53157a0e6cd9e7e9f6a08dadfb1eb2928d8f`.

The complete 3,945-frame clip is kept at its source rate of 30 Hz (131.467 seconds); SONIC resamples it for the 50 Hz controller. Its documented XYZ / XYZW / 29-joint layout is converted by joint name into the SONIC MJCF order. Axis-angle joint rotations come from the target MJCF axes. Root XY is translated to center the dance footprint at tile-local (1, 1) m; height, timing and joint angles are unchanged. The resulting root bounds are approximately x [-1.142, 3.142], y [-1.559, 3.559] m. This is the root path, not a bound on all robot body geometry.

The CSV passed finite-value, unit-quaternion and target joint-limit screens. Retargeted data is kinematic; these checks do not establish dynamic feasibility. The conversion provides no genuine SMPL observations and must use the **G1 encoder only**. No other motion is mixed into this pool, and there is no held-out motion split in a single-motion fitting experiment. The original configured 84-motion pool is left untouched.

```bash
pip install -r requirements-sim.txt
python prepare_lafan_motion.py /path/to/dance1_subject1.csv \
  --output outputs/lafan-motion
```

The output contains `motion_pool.pkl`, a one-motion `dataset.json` manifest, an explicitly empty reserved pool, and `conversion.json`. Download data separately under its source terms; no motion data or model weights are committed here.

## Prepare the requested map and configuration

A 64 × 64 map provides 4,096 exclusive tiles, 1,024 at each curriculum level, spanning 671.5 × 671.5 m. Static particle previews can be omitted from the large USD export; this does not remove live material physics.

```bash
python dry_course.py --rows 64 --columns 64 --no-particle-previews \
  --output assets/dance-1024
TERRAIN_DATASET_MANIFEST=/absolute/path/outputs/lafan-motion/dataset.json \
TERRAIN_EVAL_MOTION_FILE=/absolute/path/outputs/lafan-motion/motion_pool.pkl \
TERRAIN_CHECKPOINT=/absolute/path/teacher/checkpoint.pt \
python prepare_dry_run.py --course-dir assets/dance-1024 \
  --output outputs/lafan-1024 --num-envs 1024 \
  --training-iterations 2000 --rollout-steps 24 --ppo-epochs 1
```

For the larger batch, set `algo.config.num_mini_batches: 8` and the model-save callback frequencies to 100 (numbered) / 25 (last) in the generated `train.yaml`. Ensure `encoder_sample_probs` is G1=1, teleop=0, SMPL=0. The starting model used in the compatibility test is the previously compared step-8,000 SONIC teacher, SHA256 `48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed`. The generated training runtime starts at ground level 0 with normal random curriculum allocation.

## Why the full live run is blocked on this machine

At the checked 4 cm voxel size a full single-material tile has 105,600 particles. Seven Newton state fields alone use 172 bytes per particle: position, velocity, four 3×3 matrix fields and plastic volume. A hypothetical 1,024 full-material-tile batch therefore uses about 17.3 GiB for those fields alone.

Accounting for the actual mixed map's partial material regions and rock exclusions gives a tighter **13.98 GiB particle-state lower bound for a reachable 1,024-env level-3 allocation**, versus approximately **10.83 GiB free on the 16 GB RTX 5080** at the check. Material model arrays, solver grids, collision buffers, Isaac/PhysX, the policy and PPO storage are additional. Starting on ground does not make the later curriculum fit.

```bash
python terrain_capacity.py outputs/lafan-1024/train-runtime.json \
  --output outputs/lafan-1024/capacity.json
```

`run_dry.py` screens batches above four environments before launching Isaac and rejects a curriculum whose particle-state bound already exceeds free GPU memory. A passing screen is **not** evidence of sufficient total memory or throughput. Workers still use separate worlds stepped sequentially over IPC, and large-batch performance remains unvalidated. The old hardcoded 64-env scene check now uses the actual map capacity.

Possible next routes are a smaller validated live-material batch, a separately implemented and explicitly labeled rigid-terrain approximation for 1,024 environments, or larger hardware plus a scalable MPM implementation. The toolkit does not silently replace live sand/mud physics with rigid collision surfaces.
