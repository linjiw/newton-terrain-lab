# Third-party notices

This repository contains terrain generation and integration code; it does not redistribute external teacher checkpoints, motion datasets, robot meshes, Isaac Sim, or the SONIC source tree.

- [Newton](https://github.com/newton-physics/newton), Apache-2.0: the MPM probes subclass its two-way coupling example and the bridge uses the same explicit coupling concepts. Copyright NVIDIA CORPORATION & AFFILIATES and Newton contributors as stated upstream. Local adaptations add course geometry, sample-volume correction, host-state conversion, per-environment workers and receipt checks.
- [NVIDIA SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl): native adapters import a separately installed SONIC TRL fork. The inspected fork licenses source under Apache-2.0 and model weights under the NVIDIA Open Model License. No weights are included or relicensed here.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [Isaac Sim](https://github.com/isaac-sim/IsaacSim), [MuJoCo](https://github.com/google-deepmind/mujoco), [Warp](https://github.com/NVIDIA/warp) and [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) are separately installed dependencies governed by their own licenses.

The Apache-2.0 license in this repository applies to this repository's code. It grants no rights to third-party data, weights or trademarks. Preserve applicable upstream notices when redistributing adapted components.
