"""Native SONIC configuration for the water-free multi-environment map."""

import copy
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import ModularTrackingEnvCfg


class DryTrackingEnvCfg(ModularTrackingEnvCfg):
    def _setup_from_hydra(self, config, *args, **kwargs):
        # The source dataset adapter only permits one robot in a global USD.
        # Construct that scene once, then configure robot replication explicitly;
        # DryTerrainRuntime supplies exclusive tile origins before the first reset.
        n = int(config["num_envs"])
        local = copy.deepcopy(config)
        local["num_envs"] = 1
        super()._setup_from_hydra(local, *args, **kwargs)
        self.config["num_envs"] = n
        self.scene.num_envs = n
        self.scene.terrain.num_envs = n

    def override_settings(self):
        super().override_settings()
        n = self.scene.num_envs
        if n > 64:
            raise ValueError(
                "Default map has 64 exclusive tiles per level; enlarge it before increasing num_envs"
            )
        p = self.sim.physx
        p.gpu_max_rigid_contact_count = max(2**18, n * 8192)
        p.gpu_max_rigid_patch_count = max(2**16, n * 2048)
        p.gpu_found_lost_pairs_capacity = 2**20
        p.gpu_found_lost_aggregate_pairs_capacity = 2**20
        p.gpu_total_aggregate_pairs_capacity = 2**20
        p.gpu_collision_stack_size = 2**25
        from dry_rewards import undesired_contacts

        self.rewards.undesired_contacts.func = undesired_contacts
        self.scene.left_foot_support_contact = None
        self.scene.right_foot_support_contact = None
