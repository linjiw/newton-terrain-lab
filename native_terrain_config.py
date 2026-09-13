"""Bounded PhysX buffers for a single-environment integration test only."""

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import ModularTrackingEnvCfg


class TerrainTrackingEnvCfg(ModularTrackingEnvCfg):
    def override_settings(self):
        super().override_settings()
        if self.scene.num_envs != 1:
            raise ValueError(
                "Small-buffer diagnostic config supports exactly one environment"
            )
        p = self.sim.physx
        p.gpu_max_rigid_contact_count = 2**18
        p.gpu_max_rigid_patch_count = 2**16
        p.gpu_found_lost_pairs_capacity = 2**18
        p.gpu_found_lost_aggregate_pairs_capacity = 2**18
        p.gpu_total_aggregate_pairs_capacity = 2**18
        p.gpu_collision_stack_size = 2**24
        p.gpu_heap_capacity = 2**25
        p.gpu_temp_buffer_capacity = 2**23
        # The source dataset's single named Floor is absent in this course.
        # Its recorder assumes exactly one pair filter; do not substitute many
        # colliders and silently record only the first. All-body contact sensing
        # remains active; MPM exposure is measured from material wrenches.
        self.scene.left_foot_support_contact = None
        self.scene.right_foot_support_contact = None
