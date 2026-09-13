"""Episode metrics adapter for SONIC evaluation and future Isaac curriculum terms.

This reports proposed levels only. Terrain changes require validated reset and
training infrastructure; no live terrain migration is performed here.
"""

import numpy as np
from large_map import Curriculum


class TerrainCurriculumState:
    def __init__(self, num_envs):
        self.scheduler = Curriculum(num_envs)
        self.samples = [[] for _ in range(num_envs)]
        self.receipts = []

    def record(self, env_id, root_error_m, joint_rmse_rad, exposure_seconds, dt):
        values = [root_error_m, joint_rmse_rad, exposure_seconds, dt]
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or dt <= 0
            or exposure_seconds > dt + 1e-8
        ):
            raise ValueError("Invalid curriculum sample")
        self.samples[env_id].append(values)

    def finish(self, env_ids, fell=False):
        results = []
        for env_id in env_ids:
            values = np.asarray(self.samples[env_id])
            if not len(values):
                continue  # Initial environment reset is not an episode.
            dt = values[:, 3]
            outcome = dict(
                fell=bool(fell),
                root_rmse=float(np.sqrt(np.average(values[:, 0] ** 2, weights=dt))),
                joint_rmse=float(np.sqrt(np.average(values[:, 1] ** 2, weights=dt))),
                exposure_seconds=float(values[:, 2].sum()),
                episode_seconds=float(dt.sum()),
                numerically_valid=True,
            )
            level = self.scheduler.update(
                env_id,
                require_material_exposure=self.scheduler.levels[env_id] != 0,
                **outcome,
            )
            receipt = dict(env_id=int(env_id), proposed_level=level, **outcome)
            self.receipts.append(receipt)
            results.append(receipt)
            self.samples[env_id] = []
        return results


def terrain_curriculum_term(env, env_ids):
    """Isaac CurriculumTermCfg(func=...) hook; adapter must be installed first.

    Call only after complete episode metrics are recorded. Native evaluator
    consumes it directly; a trainer must wire terminal outcomes before reset.
    """
    adapter = env.terrain_curriculum_state
    for i in env_ids:
        adapter.finish([int(i)], fell=bool(env.termination_manager.terminated[int(i)]))
    return float(np.mean(adapter.scheduler.levels))
