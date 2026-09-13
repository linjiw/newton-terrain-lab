"""SONIC PPO with isolated warm-start receipts and no historical run callbacks."""

import hashlib
import json
from pathlib import Path
import torch
from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from gear_sonic.research.hindsight_training.runtime import load_release_checkpoint


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, p in module.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DryPPOTrainer(TRLAuxLossPPOTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError(
                "This configuration uses a fresh optimizer; resume requires separate configuration"
            )
        checkpoint = load_release_checkpoint(checkpoint_path)
        model = self.accelerator.unwrap_model(self.model)
        model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        model.value_model.load_state_dict(checkpoint["value_state_dict"], strict=True)
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        if optimizer.state:
            raise ValueError("Expected a fresh optimizer")
        self.initial_policy_hash = parameter_hash(model.policy)
        self.receipt_dir = Path(self.env.dry_terrain.config["output"])
        with (self.receipt_dir / "initialization.json").open("x") as f:
            json.dump(
                dict(
                    checkpoint=str(checkpoint_path),
                    policy_hash=self.initial_policy_hash,
                    policy_strict=True,
                    critic_strict=True,
                    optimizer_state_entries=0,
                ),
                f,
                indent=2,
            )
        return checkpoint

    def train(self):
        result = super().train()
        model = self.accelerator.unwrap_model(self.model)
        final_hash = parameter_hash(model.policy)
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        receipt = dict(
            global_step=int(self.state.global_step),
            policy_changed=final_hash != self.initial_policy_hash,
            finite_parameters=finite,
            optimizer_state_entries=len(optimizer.state),
            final_policy_hash=final_hash,
            num_envs=self.env.num_envs,
        )
        (self.receipt_dir / "optimizer-receipt.json").write_text(
            json.dumps(receipt, indent=2)
        )
        if not finite:
            raise FloatingPointError("Nonfinite learned parameters")
        return result
