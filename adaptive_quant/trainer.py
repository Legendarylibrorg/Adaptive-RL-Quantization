from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.base_trainer import TrainerBase
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import write_json
from adaptive_quant.policy import PolicyTrace, UniversalQuantizationPolicy
from adaptive_quant.trainer_utils import online_update_summary, reward_summary, training_row


class Trainer(TrainerBase):
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        super().__init__(config, log_path=log_path)
        self.policy = UniversalQuantizationPolicy(config)

    def train(self) -> dict[str, float]:
        if self.config.continuous_training:
            return self._train_continuous()
        return self._train_fixed()

    def _train_fixed(self) -> dict[str, float]:
        rewards: list[float] = []
        for episode_index in range(self.config.training_episodes):
            state = self.env.reset(previous_action=self.previous_action, phase="train")
            decision, trace = self.policy.act(state, deterministic=False)
            result = self.env.evaluate_current(decision, episode_index=episode_index)
            self.policy.update(trace, result.metrics.reward)
            self.previous_action = self._feedback_vector(result.decision)
            rewards.append(result.metrics.reward)
            self.training_history.append(training_row(float(episode_index), result))

        return reward_summary(rewards, updates=len(self.training_history))

    def _train_continuous(self) -> dict[str, float]:
        """Continuous learning: train up to max_training_episodes with periodic eval."""
        rewards: list[float] = []
        target = self.config.max_training_episodes
        eval_interval = self.config.eval_interval
        ckpt_interval = self.config.checkpoint_interval

        for episode_index in range(target):
            state = self.env.reset(previous_action=self.previous_action, phase="train")
            decision, trace = self.policy.act(state, deterministic=False)
            result = self.env.evaluate_current(decision, episode_index=episode_index)
            self.policy.update(trace, result.metrics.reward)
            self.previous_action = self._feedback_vector(result.decision)
            rewards.append(result.metrics.reward)
            self.training_history.append(training_row(float(episode_index), result))

            if eval_interval > 0 and (episode_index + 1) % eval_interval == 0:
                recent = rewards[-eval_interval:]
                eval_summary = self.evaluate()
                print(
                    f"[episode {episode_index + 1:,}] "
                    f"recent_reward={sum(recent) / len(recent):.3f}  "
                    f"eval_reward={eval_summary.get('mean_reward', 0):.3f}",
                    file=sys.stderr,
                )

            if ckpt_interval > 0 and (episode_index + 1) % ckpt_interval == 0:
                ckpt_path = self.config.final_checkpoint_path().replace(
                    "_final", f"_ep{episode_index + 1}"
                )
                self.save_checkpoint(ckpt_path)

        return reward_summary(rewards, updates=len(self.training_history))

    def update_online(self, updates: list[tuple[PolicyTrace, float]]) -> dict[str, float]:
        rewards: list[float] = []
        for trace, reward in updates:
            self.policy.update(trace, reward)
            rewards.append(reward)
        return online_update_summary(rewards)

    def save_checkpoint(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.config.run_name,
            "previous_action": self.previous_action,
            "training_history": self.training_history,
        }
        write_json(str(target.with_suffix(".json")), payload)
        return str(target.with_suffix(".json"))


def build_trainer(config: FrameworkConfig, log_path: str | None = None) -> Trainer:
    if config.training_backend == "pytorch":
        from adaptive_quant.torch_trainer import TorchTrainer

        return TorchTrainer(config, log_path=log_path)
    return Trainer(config, log_path=log_path)
