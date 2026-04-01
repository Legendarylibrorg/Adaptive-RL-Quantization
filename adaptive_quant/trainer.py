from __future__ import annotations

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
