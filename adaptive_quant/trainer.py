from __future__ import annotations

from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.policy import PolicyTrace, UniversalQuantizationPolicy
from adaptive_quant.trainer_utils import EvaluationAccumulator, feedback_vector, training_row
from adaptive_quant.types import EpisodeResult, HardwareType


class Trainer:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        self.config = config
        self.env = AdaptiveQuantizationEnv(config, log_path=log_path)
        self.policy = UniversalQuantizationPolicy(config)
        self.previous_action = [0.0, 0.0, 0.0]
        self.training_history: list[dict[str, float]] = []
        self._max_bits = max(config.discrete_bit_widths)
        self._scale_upper = config.scale_bounds[1]
        self._clip_upper = config.clip_bounds[1]

    def train(self) -> dict[str, float]:
        rewards: list[float] = []
        for episode_index in range(self.config.training_episodes):
            state = self.env.reset(previous_action=self.previous_action)
            decision, trace = self.policy.act(state, deterministic=False)
            result = self.env.evaluate_current(decision, episode_index=episode_index)
            self.policy.update(trace, result.metrics.reward)
            self.previous_action = self._feedback_vector(result.decision)
            rewards.append(result.metrics.reward)
            self.training_history.append(training_row(float(episode_index), result))

        return {
            "episodes": float(len(rewards)),
            "mean_reward": mean(rewards),
            "best_reward": max(rewards) if rewards else 0.0,
            "final_reward": rewards[-1] if rewards else 0.0,
            "updates": float(len(self.training_history)),
        }

    def evaluate(self, episodes: int | None = None, hardware: HardwareType | None = None) -> dict[str, float]:
        accumulator = EvaluationAccumulator()
        previous_action = list(self.previous_action)
        for episode_index in range(episodes or self.config.evaluation_episodes):
            state = self.env.reset(previous_action=previous_action, forced_hardware=hardware)
            decision, _trace = self.policy.act(state, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=1_000_000 + episode_index)
            previous_action = self._feedback_vector(result.decision)
            accumulator.add_metrics(result.metrics)

        return accumulator.summary()

    def rollout(self, episodes: int) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        previous_action = list(self.previous_action)
        for episode_index in range(episodes):
            state = self.env.reset(previous_action=previous_action)
            decision, _trace = self.policy.act(state, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=2_000_000 + episode_index)
            previous_action = self._feedback_vector(result.decision)
            results.append(result)
        return results

    def close(self) -> None:
        self.env.logger.close()

    def act_online(self, state, deterministic: bool = False):
        return self.policy.act(state, deterministic=deterministic)

    def update_online(self, updates: list[tuple[PolicyTrace, float]]) -> dict[str, float]:
        rewards: list[float] = []
        for trace, reward in updates:
            self.policy.update(trace, reward)
            rewards.append(reward)
        return {
            "batch_size": float(len(rewards)),
            "mean_reward": mean(rewards),
        }

    def snapshot_policy(self):
        return self.policy.snapshot()

    def restore_policy(self, snapshot) -> None:
        self.policy.restore(snapshot)

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

    def _feedback_vector(self, decision) -> list[float]:
        return feedback_vector(
            decision,
            max_bits=self._max_bits,
            scale_upper=self._scale_upper,
            clip_upper=self._clip_upper,
        )


def build_trainer(config: FrameworkConfig, log_path: str | None = None) -> Trainer:
    if config.training_backend == "pytorch":
        from adaptive_quant.torch_trainer import TorchTrainer

        return TorchTrainer(config, log_path=log_path)
    return Trainer(config, log_path=log_path)
