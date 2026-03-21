from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.math_utils import mean
from adaptive_quant.policy import PolicyTrace, UniversalQuantizationPolicy
from adaptive_quant.types import EpisodeResult, HardwareType


class Trainer:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        self.config = config
        self.env = AdaptiveQuantizationEnv(config, log_path=log_path)
        self.policy = UniversalQuantizationPolicy(config)
        self.previous_action = [0.0, 0.0, 0.0]

    def train(self) -> dict[str, float]:
        rewards: list[float] = []
        for episode_index in range(self.config.training_episodes):
            state = self.env.reset(previous_action=self.previous_action)
            decision, trace = self.policy.act(state, deterministic=False)
            result = self.env.evaluate_current(decision, episode_index=episode_index)
            self.policy.update(trace, result.metrics.reward)
            self.previous_action = result.decision.feedback_vector(
                max_bits=max(self.config.discrete_bit_widths),
                scale_upper=self.config.scale_bounds[1],
                clip_upper=self.config.clip_bounds[1],
            )
            rewards.append(result.metrics.reward)

        return {
            "episodes": float(len(rewards)),
            "mean_reward": mean(rewards),
            "best_reward": max(rewards) if rewards else 0.0,
            "final_reward": rewards[-1] if rewards else 0.0,
        }

    def evaluate(self, episodes: int | None = None, hardware: HardwareType | None = None) -> dict[str, float]:
        rewards: list[float] = []
        perplexities: list[float] = []
        latencies: list[float] = []
        throughputs: list[float] = []
        memories: list[float] = []
        stabilities: list[float] = []
        previous_action = list(self.previous_action)
        for episode_index in range(episodes or self.config.evaluation_episodes):
            state = self.env.reset(previous_action=previous_action, forced_hardware=hardware)
            decision, _trace = self.policy.act(state, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=1_000_000 + episode_index)
            previous_action = result.decision.feedback_vector(
                max_bits=max(self.config.discrete_bit_widths),
                scale_upper=self.config.scale_bounds[1],
                clip_upper=self.config.clip_bounds[1],
            )
            rewards.append(result.metrics.reward)
            perplexities.append(result.metrics.perplexity)
            latencies.append(result.metrics.latency_ms)
            throughputs.append(result.metrics.throughput_tps)
            memories.append(result.metrics.memory_mb)
            stabilities.append(result.metrics.stability_penalty)

        return {
            "mean_reward": mean(rewards),
            "mean_perplexity": mean(perplexities),
            "mean_latency_ms": mean(latencies),
            "mean_throughput_tps": mean(throughputs),
            "mean_memory_mb": mean(memories),
            "mean_stability_penalty": mean(stabilities),
        }

    def rollout(self, episodes: int) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        previous_action = list(self.previous_action)
        for episode_index in range(episodes):
            state = self.env.reset(previous_action=previous_action)
            decision, _trace = self.policy.act(state, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=2_000_000 + episode_index)
            previous_action = result.decision.feedback_vector(
                max_bits=max(self.config.discrete_bit_widths),
                scale_upper=self.config.scale_bounds[1],
                clip_upper=self.config.clip_bounds[1],
            )
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


def build_trainer(config: FrameworkConfig, log_path: str | None = None) -> Trainer:
    if config.training_backend == "pytorch":
        from adaptive_quant.torch_trainer import TorchTrainer

        return TorchTrainer(config, log_path=log_path)
    return Trainer(config, log_path=log_path)
