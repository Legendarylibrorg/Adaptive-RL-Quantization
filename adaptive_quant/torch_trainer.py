from __future__ import annotations

import random
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.math_utils import mean
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, TorchPolicyAdapter, torch
from adaptive_quant.types import EpisodeResult, HardwareType


class TorchTrainer:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for `training_backend=\"pytorch\"`. "
                "Install a CUDA-enabled PyTorch build on the 4090 host before running `run_pytorch_4090.py`."
            ) from TORCH_IMPORT_ERROR
        self.config = config
        self.env = AdaptiveQuantizationEnv(config, log_path=log_path)
        self.policy = TorchPolicyAdapter(config)
        self.optimizer = self._build_optimizer()
        self.previous_action = [0.0, 0.0, 0.0]
        self.rng = random.Random(config.seed + 401)
        self.global_episode = 0
        self.ordered_hardware = config.ordered_hardware()

    def _build_optimizer(self):
        kwargs = {
            "lr": self.config.torch_learning_rate,
            "weight_decay": self.config.torch_weight_decay,
        }
        if self.config.torch_fused_optimizer and self.policy.device.type == "cuda":
            try:
                return torch.optim.AdamW(self.policy.model.parameters(), fused=True, **kwargs)
            except TypeError:
                pass
        return torch.optim.AdamW(self.policy.model.parameters(), **kwargs)

    def train(self) -> dict[str, float]:
        rewards: list[float] = []
        while self.global_episode < self.config.training_episodes:
            batch_size = min(self.config.torch_batch_episodes, self.config.training_episodes - self.global_episode)
            batch_records, batch_rewards = self._collect_batch(batch_size)
            self._update_policy(batch_records)
            rewards.extend(batch_rewards)

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
        for episode_index in range(episodes or self.config.evaluation_episodes):
            state = self.env.reset(previous_action=self.previous_action, forced_hardware=hardware)
            state_vector = state.to_vector(self.ordered_hardware)
            decision, _record = self.policy.act(state_vector, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=1_000_000 + episode_index)
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
        for episode_index in range(episodes):
            state = self.env.reset(previous_action=self.previous_action)
            state_vector = state.to_vector(self.ordered_hardware)
            decision, _record = self.policy.act(state_vector, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=2_000_000 + episode_index)
            results.append(result)
        return results

    def _collect_batch(self, batch_size: int) -> tuple[list[dict[str, Any]], list[float]]:
        records: list[dict[str, Any]] = []
        rewards: list[float] = []
        for _ in range(batch_size):
            state = self.env.reset(previous_action=self.previous_action)
            state_vector = state.to_vector(self.ordered_hardware)
            decision, record = self.policy.act(state_vector, deterministic=False)
            result = self.env.evaluate_current(decision, episode_index=self.global_episode)
            self.global_episode += 1
            self.previous_action = result.decision.feedback_vector(
                max_bits=max(self.config.discrete_bit_widths),
                scale_upper=self.config.scale_bounds[1],
                clip_upper=self.config.clip_bounds[1],
            )
            record["state_vector"] = state_vector
            record["reward"] = float(result.metrics.reward)
            records.append(record)
            rewards.append(float(result.metrics.reward))
        return records, rewards

    def _update_policy(self, records: list[dict[str, Any]]) -> None:
        device = self.policy.device
        states = self.policy.state_tensor([record["state_vector"] for record in records])
        rewards = torch.tensor([record["reward"] for record in records], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([record["log_prob"] for record in records], dtype=torch.float32, device=device)
        old_values = torch.tensor([record["value"] for record in records], dtype=torch.float32, device=device)
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        indices = list(range(len(records)))
        for _ in range(self.config.torch_update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.torch_minibatch_size):
                batch_indices = indices[start : start + self.config.torch_minibatch_size]
                if not batch_indices:
                    continue
                state_batch = states[batch_indices]
                record_batch = [records[index] for index in batch_indices]
                reward_batch = rewards[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                advantage_batch = advantages[batch_indices]

                log_probs, entropies, values = self.policy.evaluate_actions(state_batch, record_batch)
                ratios = torch.exp(log_probs - old_log_prob_batch)
                unclipped = ratios * advantage_batch
                clipped = torch.clamp(ratios, 1.0 - self.config.torch_ppo_clip, 1.0 + self.config.torch_ppo_clip) * advantage_batch
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values.float(), reward_batch)
                entropy_bonus = entropies.mean()
                loss = policy_loss + self.config.torch_value_coef * value_loss - self.config.torch_entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config.torch_max_grad_norm)
                self.optimizer.step()

    def close(self) -> None:
        self.env.logger.close()
