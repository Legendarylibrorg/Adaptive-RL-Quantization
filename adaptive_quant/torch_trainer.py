from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.math_utils import mean
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, TorchPolicyAdapter, torch
from adaptive_quant.trainer_utils import EvaluationAccumulator, feedback_vector
from adaptive_quant.types import EpisodeResult, HardwareType


class TorchTrainer:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for `training_backend=\"pytorch\"`. "
                "Install a CUDA-enabled PyTorch build on the target GPU host before running a PyTorch entrypoint."
            ) from TORCH_IMPORT_ERROR
        self.config = config
        self.env = AdaptiveQuantizationEnv(config, log_path=log_path)
        self.policy = TorchPolicyAdapter(config)
        self.optimizer = self._build_optimizer()
        self.previous_action = [0.0, 0.0, 0.0]
        self.rng = random.Random(config.seed + 401)
        self.global_episode = 0
        self.update_index = 0
        self.ordered_hardware = config.ordered_hardware()
        self.training_history: list[dict[str, float]] = []
        self._max_bits = max(config.discrete_bit_widths)
        self._scale_upper = config.scale_bounds[1]
        self._clip_upper = config.clip_bounds[1]
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

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
            update_summary = self._update_policy(batch_records)
            rewards.extend(batch_rewards)
            self.update_index += 1
            history_row = {
                "step": float(self.update_index),
                "episode": float(self.global_episode),
                "batch_reward": mean(batch_rewards),
                **update_summary,
            }
            self.training_history.append(history_row)

        return {
            "episodes": float(len(rewards)),
            "mean_reward": mean(rewards),
            "best_reward": max(rewards) if rewards else 0.0,
            "final_reward": rewards[-1] if rewards else 0.0,
            "updates": float(self.update_index),
        }

    def evaluate(self, episodes: int | None = None, hardware: HardwareType | None = None) -> dict[str, float]:
        accumulator = EvaluationAccumulator()
        previous_action = list(self.previous_action)
        for episode_index in range(episodes or self.config.evaluation_episodes):
            state = self.env.reset(previous_action=previous_action, forced_hardware=hardware)
            state_vector = state.to_vector(self.ordered_hardware)
            decision, _record = self.policy.act(state_vector, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=1_000_000 + episode_index)
            previous_action = self._feedback_vector(result.decision)
            accumulator.add_metrics(result.metrics)

        return accumulator.summary()

    def rollout(self, episodes: int) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        previous_action = list(self.previous_action)
        for episode_index in range(episodes):
            state = self.env.reset(previous_action=previous_action)
            state_vector = state.to_vector(self.ordered_hardware)
            decision, _record = self.policy.act(state_vector, deterministic=True)
            result = self.env.evaluate_current(decision, episode_index=2_000_000 + episode_index)
            previous_action = self._feedback_vector(result.decision)
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
            self.previous_action = self._feedback_vector(result.decision)
            record["state_vector"] = state_vector
            record["reward"] = float(result.metrics.reward)
            records.append(record)
            rewards.append(float(result.metrics.reward))
        return records, rewards

    def _update_policy(self, records: list[dict[str, Any]]) -> dict[str, float]:
        device = self.policy.device
        states = self.policy.state_tensor([record["state_vector"] for record in records])
        rewards = torch.tensor([record["reward"] for record in records], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([record["log_prob"] for record in records], dtype=torch.float32, device=device)
        old_values = torch.tensor([record["value"] for record in records], dtype=torch.float32, device=device)
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        indices = list(range(len(records)))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies_seen: list[float] = []
        ratio_means: list[float] = []
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
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies_seen.append(float(entropy_bonus.detach().item()))
                ratio_means.append(float(ratios.mean().detach().item()))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config.torch_max_grad_norm)
                self.optimizer.step()
        return {
            "policy_loss": mean(policy_losses),
            "value_loss": mean(value_losses),
            "entropy": mean(entropies_seen),
            "ratio_mean": mean(ratio_means),
            "advantage_std": float(advantages.std(unbiased=False).detach().item()),
        }

    def close(self) -> None:
        self.env.logger.close()

    def act_online(self, state, deterministic: bool = False):
        state_vector = state.to_vector(self.ordered_hardware)
        return self.policy.act(state_vector, deterministic=deterministic)

    def update_online(self, updates: list[tuple[dict[str, Any], float]]) -> dict[str, float]:
        records: list[dict[str, Any]] = []
        rewards: list[float] = []
        for record, reward in updates:
            replay_record = dict(record)
            replay_record["reward"] = float(reward)
            records.append(replay_record)
            rewards.append(float(reward))
        if records:
            self._update_policy(records)
        return {
            "batch_size": float(len(records)),
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
            "config": asdict(self.config),
            "model_state": self.policy.snapshot(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_episode": self.global_episode,
            "update_index": self.update_index,
            "previous_action": self.previous_action,
            "training_history": self.training_history,
        }
        torch.save(payload, target)
        return str(target)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.policy.restore(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._optimizer_to_device()
        self.global_episode = int(checkpoint.get("global_episode", 0))
        self.update_index = int(checkpoint.get("update_index", 0))
        self.previous_action = list(checkpoint.get("previous_action", [0.0, 0.0, 0.0]))
        self.training_history = list(checkpoint.get("training_history", []))

    def _optimizer_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.policy.device)

    def _feedback_vector(self, decision) -> list[float]:
        return feedback_vector(
            decision,
            max_bits=self._max_bits,
            scale_upper=self._scale_upper,
            clip_upper=self._clip_upper,
        )
