from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path
from typing import Any

from adaptive_quant.base_trainer import TrainerBase, coerce_previous_action
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.torch_policy import (
    TORCH_BACKEND_REQUIRED_MESSAGE,
    TORCH_IMPORT_ERROR,
    TorchPolicyAdapter,
    torch,
)
from adaptive_quant.trainer_utils import online_update_summary, reward_summary

_CHECKPOINT_FORMAT_V2 = 2


def _crossed_episode_milestone(episode_before: int, episode_after: int, interval: int) -> bool:
    """True once ``episode_after`` passes a multiple of ``interval`` (stdlib trainer semantics)."""
    if interval <= 0:
        return False
    return (episode_before // interval) < (episode_after // interval)


def _checkpoint_meta_path(pt_path: str) -> str:
    """Sidecar JSON for v2 checkpoints (tensor .pt + metadata)."""
    p = Path(pt_path)
    return str(p.with_name(f"{p.stem}.checkpoint.json"))


def _torch_load_v2_tensor_file(path: str) -> dict[str, Any]:
    """
    Load v2 tensor shard: only model_state + optimizer_state tensors.
    Prefer weights_only=True when PyTorch supports it to avoid arbitrary pickle execution.
    """
    load_kw: dict[str, Any] = {"map_location": "cpu"}
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            load_kw["weights_only"] = True
    except (TypeError, ValueError):
        pass
    return torch.load(path, **load_kw)


if torch is not None:

    class _GPUReplayBuffer:
        """Ring buffer of transitions on a torch device (VRAM when CUDA)."""

        def __init__(
            self,
            capacity: int,
            state_dim: int,
            device: torch.device,
            *,
            deterministic_sampling: bool = False,
            sampling_seed: int = 0,
        ) -> None:
            self.capacity = capacity
            self.device = device
            self.size = 0
            self.cursor = 0
            self.states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.records: list[dict[str, Any] | None] = [None] * capacity
            self._sample_gen: torch.Generator | None = None
            if deterministic_sampling:
                gen = torch.Generator(device=device)
                gen.manual_seed(int(sampling_seed) + 17_713)
                self._sample_gen = gen

        def push(
            self,
            state_vector: list[float],
            reward: float,
            log_prob: float,
            value: float,
            record: dict[str, Any],
        ) -> None:
            idx = self.cursor % self.capacity
            self.states[idx] = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
            self.rewards[idx] = reward
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.records[idx] = record
            self.cursor += 1
            self.size = min(self.size + 1, self.capacity)

        def push_batch(self, records_batch: list[dict[str, Any]]) -> None:
            if not records_batch:
                return

            # Batch tensor updates to reduce Python overhead and device sync points.
            state_vectors = [rec["state_vector"] for rec in records_batch]
            rewards = [rec["reward"] for rec in records_batch]
            log_probs = [rec["log_prob"] for rec in records_batch]
            values = [rec["value"] for rec in records_batch]

            batch_states = torch.tensor(state_vectors, dtype=torch.float32, device=self.device)
            batch_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            batch_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
            batch_values = torch.tensor(values, dtype=torch.float32, device=self.device)

            n = int(batch_states.shape[0])
            base = int(self.cursor % self.capacity)
            indices = (torch.arange(n, device=self.device) + base) % self.capacity

            self.states[indices] = batch_states
            self.rewards[indices] = batch_rewards
            self.log_probs[indices] = batch_log_probs
            self.values[indices] = batch_values

            for offset, rec in enumerate(records_batch):
                idx = int((base + offset) % self.capacity)
                self.records[idx] = rec

            self.cursor += n
            self.size = min(self.size + n, self.capacity)

        def sample(
            self, batch_size: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
            n = min(batch_size, self.size)
            if self._sample_gen is not None:
                indices = torch.randint(
                    0, self.size, (n,), device=self.device, generator=self._sample_gen
                )
            else:
                indices = torch.randint(0, self.size, (n,), device=self.device)
            return (
                self.states[indices],
                self.rewards[indices],
                self.log_probs[indices],
                self.values[indices],
                [self.records[int(i)] for i in indices.cpu().tolist()],
            )

        def vram_bytes(self) -> int:
            total = 0
            for t in (self.states, self.rewards, self.log_probs, self.values):
                total += t.nelement() * t.element_size()
            return total

    class TorchTrainer(TrainerBase):
        def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
            if torch is None:
                raise ImportError(TORCH_BACKEND_REQUIRED_MESSAGE) from TORCH_IMPORT_ERROR
            super().__init__(config, log_path=log_path)
            self.policy = TorchPolicyAdapter(config)
            self.optimizer = self._build_optimizer()
            self.rng = random.Random(config.seed + 401)
            self.global_episode = 0
            self.update_index = 0
            self.ordered_hardware = config.ordered_hardware()
            self.replay_buffer: Any = None
            if config.replay_buffer_capacity > 0:
                device = self.policy.device if config.replay_buffer_on_gpu else torch.device("cpu")
                self.replay_buffer = _GPUReplayBuffer(
                    capacity=config.replay_buffer_capacity,
                    state_dim=config.state_vector_dim(),
                    device=device,
                    deterministic_sampling=config.torch_deterministic,
                    sampling_seed=config.seed,
                )
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
            if self.config.continuous_training:
                return self._train_continuous()
            return self._train_fixed()

        def _commit_training_batch(
            self,
            batch_records: list[dict[str, Any]],
            batch_rewards: list[float],
            rewards_accum: list[float] | None = None,
        ) -> None:
            if self.replay_buffer is not None:
                self.replay_buffer.push_batch(batch_records)
            update_summary = self._update_from_replay_or_batch(batch_records)
            self.update_index += 1
            self.training_history.append(
                {
                    "step": float(self.update_index),
                    "episode": float(self.global_episode),
                    "batch_reward": mean(batch_rewards),
                    **update_summary,
                    **self._vram_stats(),
                }
            )
            if rewards_accum is not None:
                rewards_accum.extend(batch_rewards)

        def _train_fixed(self) -> dict[str, float]:
            rewards: list[float] = []
            while self.global_episode < self.config.training_episodes:
                batch_size = min(
                    self.config.torch_batch_episodes,
                    self.config.training_episodes - self.global_episode,
                )
                batch_records, batch_rewards = self._collect_batch(batch_size)
                self._commit_training_batch(batch_records, batch_rewards, rewards)

            return reward_summary(rewards, updates=self.update_index)

        def _train_continuous(self) -> dict[str, float]:
            """Train until max_training_episodes with periodic eval and checkpoint."""
            all_rewards: list[float] = []
            target = self.config.max_training_episodes
            eval_interval = self.config.eval_interval
            ckpt_interval = self.config.checkpoint_interval
            batch_episodes = max(1, int(self.config.torch_batch_episodes))
            if eval_interval > 0 and eval_interval < batch_episodes:
                raise ValueError(
                    "FrameworkConfig.eval_interval must be >= torch_batch_episodes "
                    f"(got eval_interval={eval_interval}, torch_batch_episodes={batch_episodes}); "
                    "the modular trigger expects intervals to be at least one full batch."
                )
            if ckpt_interval > 0 and ckpt_interval < batch_episodes:
                raise ValueError(
                    "FrameworkConfig.checkpoint_interval must be >= torch_batch_episodes "
                    f"(got checkpoint_interval={ckpt_interval}, torch_batch_episodes={batch_episodes})."
                )

            while self.global_episode < target:
                batch_size = min(self.config.torch_batch_episodes, target - self.global_episode)
                ep_before = self.global_episode
                batch_records, batch_rewards = self._collect_batch(batch_size)
                self._commit_training_batch(batch_records, batch_rewards, all_rewards)
                ep_after = self.global_episode

                if eval_interval > 0 and _crossed_episode_milestone(
                    ep_before, ep_after, eval_interval
                ):
                    eval_summary = self.evaluate()
                    print(
                        f"[episode {self.global_episode:,}] "
                        f"batch_reward={mean(batch_rewards):.3f}  "
                        f"eval_reward={eval_summary.get('mean_reward', 0):.3f}  "
                        f"{self._vram_summary()}",
                        file=sys.stderr,
                    )

                if ckpt_interval > 0 and _crossed_episode_milestone(
                    ep_before, ep_after, ckpt_interval
                ):
                    ckpt_path = self.config.final_checkpoint_path().replace(
                        "_final", f"_ep{self.global_episode}"
                    )
                    self.save_checkpoint(ckpt_path)

            return reward_summary(all_rewards, updates=self.update_index)

        def _update_from_replay_or_batch(
            self, batch_records: list[dict[str, Any]]
        ) -> dict[str, float]:
            if (
                self.replay_buffer is not None
                and self.replay_buffer.size >= self.config.torch_minibatch_size
            ):
                _, _, _, _, replay_records = self.replay_buffer.sample(
                    max(len(batch_records), self.config.torch_batch_episodes)
                )
                combined = batch_records + [r for r in replay_records if r is not None]
                return self._update_policy(combined)
            return self._update_policy(batch_records)

        def _vram_stats(self) -> dict[str, float]:
            if self.policy.device.type != "cuda":
                return {}
            allocated_mb = torch.cuda.memory_allocated(self.policy.device) / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(self.policy.device) / (1024**2)
            replay_mb = (
                (self.replay_buffer.vram_bytes() / (1024**2))
                if self.replay_buffer is not None
                else 0.0
            )
            return {
                "vram_allocated_mb": round(allocated_mb, 1),
                "vram_reserved_mb": round(reserved_mb, 1),
                "replay_buffer_mb": round(replay_mb, 1),
                "replay_buffer_size": float(self.replay_buffer.size)
                if self.replay_buffer is not None
                else 0.0,
            }

        def _vram_summary(self) -> str:
            stats = self._vram_stats()
            if not stats:
                return "device=cpu"
            return (
                f"vram={stats.get('vram_allocated_mb', 0):.0f}MB "
                f"replay={stats.get('replay_buffer_size', 0):.0f}"
            )

        def _collect_batch(self, batch_size: int) -> tuple[list[dict[str, Any]], list[float]]:
            records: list[dict[str, Any]] = []
            rewards: list[float] = []
            for _ in range(batch_size):
                state = self.env.reset(
                    previous_action=self.previous_action,
                    phase="train",
                    episode_index=self.global_episode,
                )
                state_vector = state.to_vector(self.ordered_hardware)
                decision, record = self.policy.act(
                    state_vector,
                    deterministic=self.config.rl_train_deterministic(),
                    moe_context=state.moe_context,
                )
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
            rewards = torch.tensor(
                [record["reward"] for record in records], dtype=torch.float32, device=device
            )
            old_log_probs = torch.tensor(
                [record["log_prob"] for record in records], dtype=torch.float32, device=device
            )
            old_values = torch.tensor(
                [record["value"] for record in records], dtype=torch.float32, device=device
            )
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

                    log_probs, entropies, values = self.policy.evaluate_actions(
                        state_batch, record_batch
                    )
                    ratios = torch.exp(log_probs - old_log_prob_batch)
                    algo = self.config.torch_policy_algorithm.strip().lower()
                    if algo == "vpg":
                        policy_loss = -(log_probs * advantage_batch).mean()
                    elif algo == "awr":
                        beta = max(float(self.config.torch_awr_beta), 1e-6)
                        raw_w = torch.exp(torch.clamp(advantage_batch / beta, min=-5.0, max=5.0))
                        w = raw_w / (raw_w.mean() + 1e-8)
                        policy_loss = -(w.detach() * log_probs).mean()
                    else:
                        unclipped = ratios * advantage_batch
                        clipped = (
                            torch.clamp(
                                ratios,
                                1.0 - self.config.torch_ppo_clip,
                                1.0 + self.config.torch_ppo_clip,
                            )
                            * advantage_batch
                        )
                        policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(values.float(), reward_batch)
                    entropy_bonus = entropies.mean()
                    loss = (
                        policy_loss
                        + self.config.torch_value_coef * value_loss
                        - self.config.torch_entropy_coef * entropy_bonus
                    )
                    policy_losses.append(float(policy_loss.detach().item()))
                    value_losses.append(float(value_loss.detach().item()))
                    entropies_seen.append(float(entropy_bonus.detach().item()))
                    ratio_means.append(float(ratios.mean().detach().item()))

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.model.parameters(), self.config.torch_max_grad_norm
                    )
                    self.optimizer.step()
            return {
                "policy_loss": mean(policy_losses),
                "value_loss": mean(value_losses),
                "entropy": mean(entropies_seen),
                "ratio_mean": mean(ratio_means),
                "advantage_std": float(advantages.std(unbiased=False).detach().item()),
            }

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
            return online_update_summary(rewards)

        def save_checkpoint(self, path: str) -> str:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            meta_path = _checkpoint_meta_path(str(target))
            meta = {
                "format": _CHECKPOINT_FORMAT_V2,
                "run_name": self.config.run_name,
                "global_episode": self.global_episode,
                "update_index": self.update_index,
                "previous_action": self.previous_action,
                "training_history": self.training_history,
            }
            write_json(meta_path, meta)
            torch.save(
                {
                    "model_state": self.policy.snapshot(),
                    "optimizer_state": self.optimizer.state_dict(),
                },
                target,
            )
            return str(target)

        def load_checkpoint(self, path: str) -> None:
            pt_path = Path(path)
            meta_path = Path(_checkpoint_meta_path(str(pt_path)))
            if meta_path.is_file():
                raw_meta = read_json(meta_path, label="Checkpoint sidecar")
                if int(raw_meta.get("format", 0)) != _CHECKPOINT_FORMAT_V2:
                    raise ValueError(f"Unsupported checkpoint metadata format in {meta_path}")
                tensors = _torch_load_v2_tensor_file(str(pt_path))
                self.policy.restore(tensors["model_state"])
                self.optimizer.load_state_dict(tensors["optimizer_state"])
                self._optimizer_to_device()
                self.global_episode = int(raw_meta.get("global_episode", 0))
                self.update_index = int(raw_meta.get("update_index", 0))
                self.previous_action = coerce_previous_action(raw_meta.get("previous_action"))
                self.training_history = list(raw_meta.get("training_history", []))
                return

            raise RuntimeError(
                f"Refusing to load legacy pickle checkpoint {pt_path}: missing sidecar "
                f"{meta_path.name}. Re-save the checkpoint with a current trainer in a trusted "
                "environment. Pickle-based .pt checkpoint loading is no longer supported here."
            )

        def _optimizer_to_device(self) -> None:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.policy.device)

        def _policy_input(self, state):
            return state.to_vector(self.ordered_hardware)

else:

    class TorchTrainer(TrainerBase):
        def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
            del config, log_path
            raise ImportError(TORCH_BACKEND_REQUIRED_MESSAGE) from TORCH_IMPORT_ERROR
