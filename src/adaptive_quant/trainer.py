from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.base_trainer import TrainerBase, coerce_previous_action
from adaptive_quant.checkpoint_integrity import attach_dict_integrity, verify_dict_integrity
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.policy import PolicyTrace, UniversalQuantizationPolicy
from adaptive_quant.trainer_utils import (
    online_update_summary,
    reward_summary,
    training_row,
)

_PYTHON_CHECKPOINT_FORMAT = 1


class Trainer(TrainerBase):
    """Stdlib RL trainer (REINFORCE-style updates on ``UniversalQuantizationPolicy`` inside ``AdaptiveQuantizationEnv``)."""

    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        super().__init__(config, log_path=log_path)
        self.policy = UniversalQuantizationPolicy(config)
        self.completed_episodes = 0
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train(self) -> dict[str, float]:
        if self.config.continuous_training:
            return self._train_continuous()
        return self._train_fixed()

    def _run_training_episode(self, episode_index: int) -> float:
        state = self.env.reset(
            previous_action=self.previous_action, phase="train", episode_index=episode_index
        )
        decision, trace = self.policy.act(state, deterministic=self.config.rl_train_deterministic())
        result = self.env.evaluate_current(decision, episode_index=episode_index)
        self.policy.update(trace, result.metrics.reward)
        self.previous_action = self._feedback_vector(result.decision)
        self.training_history.append(training_row(float(episode_index), result))
        self.completed_episodes = max(self.completed_episodes, episode_index + 1)
        return result.metrics.reward

    def _train_fixed(self) -> dict[str, float]:
        rewards = [
            self._run_training_episode(i)
            for i in range(self.completed_episodes, self.config.training_episodes)
        ]
        return reward_summary(rewards, updates=len(self.training_history))

    def _train_continuous(self) -> dict[str, float]:
        """Continuous learning: train up to max_training_episodes with periodic eval."""
        rewards: list[float] = []
        target = self.config.max_training_episodes
        eval_interval = self.config.eval_interval
        ckpt_interval = self.config.checkpoint_interval

        for episode_index in range(self.completed_episodes, target):
            rewards.append(self._run_training_episode(episode_index))

            if eval_interval > 0 and (episode_index + 1) % eval_interval == 0:
                recent = rewards[-eval_interval:]
                eval_summary = self.evaluate()
                print(
                    f"[episode {episode_index + 1:,}] "
                    f"recent_reward={mean(recent):.3f}  "
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
        target = _python_checkpoint_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = attach_dict_integrity(
            {
                "format": _PYTHON_CHECKPOINT_FORMAT,
                "run_name": self.config.run_name,
                "completed_episodes": self.completed_episodes,
                "previous_action": self.previous_action,
                "training_history": self.training_history,
                "policy_state": self.policy.checkpoint_state(),
            }
        )
        write_json(str(target), payload)
        return str(target)

    def load_checkpoint(self, path: str) -> None:
        target = _resolve_python_checkpoint_path(path)
        payload = read_json(target, label="Python trainer checkpoint")
        verify_dict_integrity(payload, label="Python trainer checkpoint")
        if int(payload.get("format", 0)) != _PYTHON_CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported Python trainer checkpoint format in {target}")
        policy_state = payload.get("policy_state")
        if not isinstance(policy_state, dict):
            raise RuntimeError(
                f"Refusing to load legacy Python checkpoint {target}: missing serialized policy state."
            )
        self.policy.restore_checkpoint_state(policy_state)
        self.completed_episodes = int(
            payload.get("completed_episodes", len(payload.get("training_history", [])))
        )
        self.previous_action = coerce_previous_action(payload.get("previous_action"))
        self.training_history = list(payload.get("training_history", []))


def build_trainer(config: FrameworkConfig, log_path: str | None = None) -> Trainer:
    """Factory: ``TorchTrainer`` when ``training_backend="pytorch"``, else stdlib ``Trainer``."""
    if config.training_backend == "pytorch":
        from adaptive_quant.torch_trainer import TorchTrainer

        return TorchTrainer(config, log_path=log_path)
    return Trainer(config, log_path=log_path)


def _python_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.suffix == ".json" else p.with_suffix(".json")


def _resolve_python_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    fallback = _python_checkpoint_path(candidate)
    if fallback.is_file():
        return fallback
    return fallback
