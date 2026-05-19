from __future__ import annotations

import math
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.trainer_utils import (
    collect_episode_results,
    feedback_vector,
    summarize_episode_results,
    zero_previous_action,
)
from adaptive_quant.types import EpisodeResult, HardwareType

# Length of ``previous_action`` feedback vector emitted by ``feedback_vector``
# (bits / scale / clip). Persisted in checkpoints; validated on resume.
_PREVIOUS_ACTION_LEN = 3


def coerce_previous_action(value: Any) -> list[float]:
    """Validate a ``previous_action`` payload loaded from a checkpoint.

    A malicious or truncated checkpoint could otherwise inject NaN/Inf or a
    differently-sized list, which silently propagates into the next state vector.
    """
    if value is None:
        return [0.0] * _PREVIOUS_ACTION_LEN
    if not isinstance(value, list):
        raise TypeError(
            f"previous_action must be a list of {_PREVIOUS_ACTION_LEN} finite floats, "
            f"got {type(value).__name__}"
        )
    if len(value) != _PREVIOUS_ACTION_LEN:
        raise ValueError(
            f"previous_action must have length {_PREVIOUS_ACTION_LEN}, got {len(value)}"
        )
    coerced: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"previous_action[{index}] must be numeric, got {type(item).__name__}")
        f = float(item)
        if not math.isfinite(f):
            raise ValueError(f"previous_action[{index}] must be finite, got {f!r}")
        coerced.append(f)
    return coerced


class TrainerBase:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        self.config = config
        self.env = AdaptiveQuantizationEnv(config, log_path=log_path)
        self.previous_action = zero_previous_action()
        self.training_history: list[dict[str, float]] = []
        self._next_eval_episode = 1_000_000
        self._max_bits = max(config.discrete_bit_widths)
        self._scale_upper = config.scale_bounds[1]
        self._clip_upper = config.clip_bounds[1]

    def _policy_input(self, state):
        return state

    def _policy_act(self, state, *, deterministic: bool):
        return self.policy.act(self._policy_input(state), deterministic=deterministic)

    def _collect_episodes(
        self,
        episodes: int,
        *,
        episode_offset: int,
        hardware: HardwareType | None = None,
        phase: str = "train",
    ) -> list[EpisodeResult]:
        return collect_episode_results(
            episodes,
            initial_previous_action=self.previous_action,
            reset=self.env.reset,
            act=lambda state: self._policy_act(state, deterministic=True)[0],
            evaluate_current=self.env.evaluate_current,
            feedback=self._feedback_vector,
            episode_offset=episode_offset,
            hardware=hardware,
            phase=phase,
        )

    def evaluate(
        self, episodes: int | None = None, hardware: HardwareType | None = None
    ) -> dict[str, float]:
        count = self.config.evaluation_episodes if episodes is None else episodes
        episode_offset = self._next_eval_episode
        self._next_eval_episode += max(0, int(count))
        results = self._collect_episodes(
            count,
            episode_offset=episode_offset,
            hardware=hardware,
            phase="eval",
        )
        return summarize_episode_results(results)

    def rollout(self, episodes: int) -> list[EpisodeResult]:
        return self._collect_episodes(episodes, episode_offset=2_000_000)

    def close(self) -> None:
        self.env.logger.close()

    def act_online(self, state, deterministic: bool = False):
        return self._policy_act(state, deterministic=deterministic)

    def snapshot_policy(self):
        return self.policy.snapshot()

    def restore_policy(self, snapshot) -> None:
        self.policy.restore(snapshot)

    def _feedback_vector(self, decision) -> list[float]:
        return feedback_vector(
            decision,
            max_bits=self._max_bits,
            scale_upper=self._scale_upper,
            clip_upper=self._clip_upper,
        )
