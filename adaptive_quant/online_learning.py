from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import JsonlLogger, to_jsonable, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.types import HardwareType, OnlineRequest, PromptSample


@dataclass
class ReplayEntry:
    payload: Any
    reward: float
    served_reward: float
    prompt_id: str
    hardware_mode: str
    accepted: bool
    canary: bool
    explore: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = max(1, capacity)
        self.rng = rng
        self._entries: deque[ReplayEntry] = deque(maxlen=self.capacity)

    def add(self, entry: ReplayEntry) -> None:
        self._entries.append(entry)

    def sample(self, count: int) -> list[ReplayEntry]:
        if not self._entries:
            return []
        sample_size = min(max(1, count), len(self._entries))
        return self.rng.sample(list(self._entries), sample_size)

    def __len__(self) -> int:
        return len(self._entries)


class OnlineLearningLoop:
    def __init__(self, config: FrameworkConfig, trainer=None) -> None:
        from adaptive_quant.trainer import build_trainer

        self.config = config
        self.trainer = trainer or build_trainer(config)
        self._owns_trainer = trainer is None
        self.rng = random.Random(config.seed + 1201)
        self.prompt_library = PromptLibrary()
        self.telemetry_logger = JsonlLogger(config.online_telemetry_path())
        self.replay_logger = JsonlLogger(config.online_replay_path())
        self.replay_buffer = ReplayBuffer(config.online_replay_capacity, self.rng)
        self.previous_action = [0.0, 0.0, 0.0]
        self.pending_experiences = 0
        self.request_index = 0
        self.safe_mode_remaining = 0
        self.total_updates = 0
        self.total_rollbacks = 0
        self.total_explorations = 0
        self.total_canaries = 0
        self.total_candidate_accepts = 0
        self.total_candidate_rejects = 0
        self.recent_served_rewards: deque[float] = deque(maxlen=max(4, config.online_drift_window))
        self.best_recent_reward = float("-inf")
        self.best_policy_snapshot = self.trainer.snapshot_policy()

    def serve_request(self, request: OnlineRequest) -> dict[str, Any]:
        prompt = self._request_prompt(request)
        safe_mode_active = self.safe_mode_remaining > 0
        explore = bool(self.config.online_learning and not safe_mode_active and self.rng.random() < self.config.online_exploration_rate)
        canary = bool(explore and self.rng.random() < self.config.online_canary_ratio)

        state = self.trainer.env.reset(
            previous_action=self.previous_action,
            forced_hardware=request.hardware,
            forced_prompt=prompt,
        )
        baseline_decision, _baseline_payload = self.trainer.act_online(state, deterministic=True)

        if explore:
            self.total_explorations += 1
            candidate_decision, candidate_payload = self.trainer.act_online(state, deterministic=False)
        else:
            candidate_decision, candidate_payload = baseline_decision, None

        drift_event = "steady"
        update_summary = None
        baseline_result = None
        candidate_result = self.trainer.env.evaluate_current(
            candidate_decision,
            episode_index=10_000_000 + self.request_index,
            log_episode=False,
        )
        accepted_candidate = True

        if canary:
            self.total_canaries += 1
            baseline_result = self.trainer.env.evaluate_current(
                baseline_decision,
                episode_index=20_000_000 + self.request_index,
                log_episode=False,
            )
            accepted_candidate = self._passes_guardrails(candidate_result, baseline_result)
            if accepted_candidate:
                self.total_candidate_accepts += 1
            else:
                self.total_candidate_rejects += 1
        elif explore:
            self.total_candidate_accepts += 1

        served_result = candidate_result if accepted_candidate or baseline_result is None else baseline_result
        self.previous_action = served_result.decision.feedback_vector(
            max_bits=max(self.config.discrete_bit_widths),
            scale_upper=self.config.scale_bounds[1],
            clip_upper=self.config.clip_bounds[1],
        )
        self.trainer.previous_action = list(self.previous_action)

        if explore and candidate_payload is not None:
            entry = ReplayEntry(
                payload=candidate_payload,
                reward=float(candidate_result.metrics.reward),
                served_reward=float(served_result.metrics.reward),
                prompt_id=prompt.prompt_id,
                hardware_mode=request.hardware.value,
                accepted=accepted_candidate,
                canary=canary,
                explore=explore,
            )
            self.replay_buffer.add(entry)
            self.pending_experiences += 1
            self.replay_logger.log(
                {
                    "request_index": self.request_index,
                    "prompt_id": prompt.prompt_id,
                    "hardware_mode": request.hardware.value,
                    "candidate_metrics": candidate_result.metrics,
                    "served_metrics": served_result.metrics,
                    "accepted_candidate": accepted_candidate,
                    "canary": canary,
                    "safe_mode_active": safe_mode_active,
                    "update_payload": to_jsonable(candidate_payload),
                }
            )

        if self.safe_mode_remaining > 0:
            self.safe_mode_remaining -= 1

        update_summary = self._maybe_update_policy()
        drift_event = self._maybe_handle_drift(served_result.metrics.reward)
        self.request_index += 1

        telemetry = {
            "request_index": self.request_index - 1,
            "run_name": self.config.run_name,
            "hardware_mode": request.hardware.value,
            "prompt_id": prompt.prompt_id,
            "prompt_domain": prompt.domain,
            "input_features": state.input_features,
            "decision": served_result.decision,
            "served_metrics": served_result.metrics,
            "candidate_metrics": candidate_result.metrics,
            "baseline_metrics": baseline_result.metrics if baseline_result is not None else None,
            "explore": explore,
            "canary": canary,
            "accepted_candidate": accepted_candidate,
            "safe_mode_active": safe_mode_active,
            "online_update_applied": update_summary is not None,
            "online_update_summary": update_summary,
            "drift_event": drift_event,
            "replay_size": len(self.replay_buffer),
        }
        self.telemetry_logger.log(telemetry)
        return telemetry

    def run_stream(self, requests: list[OnlineRequest]) -> dict[str, Any]:
        records = [self.serve_request(request) for request in requests]
        served_rewards = [float(record["served_metrics"].reward) for record in records]
        candidate_rewards = [float(record["candidate_metrics"].reward) for record in records if record.get("candidate_metrics") is not None]
        summary = {
            "requests": len(records),
            "mean_served_reward": mean(served_rewards),
            "mean_candidate_reward": mean(candidate_rewards),
            "exploration_rate_observed": self.total_explorations / max(1, len(records)),
            "canary_rate_observed": self.total_canaries / max(1, len(records)),
            "candidate_accept_rate": self.total_candidate_accepts / max(1, self.total_candidate_accepts + self.total_candidate_rejects),
            "total_updates": self.total_updates,
            "total_rollbacks": self.total_rollbacks,
            "replay_size": len(self.replay_buffer),
            "telemetry_path": self.config.online_telemetry_path(),
            "replay_path": self.config.online_replay_path(),
        }
        write_json(f"{self.config.benchmark_dir}/{self.config.run_name}_online_summary.json", summary)
        return summary

    def close(self) -> None:
        self.telemetry_logger.close()
        self.replay_logger.close()
        if self._owns_trainer:
            self.trainer.close()

    def refresh_best_snapshot(self) -> None:
        self.best_policy_snapshot = self.trainer.snapshot_policy()
        self.best_recent_reward = float("-inf")
        self.recent_served_rewards.clear()

    def _request_prompt(self, request: OnlineRequest) -> PromptSample:
        if request.prompt_id is not None:
            try:
                library_prompt = self.prompt_library.by_id(request.prompt_id)
            except KeyError:
                pass
            else:
                return library_prompt

        prompt_id = request.prompt_id or f"online_{self.request_index:06d}"
        return PromptSample(prompt_id=prompt_id, text=request.prompt_text, domain=request.prompt_domain)

    def _passes_guardrails(self, candidate_result, baseline_result) -> bool:
        if candidate_result.decision.unstable or candidate_result.decision.fallback_applied:
            return False
        if candidate_result.metrics.reward < baseline_result.metrics.reward - self.config.online_reward_guard:
            return False
        if candidate_result.metrics.latency_ms > baseline_result.metrics.latency_ms * self.config.online_max_latency_ratio:
            return False
        if candidate_result.metrics.memory_mb > baseline_result.metrics.memory_mb * self.config.online_max_memory_ratio:
            return False
        if candidate_result.metrics.perplexity > baseline_result.metrics.perplexity + self.config.online_max_perplexity_delta:
            return False
        return True

    def _maybe_update_policy(self) -> dict[str, float] | None:
        if not self.config.online_learning:
            return None
        if len(self.replay_buffer) < self.config.online_min_replay_size:
            return None
        if self.pending_experiences < self.config.online_update_interval:
            return None

        sampled = self.replay_buffer.sample(self.config.online_batch_size)
        updates = [(entry.payload, entry.reward) for entry in sampled]
        summary = self.trainer.update_online(updates)
        self.total_updates += 1
        self.pending_experiences = 0
        return summary

    def _maybe_handle_drift(self, served_reward: float) -> str:
        self.recent_served_rewards.append(float(served_reward))
        if len(self.recent_served_rewards) < self.recent_served_rewards.maxlen:
            return "warming_up"

        recent_mean = mean(list(self.recent_served_rewards))
        if recent_mean > self.best_recent_reward:
            self.best_recent_reward = recent_mean
            self.best_policy_snapshot = self.trainer.snapshot_policy()
            return "improved"

        if recent_mean < self.best_recent_reward - self.config.online_drift_reward_delta:
            self.trainer.restore_policy(self.best_policy_snapshot)
            self.safe_mode_remaining = self.config.online_safe_mode_cooldown
            self.total_rollbacks += 1
            self.recent_served_rewards.clear()
            return "rollback"
        return "steady"


def build_request_stream(config: FrameworkConfig, request_count: int | None = None) -> list[OnlineRequest]:
    rng = random.Random(config.seed + 1409)
    library = PromptLibrary()
    hardware_options = config.ordered_hardware()
    count = request_count or config.online_requests
    requests: list[OnlineRequest] = []
    for index in range(count):
        prompt = library.prompts[rng.randrange(len(library.prompts))]
        hardware = hardware_options[rng.randrange(len(hardware_options))]
        requests.append(
            OnlineRequest(
                prompt_text=prompt.text,
                hardware=hardware,
                prompt_id=f"{prompt.prompt_id}_{index:04d}",
                prompt_domain=prompt.domain,
            )
        )
    return requests


__all__ = [
    "OnlineLearningLoop",
    "ReplayBuffer",
    "build_request_stream",
]
