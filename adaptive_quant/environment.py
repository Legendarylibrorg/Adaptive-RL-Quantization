from __future__ import annotations

import random
from dataclasses import replace

from adaptive_quant.backend import LlamaCppBackend, SimulatorBackend
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.features import estimate_layer_sensitivity, extract_input_features
from adaptive_quant.hardware import detect_host_hardware, host_aware_hardware_profiles
from adaptive_quant.logging_utils import JsonlLogger, NullJsonlLogger
from adaptive_quant.math_utils import variance
from adaptive_quant.moe import ExpertBank
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.quantization import finalize_decision, safe_fallback_decision
from adaptive_quant.trainer_utils import zero_previous_action
from adaptive_quant.types import (
    EpisodeMetrics,
    EpisodeResult,
    EpisodeState,
    HardwareType,
    PromptSample,
    QuantizationDecision,
)


class AdaptiveQuantizationEnv:
    """One-step RL interface: reset (prompt + hardware + features) → policy act → backend measures → reward.

    ``SimulatorBackend`` is the default measurable world; ``LlamaCppBackend`` delegates to your **llama.cpp**
    binary when ``config.backend="llama_cpp"``. MoE adds ``ExpertBank`` routing and penalties when enabled.
    Episodes append structured rows to JSONL for downstream ``analysis/`` tools.
    """

    def __init__(
        self,
        config: FrameworkConfig,
        log_path: str | None = None,
        *,
        enable_logging: bool = True,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.prompt_library = PromptLibrary()
        self._prompt_split_rng = random.Random(config.prompt_split_seed)
        self.train_prompt_ids: set[str] | None = None
        self.eval_prompt_ids: set[str] | None = None
        if config.prompt_split_enabled:
            self.train_prompt_ids, self.eval_prompt_ids = self.prompt_library.split_ids(
                rng=self._prompt_split_rng,
                train_fraction=config.prompt_train_fraction,
            )
        self.detected_hardware = detect_host_hardware() if config.detect_host_hardware else None
        self.hardware_profiles = host_aware_hardware_profiles(self.detected_hardware)
        self.backend = LlamaCppBackend(config) if config.backend == "llama_cpp" else SimulatorBackend(config)
        self.expert_bank = ExpertBank(config) if config.moe_enabled else None
        self.logger = (
            JsonlLogger(log_path or f"{config.log_dir}/{config.run_name}.jsonl")
            if enable_logging
            else NullJsonlLogger()
        )
        self.current_state: EpisodeState | None = None
        self._current_phase: str = "train"
        self._prompt_cache: dict[str, tuple] = {}
        if config.cache_prompt_features:
            for prompt in self.prompt_library.prompts:
                self._prompt_cache[prompt.prompt_id] = self._build_prompt_context(prompt)

    def reset(
        self,
        previous_action: list[float] | None = None,
        forced_hardware: HardwareType | None = None,
        forced_prompt_id: str | None = None,
        forced_prompt: PromptSample | None = None,
        phase: str = "train",
        episode_index: int | None = None,
    ) -> EpisodeState:
        mode = self.config.env_sampling_mode.strip().lower()
        ep = 0 if episode_index is None else int(episode_index)

        if forced_prompt is not None:
            prompt = forced_prompt
        elif mode == "sequential":
            pid = forced_prompt_id or self._sequential_prompt_id(ep, phase)
            prompt = self.prompt_library.by_id(pid)
        elif mode == "forced":
            pid = forced_prompt_id or self.config.env_forced_prompt_id
            if pid is None:
                raise ValueError(
                    "env_sampling_mode='forced' requires forced_prompt, forced_prompt_id, or env_forced_prompt_id"
                )
            prompt = self.prompt_library.by_id(pid)
        else:
            prompt = self._sample_prompt_random(forced_prompt_id, phase=phase)

        if forced_hardware is not None:
            hardware = forced_hardware
        elif mode == "sequential":
            hardware = self._sequential_hardware(ep)
        elif mode == "forced":
            raw_hw = self.config.env_forced_hardware
            if raw_hw is None:
                raise ValueError(
                    "env_sampling_mode='forced' requires forced_hardware on reset() or env_forced_hardware in config"
                )
            hardware = HardwareType(raw_hw)
        else:
            hardware = self._sample_hardware_random()

        previous = previous_action or zero_previous_action()
        input_features, sensitivity = self._get_prompt_context(prompt)
        hardware_profile = self.hardware_profiles[hardware]
        self._current_phase = phase
        self.current_state = EpisodeState(
            hardware_profile=hardware_profile,
            prompt=prompt,
            input_features=input_features,
            sensitivity=sensitivity,
            previous_action=previous,
            moe_context=self.expert_bank.build_context(prompt, input_features, hardware_profile) if self.expert_bank is not None else None,
        )
        return self.current_state

    def evaluate_current(
        self,
        decision: QuantizationDecision,
        episode_index: int | None = None,
        log_episode: bool = True,
    ) -> EpisodeResult:
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before evaluation.")

        finalized = finalize_decision(decision, self.current_state, self.config)
        primary_metrics = self.backend.evaluate(self.current_state, finalized)
        stability_penalty = self._stability_penalty(finalized, self.current_state)

        if stability_penalty > self.config.instability_threshold:
            fallback = finalize_decision(safe_fallback_decision(self.config), self.current_state, self.config)
            fallback.fallback_applied = True
            fallback.unstable = True
            fallback.metadata["fallback_reason"] = "instability"
            finalized = fallback
            primary_metrics = self.backend.evaluate(self.current_state, finalized)
            stability_penalty = self._stability_penalty(finalized, self.current_state)

        reward = self._compute_reward(primary_metrics, stability_penalty)
        metrics = EpisodeMetrics(
            latency_ms=primary_metrics["latency_ms"],
            throughput_tps=primary_metrics["throughput_tps"],
            perplexity=primary_metrics["perplexity"],
            memory_mb=primary_metrics["memory_mb"],
            stability_penalty=stability_penalty,
            reward=reward,
            tokens_processed=primary_metrics.get("tokens_processed", 0.0),
            latency_ms_per_token=primary_metrics.get("latency_ms_per_token", 0.0),
            swap_cost_ms=primary_metrics.get("swap_cost_ms", 0.0),
            cache_miss_count=primary_metrics.get("cache_miss_count", 0.0),
            variant_churn=primary_metrics.get("variant_churn", 0.0),
        )
        result = EpisodeResult(state=self.current_state, decision=finalized, metrics=metrics)
        if log_episode:
            self._log_episode(result, episode_index)
        return result

    def _sequential_prompt_id(self, episode_index: int, phase: str) -> str:
        if self.config.prompt_split_enabled:
            split_ids = self.train_prompt_ids if phase == "train" else self.eval_prompt_ids
            allowed = split_ids or {p.prompt_id for p in self.prompt_library.prompts}
        else:
            allowed = {p.prompt_id for p in self.prompt_library.prompts}
        ordered = sorted(allowed)
        return ordered[episode_index % len(ordered)]

    def _sequential_hardware(self, episode_index: int) -> HardwareType:
        modes = self.config.ordered_hardware()
        if not self.config.multi_hardware:
            return modes[0]
        return modes[episode_index % len(modes)]

    def _sample_hardware_random(self) -> HardwareType:
        hardware_modes = self.config.ordered_hardware()
        if not self.config.multi_hardware:
            return hardware_modes[0]
        return hardware_modes[self.rng.randrange(len(hardware_modes))]

    def _sample_prompt_random(self, forced_prompt_id: str | None, *, phase: str = "train") -> PromptSample:
        if forced_prompt_id is not None:
            return self.prompt_library.by_id(forced_prompt_id)
        if not self.config.prompt_split_enabled:
            return self.prompt_library.sample(self.rng)
        allowed = self.train_prompt_ids if phase == "train" else self.eval_prompt_ids
        if not allowed:
            return self.prompt_library.sample(self.rng)
        ordered_ids = sorted(allowed)
        prompt_id = ordered_ids[self.rng.randrange(len(ordered_ids))]
        return self.prompt_library.by_id(prompt_id)

    def _get_prompt_context(self, prompt):
        if prompt.prompt_id in self._prompt_cache:
            return self._prompt_cache[prompt.prompt_id]
        context = self._build_prompt_context(prompt)
        if self.config.cache_prompt_features:
            self._prompt_cache[prompt.prompt_id] = context
        return context

    def _build_prompt_context(self, prompt):
        input_features = extract_input_features(prompt)
        sensitivity = estimate_layer_sensitivity(prompt, input_features, self.config.num_layers)
        return input_features, sensitivity

    def _compute_reward(self, metrics: dict[str, float], stability_penalty: float) -> float:
        weights = self.config.reward_weights
        reward = (
            -weights.alpha_latency * metrics["latency_ms"]
            + weights.beta_throughput * metrics["throughput_tps"]
            - weights.gamma_perplexity * metrics["perplexity"]
            - weights.delta_memory * metrics["memory_mb"]
            - weights.epsilon_instability * stability_penalty
            - weights.eta_token_latency * metrics.get("latency_ms_per_token", 0.0)
            - self.config.moe_swap_penalty * metrics.get("swap_cost_ms", 0.0)
            - self.config.moe_cache_miss_penalty * metrics.get("cache_miss_count", 0.0)
            - self.config.moe_variant_churn_penalty * metrics.get("variant_churn", 0.0)
        )
        ref = self.config.reward_perplexity_reference
        zeta = weights.zeta_perplexity_over_ref
        if ref is not None and zeta > 0.0:
            over = max(0.0, float(metrics["perplexity"]) - float(ref))
            reward -= zeta * over
        return reward

    def _stability_penalty(self, decision: QuantizationDecision, state: EpisodeState) -> float:
        if self.config.stability_probe_count <= 1:
            return 0.0
        perplexities = []
        allowed = None
        if self.config.prompt_split_enabled:
            # Keep probes within the same split as the active prompt.
            if self.eval_prompt_ids is not None and state.prompt.prompt_id in self.eval_prompt_ids:
                allowed = self.eval_prompt_ids
            elif self.train_prompt_ids is not None:
                allowed = self.train_prompt_ids
        if self.config.stability_probe_sampling.strip().lower() == "deterministic":
            probes = self.prompt_library.probes_deterministic(
                state.prompt, self.config.stability_probe_count, allowed_ids=allowed
            )
        else:
            probes = self.prompt_library.probes(state.prompt, self.config.stability_probe_count, self.rng, allowed_ids=allowed)
        for probe in probes:
            probe_features, probe_sensitivity = self._get_prompt_context(probe)
            probe_state = replace(
                state,
                prompt=probe,
                input_features=probe_features,
                sensitivity=probe_sensitivity,
            )
            probe_decision = finalize_decision(replace(decision), probe_state, self.config)
            metrics = self.backend.evaluate(probe_state, probe_decision)
            perplexities.append(metrics["perplexity"])
        return variance(perplexities)

    def _log_episode(self, result: EpisodeResult, episode_index: int | None) -> None:
        if episode_index is not None and self.config.log_every_n_episodes > 1:
            if episode_index % self.config.log_every_n_episodes != 0:
                return
        record = {
            "episode": episode_index,
            "phase": self._current_phase,
            "run_name": self.config.run_name,
            "hardware_mode": result.state.hardware_profile.hardware_type.value,
            "prompt_id": result.state.prompt.prompt_id,
            "prompt_domain": result.state.prompt.domain,
            "input_features": result.state.input_features,
            "sensitivity": result.state.sensitivity,
            "moe_context": result.state.moe_context,
            "decision": result.decision,
            "metrics": result.metrics,
        }
        self.logger.log(record)
