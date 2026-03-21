from __future__ import annotations

import random
from dataclasses import replace

from adaptive_quant.backend import LlamaCppBackend, SimulatorBackend
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.features import estimate_layer_sensitivity, extract_input_features
from adaptive_quant.hardware import default_hardware_profiles
from adaptive_quant.logging_utils import JsonlLogger
from adaptive_quant.math_utils import variance
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.quantization import finalize_decision, safe_fallback_decision
from adaptive_quant.types import EpisodeMetrics, EpisodeResult, EpisodeState, HardwareType, PromptSample, QuantizationDecision


class AdaptiveQuantizationEnv:
    def __init__(self, config: FrameworkConfig, log_path: str | None = None) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.prompt_library = PromptLibrary()
        self.hardware_profiles = default_hardware_profiles()
        self.backend = LlamaCppBackend(config) if config.backend == "llama_cpp" else SimulatorBackend(config)
        self.logger = JsonlLogger(log_path or f"{config.log_dir}/{config.run_name}.jsonl")
        self.current_state: EpisodeState | None = None
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
    ) -> EpisodeState:
        hardware = forced_hardware or self._sample_hardware()
        prompt = self._sample_prompt(forced_prompt_id, forced_prompt)
        previous = previous_action or [0.0, 0.0, 0.0]
        input_features, sensitivity = self._get_prompt_context(prompt)
        self.current_state = EpisodeState(
            hardware_profile=self.hardware_profiles[hardware],
            prompt=prompt,
            input_features=input_features,
            sensitivity=sensitivity,
            previous_action=previous,
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
        stability_penalty = self._stability_penalty(decision, self.current_state)

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
        )
        result = EpisodeResult(state=self.current_state, decision=finalized, metrics=metrics)
        if log_episode:
            self._log_episode(result, episode_index)
        return result

    def _sample_hardware(self) -> HardwareType:
        hardware_modes = self.config.ordered_hardware()
        if not self.config.multi_hardware:
            return hardware_modes[0]
        return hardware_modes[self.rng.randrange(len(hardware_modes))]

    def _sample_prompt(self, forced_prompt_id: str | None = None, forced_prompt: PromptSample | None = None):
        if forced_prompt is not None:
            return forced_prompt
        if forced_prompt_id is None:
            return self.prompt_library.sample(self.rng)
        return self.prompt_library.by_id(forced_prompt_id)

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
        return (
            -weights.alpha_latency * metrics["latency_ms"]
            + weights.beta_throughput * metrics["throughput_tps"]
            - weights.gamma_perplexity * metrics["perplexity"]
            - weights.delta_memory * metrics["memory_mb"]
            - weights.epsilon_instability * stability_penalty
        )

    def _stability_penalty(self, decision: QuantizationDecision, state: EpisodeState) -> float:
        if self.config.stability_probe_count <= 1:
            return 0.0
        perplexities = []
        probes = self.prompt_library.probes(state.prompt, self.config.stability_probe_count, self.rng)
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
            "run_name": self.config.run_name,
            "hardware_mode": result.state.hardware_profile.hardware_type.value,
            "prompt_id": result.state.prompt.prompt_id,
            "prompt_domain": result.state.prompt.domain,
            "input_features": result.state.input_features,
            "sensitivity": result.state.sensitivity,
            "decision": result.decision,
            "metrics": result.metrics,
        }
        self.logger.log(record)
