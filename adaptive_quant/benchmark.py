from __future__ import annotations

import gc

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import write_json
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.trainer import build_trainer
from adaptive_quant.trainer_utils import feedback_vector, summarize_episode_results
from adaptive_quant.types import HardwareType, QuantizationDecision, QuantMode


class BenchmarkSuite:
    """Head-to-head eval scenarios (multi-hardware, static vs dynamic, discrete vs learned, optional MoE suites)."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        results = {
            "single_vs_multi": self._single_vs_multi_hardware(),
            "static_vs_dynamic": self._static_vs_dynamic(),
            "discrete_vs_learned": self._discrete_vs_learned(),
            "static_baselines": self._static_baselines(),
        }
        if self.config.moe_enabled:
            results["dense_vs_moe"] = self._dense_vs_moe()
            results["moe_packed_vs_single_variant"] = self._moe_packed_vs_single_variant()
            results["moe_static_vs_rl"] = self._moe_static_vs_rl()
        write_json(f"{self.config.benchmark_dir}/{self.config.run_name}_benchmarks.json", results)
        return results

    def _static_baselines(self) -> dict[str, object]:
        """
        Non-RL baselines evaluated without training (simple heuristics).
        These are useful for sanity checks and for more credible comparisons.
        """
        config = self._benchmark_config()

        def nearest_allowed(bits: float) -> int:
            allowed = list(config.discrete_bit_widths)
            return min(allowed, key=lambda b: abs(float(b) - float(bits)))

        def always_safe(_state) -> QuantizationDecision:
            return QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=int(config.safe_default_bits))

        def hardware_preferred(state) -> QuantizationDecision:
            return QuantizationDecision(
                mode=QuantMode.DISCRETE,
                base_bit_width=nearest_allowed(state.hardware_profile.preferred_bits),
            )

        def complexity_aware(state) -> QuantizationDecision:
            # Lower bits for low complexity, higher bits for high complexity.
            score = float(state.input_features.complexity_score)
            if score < 0.35:
                bits = min(config.discrete_bit_widths)
            elif score < 0.85:
                bits = nearest_allowed(4.0)
            else:
                bits = max(config.discrete_bit_widths)
            return QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=int(bits))

        baselines = {
            "always_safe": always_safe,
            "hardware_preferred": hardware_preferred,
            "complexity_aware": complexity_aware,
        }

        per_baseline: dict[str, object] = {}
        for name, act_fn in baselines.items():
            per_baseline[name] = self._evaluate_heuristic(config, act_fn)
        return per_baseline

    def _evaluate_heuristic(self, config: FrameworkConfig, act_fn) -> dict[str, float]:
        env = AdaptiveQuantizationEnv(config, log_path=f"{config.log_dir}/{config.run_name}_heuristic.jsonl")
        try:
            results = []
            previous_action = [0.0, 0.0, 0.0]
            max_bits = max(config.discrete_bit_widths)
            scale_upper = config.scale_bounds[1]
            clip_upper = config.clip_bounds[1]
            for episode_index in range(config.evaluation_episodes):
                state = env.reset(
                    previous_action=previous_action, phase="eval", episode_index=episode_index
                )
                decision = act_fn(state)
                finalized = finalize_decision(decision, state, config)
                result = env.evaluate_current(finalized, episode_index=3_000_000 + episode_index)
                previous_action = feedback_vector(
                    result.decision,
                    max_bits=max_bits,
                    scale_upper=scale_upper,
                    clip_upper=clip_upper,
                )
                results.append(result)
            return summarize_episode_results(results)
        finally:
            env.logger.close()

    def _single_vs_multi_hardware(self) -> dict[str, object]:
        train, per_hardware = self._run_variants(
            {
                "single_gpu_policy": self._variant(
                    "single_gpu",
                    multi_hardware=False,
                    hardware_modes=("gpu",),
                    quant_mode=QuantMode.HYBRID.value,
                ),
                "multi_hardware_policy": self._variant(
                    "multi_hw",
                    multi_hardware=True,
                    hardware_modes=("gpu", "cpu", "low_resource"),
                    quant_mode=QuantMode.HYBRID.value,
                ),
            },
            per_hardware=(HardwareType.GPU, HardwareType.CPU, HardwareType.LOW_RESOURCE),
        )
        single_policy_gap = self._generalization_gap(per_hardware, "single_gpu_policy")
        multi_policy_gap = self._generalization_gap(per_hardware, "multi_hardware_policy")
        return {
            "train": train,
            "per_hardware": per_hardware,
            "single_policy_gap": single_policy_gap,
            "multi_policy_gap": multi_policy_gap,
            "generalization_gap_improvement": single_policy_gap - multi_policy_gap,
        }

    def _static_vs_dynamic(self) -> dict[str, object]:
        return self._compare_variants(
            {
                "static": self._variant(
                    "static",
                    dynamic_quant=False,
                    learned_quant=False,
                    quant_mode=QuantMode.DISCRETE.value,
                ),
                "dynamic": self._variant(
                    "dynamic",
                    dynamic_quant=True,
                    learned_quant=False,
                    quant_mode=QuantMode.DYNAMIC.value,
                ),
            },
            deltas={"quality_variance_delta": ("mean_stability_penalty", "static", "dynamic")},
        )

    def _discrete_vs_learned(self) -> dict[str, object]:
        return self._compare_variants(
            {
                "discrete": self._variant(
                    "discrete",
                    dynamic_quant=False,
                    learned_quant=False,
                    quant_mode=QuantMode.PER_LAYER.value,
                ),
                "learned": self._variant(
                    "learned",
                    dynamic_quant=True,
                    learned_quant=True,
                    quant_mode=QuantMode.LEARNED.value,
                ),
            },
            deltas={"reward_delta": ("mean_reward", "discrete", "learned")},
        )

    def _dense_vs_moe(self) -> dict[str, object]:
        return self._compare_variants(
            {
                "dense": self._variant(
                    "dense_adaptive",
                    moe_enabled=False,
                ),
                "moe": self._variant(
                    "moe_adaptive",
                    moe_enabled=True,
                ),
            },
            deltas={
                "reward_delta": ("mean_reward", "dense", "moe"),
                "swap_cost_delta": ("mean_swap_cost_ms", "dense", "moe"),
            },
        )

    def _moe_packed_vs_single_variant(self) -> dict[str, object]:
        return self._compare_variants(
            {
                "single_variant": self._variant(
                    "moe_single_variant",
                    moe_enabled=True,
                    moe_variant_names=("balanced",),
                    moe_fixed_variant="balanced",
                ),
                "packed_variant_bank": self._variant(
                    "moe_packed",
                    moe_enabled=True,
                    moe_variant_names=self.config.moe_variant_names,
                    moe_fixed_variant=None,
                ),
            },
            deltas={"reward_delta": ("mean_reward", "single_variant", "packed_variant_bank")},
        )

    def _moe_static_vs_rl(self) -> dict[str, object]:
        return self._compare_variants(
            {
                "static_policy": self._variant(
                    "moe_static",
                    moe_enabled=True,
                    moe_fixed_variant="balanced",
                ),
                "rl_policy": self._variant(
                    "moe_rl",
                    moe_enabled=True,
                    moe_fixed_variant=None,
                ),
            },
            deltas={
                "reward_delta": ("mean_reward", "static_policy", "rl_policy"),
                "cache_miss_delta": ("mean_cache_miss_count", "static_policy", "rl_policy"),
            },
        )

    def _benchmark_config(self, **overrides: object) -> FrameworkConfig:
        benchmark_training = (
            self.config.training_episodes
            if self.config.benchmark_training_episodes is None
            else self.config.benchmark_training_episodes
        )
        benchmark_eval = (
            self.config.evaluation_episodes
            if self.config.benchmark_evaluation_episodes is None
            else self.config.benchmark_evaluation_episodes
        )
        return self.config.clone(
            training_episodes=benchmark_training,
            evaluation_episodes=benchmark_eval,
            **overrides,
        )

    def _variant(self, suffix: str, **overrides: object) -> FrameworkConfig:
        return self._benchmark_config(run_name=f"{self.config.run_name}_{suffix}", **overrides)

    def _release_trainer(self, trainer) -> None:
        close = getattr(trainer, "close", None)
        if callable(close):
            close()
        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_variants(
        self,
        variants: dict[str, FrameworkConfig],
        *,
        per_hardware: tuple[HardwareType, ...] | None = None,
    ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        trainers = {name: build_trainer(config) for name, config in variants.items()}
        try:
            train = {name: trainer.train() for name, trainer in trainers.items()}
            if per_hardware is None:
                evaluation: dict[str, object] = {name: trainer.evaluate() for name, trainer in trainers.items()}
            else:
                evaluation = {
                    hardware.value: {name: trainer.evaluate(hardware=hardware) for name, trainer in trainers.items()}
                    for hardware in per_hardware
                }
            return train, evaluation
        finally:
            for trainer in trainers.values():
                self._release_trainer(trainer)

    def _compare_variants(
        self,
        variants: dict[str, FrameworkConfig],
        *,
        deltas: dict[str, tuple[str, str, str]],
    ) -> dict[str, object]:
        train, evaluation = self._run_variants(variants)
        result: dict[str, object] = {"train": train, "evaluation": evaluation}
        for name, (metric, left, right) in deltas.items():
            result[name] = evaluation[right][metric] - evaluation[left][metric]
        return result

    @staticmethod
    def _generalization_gap(per_hardware: dict[str, object], policy_name: str) -> float:
        gpu_reward = per_hardware[HardwareType.GPU.value][policy_name]["mean_reward"]
        unseen_reward = (
            per_hardware[HardwareType.CPU.value][policy_name]["mean_reward"]
            + per_hardware[HardwareType.LOW_RESOURCE.value][policy_name]["mean_reward"]
        ) / 2.0
        return gpu_reward - unseen_reward
