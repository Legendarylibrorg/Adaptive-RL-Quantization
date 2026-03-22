from __future__ import annotations

import gc

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import write_json
from adaptive_quant.torch_policy import torch
from adaptive_quant.trainer import build_trainer
from adaptive_quant.types import HardwareType, QuantMode


class BenchmarkSuite:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        results = {
            "single_vs_multi": self._single_vs_multi_hardware(),
            "static_vs_dynamic": self._static_vs_dynamic(),
            "discrete_vs_learned": self._discrete_vs_learned(),
        }
        if self.config.moe_enabled:
            results["dense_vs_moe"] = self._dense_vs_moe()
            results["moe_packed_vs_single_variant"] = self._moe_packed_vs_single_variant()
            results["moe_static_vs_rl"] = self._moe_static_vs_rl()
        write_json(f"{self.config.benchmark_dir}/{self.config.run_name}_benchmarks.json", results)
        return results

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
        benchmark_training = self.config.benchmark_training_episodes or self.config.training_episodes
        benchmark_eval = self.config.benchmark_evaluation_episodes or self.config.evaluation_episodes
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
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
