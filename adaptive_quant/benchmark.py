from __future__ import annotations

import gc

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.hardware import default_hardware_profiles
from adaptive_quant.logging_utils import write_json
from adaptive_quant.torch_policy import torch
from adaptive_quant.trainer import build_trainer
from adaptive_quant.types import HardwareType, QuantMode


class BenchmarkSuite:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.hardware_profiles = default_hardware_profiles()

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
        gpu_only_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_single_gpu",
            multi_hardware=False,
            hardware_modes=("gpu",),
            quant_mode=QuantMode.HYBRID.value,
        )
        multi_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_multi_hw",
            multi_hardware=True,
            hardware_modes=("gpu", "cpu", "low_resource"),
            quant_mode=QuantMode.HYBRID.value,
        )
        gpu_only = build_trainer(gpu_only_config)
        multi = build_trainer(multi_config)
        try:
            gpu_only_train = gpu_only.train()
            multi_train = multi.train()

            per_hardware = {}
            for hardware in (HardwareType.GPU, HardwareType.CPU, HardwareType.LOW_RESOURCE):
                per_hardware[hardware.value] = {
                    "single_gpu_policy": gpu_only.evaluate(hardware=hardware),
                    "multi_hardware_policy": multi.evaluate(hardware=hardware),
                }

            single_seen = per_hardware["gpu"]["single_gpu_policy"]["mean_reward"]
            single_unseen = (
                per_hardware["cpu"]["single_gpu_policy"]["mean_reward"]
                + per_hardware["low_resource"]["single_gpu_policy"]["mean_reward"]
            ) / 2.0
            multi_seen = per_hardware["gpu"]["multi_hardware_policy"]["mean_reward"]
            multi_unseen = (
                per_hardware["cpu"]["multi_hardware_policy"]["mean_reward"]
                + per_hardware["low_resource"]["multi_hardware_policy"]["mean_reward"]
            ) / 2.0
            return {
                "train": {"single_gpu_policy": gpu_only_train, "multi_hardware_policy": multi_train},
                "per_hardware": per_hardware,
                "single_policy_gap": single_seen - single_unseen,
                "multi_policy_gap": multi_seen - multi_unseen,
                "generalization_gap_improvement": (single_seen - single_unseen) - (multi_seen - multi_unseen),
            }
        finally:
            self._release_trainer(gpu_only)
            self._release_trainer(multi)

    def _static_vs_dynamic(self) -> dict[str, object]:
        static_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_static",
            dynamic_quant=False,
            learned_quant=False,
            quant_mode=QuantMode.DISCRETE.value,
        )
        dynamic_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_dynamic",
            dynamic_quant=True,
            learned_quant=False,
            quant_mode=QuantMode.DYNAMIC.value,
        )
        static_trainer = build_trainer(static_config)
        dynamic_trainer = build_trainer(dynamic_config)
        try:
            static_train = static_trainer.train()
            dynamic_train = dynamic_trainer.train()
            static_eval = static_trainer.evaluate()
            dynamic_eval = dynamic_trainer.evaluate()
            return {
                "train": {"static": static_train, "dynamic": dynamic_train},
                "evaluation": {
                    "static": static_eval,
                    "dynamic": dynamic_eval,
                },
                "quality_variance_delta": dynamic_eval["mean_stability_penalty"] - static_eval["mean_stability_penalty"],
            }
        finally:
            self._release_trainer(static_trainer)
            self._release_trainer(dynamic_trainer)

    def _discrete_vs_learned(self) -> dict[str, object]:
        discrete_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_discrete",
            dynamic_quant=False,
            learned_quant=False,
            quant_mode=QuantMode.PER_LAYER.value,
        )
        learned_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_learned",
            dynamic_quant=True,
            learned_quant=True,
            quant_mode=QuantMode.LEARNED.value,
        )
        discrete_trainer = build_trainer(discrete_config)
        learned_trainer = build_trainer(learned_config)
        try:
            discrete_train = discrete_trainer.train()
            learned_train = learned_trainer.train()
            discrete_eval = discrete_trainer.evaluate()
            learned_eval = learned_trainer.evaluate()
            return {
                "train": {"discrete": discrete_train, "learned": learned_train},
                "evaluation": {
                    "discrete": discrete_eval,
                    "learned": learned_eval,
                },
                "reward_delta": learned_eval["mean_reward"] - discrete_eval["mean_reward"],
            }
        finally:
            self._release_trainer(discrete_trainer)
            self._release_trainer(learned_trainer)

    def _dense_vs_moe(self) -> dict[str, object]:
        dense_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_dense_adaptive",
            moe_enabled=False,
        )
        moe_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_moe_adaptive",
            moe_enabled=True,
        )
        dense_trainer = build_trainer(dense_config)
        moe_trainer = build_trainer(moe_config)
        try:
            dense_train = dense_trainer.train()
            moe_train = moe_trainer.train()
            dense_eval = dense_trainer.evaluate()
            moe_eval = moe_trainer.evaluate()
            return {
                "train": {"dense": dense_train, "moe": moe_train},
                "evaluation": {"dense": dense_eval, "moe": moe_eval},
                "reward_delta": moe_eval["mean_reward"] - dense_eval["mean_reward"],
                "swap_cost_delta": moe_eval["mean_swap_cost_ms"] - dense_eval["mean_swap_cost_ms"],
            }
        finally:
            self._release_trainer(dense_trainer)
            self._release_trainer(moe_trainer)

    def _moe_packed_vs_single_variant(self) -> dict[str, object]:
        single_variant_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_moe_single_variant",
            moe_enabled=True,
            moe_variant_names=("balanced",),
            moe_fixed_variant="balanced",
        )
        packed_variant_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_moe_packed",
            moe_enabled=True,
            moe_variant_names=self.config.moe_variant_names,
            moe_fixed_variant=None,
        )
        single_variant_trainer = build_trainer(single_variant_config)
        packed_variant_trainer = build_trainer(packed_variant_config)
        try:
            single_train = single_variant_trainer.train()
            packed_train = packed_variant_trainer.train()
            single_eval = single_variant_trainer.evaluate()
            packed_eval = packed_variant_trainer.evaluate()
            return {
                "train": {"single_variant": single_train, "packed_variant_bank": packed_train},
                "evaluation": {"single_variant": single_eval, "packed_variant_bank": packed_eval},
                "reward_delta": packed_eval["mean_reward"] - single_eval["mean_reward"],
            }
        finally:
            self._release_trainer(single_variant_trainer)
            self._release_trainer(packed_variant_trainer)

    def _moe_static_vs_rl(self) -> dict[str, object]:
        static_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_moe_static",
            moe_enabled=True,
            moe_fixed_variant="balanced",
        )
        rl_config = self._benchmark_config(
            run_name=f"{self.config.run_name}_moe_rl",
            moe_enabled=True,
            moe_fixed_variant=None,
        )
        static_trainer = build_trainer(static_config)
        rl_trainer = build_trainer(rl_config)
        try:
            static_train = static_trainer.train()
            rl_train = rl_trainer.train()
            static_eval = static_trainer.evaluate()
            rl_eval = rl_trainer.evaluate()
            return {
                "train": {"static_policy": static_train, "rl_policy": rl_train},
                "evaluation": {"static_policy": static_eval, "rl_policy": rl_eval},
                "reward_delta": rl_eval["mean_reward"] - static_eval["mean_reward"],
                "cache_miss_delta": rl_eval["mean_cache_miss_count"] - static_eval["mean_cache_miss_count"],
            }
        finally:
            self._release_trainer(static_trainer)
            self._release_trainer(rl_trainer)

    def _benchmark_config(self, **overrides: object) -> FrameworkConfig:
        benchmark_training = self.config.benchmark_training_episodes or self.config.training_episodes
        benchmark_eval = self.config.benchmark_evaluation_episodes or self.config.evaluation_episodes
        return self.config.clone(
            training_episodes=benchmark_training,
            evaluation_episodes=benchmark_eval,
            **overrides,
        )

    def _release_trainer(self, trainer) -> None:
        close = getattr(trainer, "close", None)
        if callable(close):
            close()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
