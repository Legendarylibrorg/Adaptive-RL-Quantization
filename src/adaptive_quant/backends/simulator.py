from __future__ import annotations

from adaptive_quant.backends.protocol import per_token_latency_fields
from adaptive_quant.backends.quality import ExternalQualityScores, apply_external_quality
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import clamp, mean, variance
from adaptive_quant.moe import ExpertBank
from adaptive_quant.types import (
    BackendMetricDict,
    EpisodeState,
    HardwareType,
    QuantizationDecision,
    QuantMode,
)


class SimulatorBackend:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.expert_bank = ExpertBank(config) if config.moe_enabled else None
        self.external_quality = ExternalQualityScores.from_config(config)

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict:
        hardware = state.hardware_profile
        avg_bits = mean(decision.effective_layer_bits)
        bit_variance = variance(decision.effective_layer_bits)
        complexity = state.input_features.complexity_score
        sensitivity = mean(state.sensitivity.layer_stats)
        prompt_length = max(8, state.input_features.prompt_length)
        compression = max(0.0, (8.0 - avg_bits) / 6.0)

        mode_bonus = {
            QuantMode.DISCRETE: 0.10,
            QuantMode.GROUPED: 0.16,
            QuantMode.PER_LAYER: 0.18,
            QuantMode.DYNAMIC: 0.28,
            QuantMode.LEARNED: 0.34,
        }[decision.mode]

        latency_ms = (
            8.5
            * prompt_length
            * hardware.latency_bias
            / max(0.35, hardware.compute_factor + (8.0 - avg_bits) * 0.12 + mode_bonus)
        )
        latency_ms *= 1.0 + complexity * 0.55 + max(0.0, bit_variance - hardware.kernel_uniformity_preference) * 0.18

        throughput_tps = (
            140.0
            * hardware.throughput_bias
            * (1.0 + (8.0 - avg_bits) * 0.10 + mode_bonus * 0.40)
            / (1.0 + complexity * 0.80 + hardware.latency_bias * 0.08)
        )
        if hardware.hardware_type == HardwareType.GPU:
            throughput_tps *= 1.0 - min(0.12, bit_variance * 0.03)
        else:
            throughput_tps *= 1.0 + min(0.10, max(0.0, hardware.preferred_bits - avg_bits) * 0.02)

        memory_mb = 4_800.0 * (avg_bits / 16.0) * (1.0 + complexity * 0.15)
        if decision.mode in {QuantMode.PER_LAYER, QuantMode.LEARNED}:
            memory_mb *= 1.02

        perplexity = (
            5.6
            + complexity * 3.4
            + max(0.0, 5.5 - avg_bits) * (0.60 + complexity * 0.90 + sensitivity * 0.35)
            + abs(1.0 - decision.scale_factor) * 0.65
            + max(0.0, 1.05 - decision.clipping_range) * 1.20
            - mode_bonus * 0.70
        )

        hardware_alignment = abs(avg_bits - hardware.preferred_bits)
        latency_ms *= 1.0 + hardware_alignment * 0.04
        throughput_tps *= 1.0 - hardware_alignment * 0.02
        perplexity += hardware_alignment * 0.15

        if hardware.hardware_type in {HardwareType.CPU, HardwareType.LOW_RESOURCE} and avg_bits > hardware.preferred_bits:
            excess_bits = avg_bits - hardware.preferred_bits
            latency_ms *= 1.0 + excess_bits * (0.16 if hardware.hardware_type == HardwareType.CPU else 0.24)
            throughput_tps *= max(0.55, 1.0 - excess_bits * (0.07 if hardware.hardware_type == HardwareType.CPU else 0.12))
            memory_mb *= 1.0 + excess_bits * (0.10 if hardware.hardware_type == HardwareType.CPU else 0.18)
        elif hardware.hardware_type == HardwareType.GPU and avg_bits < hardware.preferred_bits:
            deficit_bits = hardware.preferred_bits - avg_bits
            perplexity += deficit_bits * 0.45
            throughput_tps *= max(0.78, 1.0 - deficit_bits * 0.03)

        if decision.mode == QuantMode.DYNAMIC:
            latency_ms *= 0.92
            throughput_tps *= 1.06
            perplexity -= 0.25 + complexity * 0.20
        elif decision.mode == QuantMode.LEARNED:
            latency_ms *= 0.82 - compression * 0.06
            throughput_tps *= 1.12 + compression * 0.08
            memory_mb *= 0.78 - compression * 0.04
            perplexity -= 0.38 + sensitivity * 0.22
        elif decision.mode == QuantMode.GROUPED and hardware.hardware_type != HardwareType.GPU:
            latency_ms *= 0.95
            throughput_tps *= 1.03

        overflow_ratio = max(0.0, memory_mb - hardware.memory_budget_mb) / hardware.memory_budget_mb
        if overflow_ratio > 0.0:
            latency_ms *= 1.0 + overflow_ratio * 2.50
            throughput_tps *= 1.0 / (1.0 + overflow_ratio * 1.8)
            perplexity += overflow_ratio * 1.50

        swap_cost_ms = 0.0
        cache_miss_count = 0.0
        variant_churn = float(decision.metadata.get("moe_variant_churn", 0.0))
        if self.expert_bank is not None and state.moe_context is not None and decision.moe_variant_indices:
            latency_ms, throughput_tps, perplexity, memory_mb, swap_cost_ms, cache_miss_count = self._apply_moe_adjustments(
                state,
                decision,
                latency_ms,
                throughput_tps,
                perplexity,
                memory_mb,
            )

        metrics: BackendMetricDict = {
            "latency_ms": clamp(latency_ms, 5.0, 20_000.0),
            "throughput_tps": clamp(throughput_tps, 1.0, 10_000.0),
            "perplexity": clamp(perplexity, 3.0, 100.0),
            "memory_mb": clamp(memory_mb, 200.0, 128_000.0),
            "swap_cost_ms": swap_cost_ms,
            "cache_miss_count": cache_miss_count,
            "variant_churn": variant_churn,
        }
        calibration = getattr(self.config, "sim_calibration", None)
        if isinstance(calibration, dict):
            hw_key = state.hardware_profile.hardware_type.value
            hw_cal = calibration.get(hw_key, {}) if isinstance(calibration.get(hw_key, {}), dict) else {}
            latency_mul = float(hw_cal.get("latency_multiplier", 1.0))
            throughput_mul = float(hw_cal.get("throughput_multiplier", 1.0))
            memory_mul = float(hw_cal.get("memory_multiplier", 1.0))
            if latency_mul > 0:
                metrics["latency_ms"] = clamp(metrics["latency_ms"] * latency_mul, 1.0, 60_000.0)
            if throughput_mul > 0:
                metrics["throughput_tps"] = clamp(metrics["throughput_tps"] * throughput_mul, 0.1, 100_000.0)
            if memory_mul > 0:
                metrics["memory_mb"] = clamp(metrics["memory_mb"] * memory_mul, 50.0, 512_000.0)
        metrics.update(per_token_latency_fields(state, metrics["latency_ms"]))
        metrics.update(
            {
                "latency_source": "simulator",
                "throughput_source": "simulator",
                "memory_source": "simulator",
                "perplexity_source": "simulator",
            }
        )
        apply_external_quality(metrics, state, self.external_quality)
        return metrics

    def _apply_moe_adjustments(
        self,
        state: EpisodeState,
        decision: QuantizationDecision,
        latency_ms: float,
        throughput_tps: float,
        perplexity: float,
        memory_mb: float,
    ) -> tuple[float, float, float, float, float, float]:
        assert self.expert_bank is not None
        total_swap_cost = 0.0
        cache_misses = 0.0
        throughput_multiplier = 1.0
        memory_multiplier = 1.0
        sensitivity_penalty = 0.0
        latency_multiplier = 1.0

        for expert, variant_index in zip(state.moe_context.experts, decision.moe_variant_indices):
            variant = self.expert_bank.variant_by_index(variant_index)
            routing_weight = 0.60 + expert.router_probability
            latency_multiplier *= 1.0 + (variant.latency_multiplier - 1.0) * routing_weight * 0.50
            throughput_multiplier *= 1.0 + (variant.throughput_multiplier - 1.0) * routing_weight * 0.55
            memory_multiplier *= 1.0 + (variant.memory_multiplier - 1.0) * routing_weight * 0.40
            sensitivity_penalty += variant.perplexity_penalty * expert.sensitivity * routing_weight
            if expert.resident_on_device < 0.5:
                cache_misses += 1.0
                total_swap_cost += variant.swap_cost_ms * (1.0 + expert.router_probability) * (1.10 - 0.35 * expert.hotness)

        latency_ms = latency_ms * latency_multiplier + total_swap_cost
        throughput_tps *= throughput_multiplier
        memory_mb *= memory_multiplier
        perplexity += sensitivity_penalty
        return latency_ms, throughput_tps, perplexity, memory_mb, total_swap_cost, cache_misses
