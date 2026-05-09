from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import (
    clamp,
    deterministic_float,
    softmax,
    stable_hash_int,
)
from adaptive_quant.types import (
    HardwareProfile,
    InputFeatures,
    MoEContext,
    MoEExpertState,
    PromptSample,
)


@dataclass(frozen=True)
class PackedExpertVariant:
    name: str
    aggressiveness: float
    latency_multiplier: float
    throughput_multiplier: float
    memory_multiplier: float
    perplexity_penalty: float
    swap_cost_ms: float


class ExpertBank:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.variants = self._build_variants(config.moe_variant_names)

    def _build_variants(self, names: tuple[str, ...]) -> list[PackedExpertVariant]:
        preset = {
            "safe": PackedExpertVariant(
                name="safe",
                aggressiveness=0.10,
                latency_multiplier=1.02,
                throughput_multiplier=0.99,
                memory_multiplier=1.03,
                perplexity_penalty=0.04,
                swap_cost_ms=1.1,
            ),
            "balanced": PackedExpertVariant(
                name="balanced",
                aggressiveness=0.50,
                latency_multiplier=0.96,
                throughput_multiplier=1.04,
                memory_multiplier=0.94,
                perplexity_penalty=0.18,
                swap_cost_ms=1.8,
            ),
            "aggressive": PackedExpertVariant(
                name="aggressive",
                aggressiveness=1.00,
                latency_multiplier=0.88,
                throughput_multiplier=1.10,
                memory_multiplier=0.82,
                perplexity_penalty=0.42,
                swap_cost_ms=2.9,
            ),
        }
        variants: list[PackedExpertVariant] = []
        for name in names:
            variants.append(preset.get(name, PackedExpertVariant(name, 0.5, 0.96, 1.03, 0.94, 0.20, 1.8)))
        return variants

    def variant_by_index(self, index: int) -> PackedExpertVariant:
        bounded = max(0, min(len(self.variants) - 1, index))
        return self.variants[bounded]

    def build_context(self, prompt: PromptSample, input_features: InputFeatures, hardware_profile: HardwareProfile) -> MoEContext:
        top_k = max(1, min(self.config.moe_top_k, self.config.moe_num_experts))
        router_scores: list[tuple[int, float]] = []
        for expert_index in range(self.config.moe_num_experts):
            score = self._router_score(prompt, input_features, hardware_profile, expert_index)
            router_scores.append((expert_index, score))
        selected = sorted(router_scores, key=lambda item: item[1], reverse=True)[:top_k]
        probabilities = softmax([score for _expert_index, score in selected])
        experts: list[MoEExpertState] = []
        for (expert_index, _score), probability in zip(selected, probabilities):
            hotness = self._expert_hotness(prompt, hardware_profile, expert_index)
            resident = self._resident_score(hardware_profile, expert_index, hotness)
            sensitivity = clamp(
                0.35
                + 0.55 * input_features.complexity_score
                + 0.35 * probability
                + deterministic_float(f"expert:sensitivity:{prompt.prompt_id}:{expert_index}", -0.08, 0.08),
                0.0,
                1.5,
            )
            experts.append(
                MoEExpertState(
                    expert_index=expert_index,
                    router_probability=probability,
                    sensitivity=sensitivity,
                    resident_on_device=resident,
                    hotness=hotness,
                    available_variants_mask=[1.0] * len(self.variants),
                )
            )
        router_entropy = 0.0
        for probability in probabilities:
            router_entropy -= probability * math.log(probability + 1e-9, 2)
        max_entropy = math.log(max(len(probabilities), 2), 2)
        normalized_entropy = clamp(router_entropy / max_entropy if max_entropy > 0.0 else 0.0, 0.0, 1.0)
        cache_pressure = clamp(
            (top_k / max(1, self.config.moe_gpu_resident_experts))
            + input_features.complexity_score * 0.45
            + max(0.0, 0.65 - sum(expert.resident_on_device for expert in experts) / max(1, len(experts))) * 0.40,
            0.0,
            1.5,
        )
        estimated_swap_cost_ms = sum(
            (1.0 - expert.resident_on_device) * (1.4 + 3.2 * expert.router_probability + 1.1 * (1.0 - expert.hotness))
            for expert in experts
        )
        return MoEContext(
            router_entropy=normalized_entropy,
            active_expert_count=float(len(experts)),
            cache_pressure=cache_pressure,
            estimated_swap_cost_ms=estimated_swap_cost_ms,
            experts=experts,
            top_k=top_k,
            variant_count=len(self.variants),
            num_experts=self.config.moe_num_experts,
        )

    def _router_score(self, prompt: PromptSample, input_features: InputFeatures, hardware_profile: HardwareProfile, expert_index: int) -> float:
        prompt_bias = deterministic_float(f"router:{prompt.prompt_id}:{expert_index}", 0.0, 1.0)
        hardware_bias = deterministic_float(f"router:{hardware_profile.hardware_type.value}:{expert_index}", 0.0, 1.0)
        domain_bucket = stable_hash_int(f"{prompt.domain}:{expert_index}", modulo=7) / 6.0
        return (
            0.40 * prompt_bias
            + 0.20 * hardware_bias
            + 0.20 * domain_bucket
            + 0.20 * input_features.complexity_score
        )

    def _expert_hotness(self, prompt: PromptSample, hardware_profile: HardwareProfile, expert_index: int) -> float:
        return clamp(
            0.35
            + 0.45 * deterministic_float(f"hot:{hardware_profile.hardware_type.value}:{expert_index}", 0.0, 1.0)
            + 0.20 * deterministic_float(f"hot-prompt:{prompt.prompt_id}:{expert_index}", 0.0, 1.0),
            0.0,
            1.0,
        )

    def _resident_score(self, hardware_profile: HardwareProfile, expert_index: int, hotness: float) -> float:
        if hardware_profile.hardware_type.value == "gpu":
            if expert_index < self.config.moe_gpu_resident_experts:
                return 1.0
            return 1.0 if hotness >= 0.78 else 0.0
        if hardware_profile.hardware_type.value == "cpu":
            return 0.45 if expert_index < max(1, self.config.moe_gpu_resident_experts // 2) else 0.15
        return 0.15 if expert_index < max(1, self.config.moe_gpu_resident_experts // 3) else 0.0
