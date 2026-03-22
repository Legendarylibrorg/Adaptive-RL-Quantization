from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_quant.math_utils import mean
from adaptive_quant.types import EpisodeMetrics, EpisodeResult, QuantizationDecision


@dataclass
class EvaluationAccumulator:
    rewards: list[float] = field(default_factory=list)
    perplexities: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    memories: list[float] = field(default_factory=list)
    stabilities: list[float] = field(default_factory=list)
    swap_costs: list[float] = field(default_factory=list)
    cache_misses: list[float] = field(default_factory=list)
    variant_churns: list[float] = field(default_factory=list)

    def add_metrics(self, metrics: EpisodeMetrics) -> None:
        self.rewards.append(float(metrics.reward))
        self.perplexities.append(float(metrics.perplexity))
        self.latencies.append(float(metrics.latency_ms))
        self.throughputs.append(float(metrics.throughput_tps))
        self.memories.append(float(metrics.memory_mb))
        self.stabilities.append(float(metrics.stability_penalty))
        self.swap_costs.append(float(metrics.swap_cost_ms))
        self.cache_misses.append(float(metrics.cache_miss_count))
        self.variant_churns.append(float(metrics.variant_churn))

    def summary(self) -> dict[str, float]:
        return {
            "mean_reward": mean(self.rewards),
            "mean_perplexity": mean(self.perplexities),
            "mean_latency_ms": mean(self.latencies),
            "mean_throughput_tps": mean(self.throughputs),
            "mean_memory_mb": mean(self.memories),
            "mean_stability_penalty": mean(self.stabilities),
            "mean_swap_cost_ms": mean(self.swap_costs),
            "mean_cache_miss_count": mean(self.cache_misses),
            "mean_variant_churn": mean(self.variant_churns),
        }


def feedback_vector(
    decision: QuantizationDecision,
    *,
    max_bits: int,
    scale_upper: float,
    clip_upper: float,
) -> list[float]:
    return decision.feedback_vector(
        max_bits=max_bits,
        scale_upper=scale_upper,
        clip_upper=clip_upper,
    )


def training_row(step: float, result: EpisodeResult) -> dict[str, float]:
    return {
        "step": step,
        "reward": float(result.metrics.reward),
        "latency_ms": float(result.metrics.latency_ms),
        "throughput_tps": float(result.metrics.throughput_tps),
        "perplexity": float(result.metrics.perplexity),
    }
