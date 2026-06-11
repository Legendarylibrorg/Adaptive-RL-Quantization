from __future__ import annotations

from typing import Protocol

from adaptive_quant.types import BackendMetricDict


class _MoEPenaltyConfig(Protocol):
    moe_swap_penalty: float
    moe_cache_miss_penalty: float
    moe_variant_churn_penalty: float


class _RewardWeights(Protocol):
    alpha_latency: float
    beta_throughput: float
    gamma_perplexity: float
    delta_memory: float
    epsilon_instability: float
    eta_token_latency: float
    zeta_perplexity_over_ref: float


def compute_weighted_reward(
    *,
    reward_weights: _RewardWeights,
    metrics: BackendMetricDict,
    stability_penalty: float = 0.0,
    perplexity_reference: float | None = None,
    include_instability: bool = True,
    latency_ms_per_token_default: float = 0.0,
) -> float:
    """Shared reward helper used by multiple pipelines.

    Keeps reward math consistent across the environment trainer and route research while
    letting callers opt out of instability terms when they are not applicable.
    """

    weights = reward_weights
    reward = (
        -weights.alpha_latency * float(metrics["latency_ms"])
        + weights.beta_throughput * float(metrics["throughput_tps"])
        - weights.gamma_perplexity * float(metrics["perplexity"])
        - weights.delta_memory * float(metrics["memory_mb"])
        - weights.eta_token_latency
        * float(metrics.get("latency_ms_per_token", latency_ms_per_token_default))
    )
    if include_instability:
        reward -= weights.epsilon_instability * float(stability_penalty)

    ref = perplexity_reference
    zeta = float(weights.zeta_perplexity_over_ref)
    if ref is not None and zeta > 0.0:
        over = max(0.0, float(metrics["perplexity"]) - float(ref))
        reward -= zeta * over

    return float(reward)


def apply_moe_reward_penalties(
    reward: float,
    metrics: BackendMetricDict,
    config: _MoEPenaltyConfig,
) -> float:
    """Subtract MoE swap/cache/churn terms shared by env and other reward call sites."""
    adjusted = float(reward)
    adjusted -= config.moe_swap_penalty * float(metrics.get("swap_cost_ms", 0.0))
    adjusted -= config.moe_cache_miss_penalty * float(metrics.get("cache_miss_count", 0.0))
    adjusted -= config.moe_variant_churn_penalty * float(metrics.get("variant_churn", 0.0))
    return float(adjusted)


__all__ = ["apply_moe_reward_penalties", "compute_weighted_reward"]
