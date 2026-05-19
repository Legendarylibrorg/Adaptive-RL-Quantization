from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig


def should_fallback_due_to_instability(stability_penalty: float, *, threshold: float) -> bool:
    return float(stability_penalty) > float(threshold)


def passes_online_guardrails(
    *,
    config: FrameworkConfig,
    candidate_reward: float,
    baseline_reward: float,
    candidate_latency_ms: float,
    baseline_latency_ms: float,
    candidate_memory_mb: float,
    baseline_memory_mb: float,
    candidate_perplexity: float,
    baseline_perplexity: float,
    candidate_unstable: bool = False,
    candidate_fallback_applied: bool = False,
) -> bool:
    """Centralized guardrail checks for online learning acceptance.

    This matches the existing OnlineLearningLoop semantics; it is intentionally strict and
    returns only a boolean so call sites can keep their own telemetry.
    """

    if candidate_unstable or candidate_fallback_applied:
        return False
    if float(candidate_reward) < float(baseline_reward) - float(config.online_reward_guard):
        return False
    if float(candidate_latency_ms) > float(baseline_latency_ms) * float(
        config.online_max_latency_ratio
    ):
        return False
    if float(candidate_memory_mb) > float(baseline_memory_mb) * float(
        config.online_max_memory_ratio
    ):
        return False
    if float(candidate_perplexity) > float(baseline_perplexity) + float(
        config.online_max_perplexity_delta
    ):
        return False
    return True


__all__ = ["passes_online_guardrails", "should_fallback_due_to_instability"]
