from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from adaptive_quant.math_utils import mean
from adaptive_quant.types import EpisodeResult, HardwareType, QuantizationDecision

StateT = TypeVar("StateT")


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
        "latency_ms_per_token": float(result.metrics.latency_ms_per_token),
    }


def collect_episode_results(
    episodes: int,
    *,
    initial_previous_action: list[float],
    reset: Callable[..., StateT],
    act: Callable[[StateT], QuantizationDecision],
    evaluate_current: Callable[[QuantizationDecision, int], EpisodeResult],
    feedback: Callable[[QuantizationDecision], list[float]],
    episode_offset: int = 0,
    hardware: HardwareType | None = None,
    phase: str = "train",
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    previous_action = list(initial_previous_action)
    for episode_index in range(episodes):
        state = reset(
            previous_action=previous_action,
            forced_hardware=hardware,
            phase=phase,
            episode_index=episode_offset + episode_index,
        )
        decision = act(state)
        result = evaluate_current(decision, episode_offset + episode_index)
        previous_action = feedback(result.decision)
        results.append(result)
    return results


def _mean_metric(results: list[EpisodeResult], attr: str) -> float:
    return mean([float(getattr(result.metrics, attr)) for result in results])


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, float]:
    return {
        "mean_reward": _mean_metric(results, "reward"),
        "mean_perplexity": _mean_metric(results, "perplexity"),
        "mean_latency_ms": _mean_metric(results, "latency_ms"),
        "mean_throughput_tps": _mean_metric(results, "throughput_tps"),
        "mean_memory_mb": _mean_metric(results, "memory_mb"),
        "mean_tokens_processed": _mean_metric(results, "tokens_processed"),
        "mean_latency_ms_per_token": _mean_metric(results, "latency_ms_per_token"),
        "mean_stability_penalty": _mean_metric(results, "stability_penalty"),
        "mean_swap_cost_ms": _mean_metric(results, "swap_cost_ms"),
        "mean_cache_miss_count": _mean_metric(results, "cache_miss_count"),
        "mean_variant_churn": _mean_metric(results, "variant_churn"),
    }


def reward_summary(rewards: list[float], *, updates: int) -> dict[str, float]:
    return {
        "episodes": float(len(rewards)),
        "mean_reward": mean(rewards),
        "best_reward": max(rewards) if rewards else 0.0,
        "final_reward": rewards[-1] if rewards else 0.0,
        "updates": float(updates),
    }


def online_update_summary(rewards: list[float]) -> dict[str, float]:
    return {
        "batch_size": float(len(rewards)),
        "mean_reward": mean(rewards),
    }
