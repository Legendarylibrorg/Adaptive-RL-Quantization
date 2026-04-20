from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.hardware import resolve_target_hardware
from adaptive_quant.math_utils import mean
from adaptive_quant.trainer_utils import feedback_vector, summarize_episode_results
from adaptive_quant.types import EpisodeResult, HardwareType, QuantizationDecision


def recommend_quantization(trainer, config: FrameworkConfig) -> dict[str, object]:
    detected = getattr(getattr(trainer, "env", None), "detected_hardware", None)
    target_hardware = resolve_target_hardware(config.ordered_hardware(), detected)
    episodes = min(config.recommendation_eval_episodes, config.evaluation_episodes)

    adaptive_results, candidates = _collect_policy_rollout(
        trainer,
        config,
        hardware=target_hardware,
        episodes=episodes,
    )
    adaptive_summary = _summarize_results(adaptive_results)

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda item: (-item["support"], -item["source_mean_reward"]),
    )[: config.recommendation_candidate_limit]

    evaluated_candidates = [
        _evaluate_fixed_candidate(
            config,
            hardware=target_hardware,
            episodes=episodes,
            candidate=item,
        )
        for item in ranked_candidates
    ]
    evaluated_candidates.sort(
        key=lambda item: (
            -float(item["evaluation"]["mean_reward"]),
            float(item["evaluation"]["mean_perplexity"]),
            float(item["evaluation"]["mean_latency_ms"]),
        )
    )

    best_fixed = evaluated_candidates[0] if evaluated_candidates else None
    return {
        "detected_hardware": detected.to_metadata() if detected is not None else None,
        "target_hardware": target_hardware.value,
        "episodes": episodes,
        "adaptive_policy": adaptive_summary,
        "candidate_count": len(evaluated_candidates),
        "recommended_quant": best_fixed,
        "candidates": evaluated_candidates,
    }


def _collect_policy_rollout(
    trainer,
    config: FrameworkConfig,
    *,
    hardware: HardwareType,
    episodes: int,
) -> tuple[list[EpisodeResult], dict[str, dict[str, object]]]:
    support = Counter()
    source_rewards: dict[str, list[float]] = defaultdict(list)
    candidates: dict[str, dict[str, object]] = {}

    def _capture_candidate(state, result: EpisodeResult) -> None:
        signature = _decision_signature(result.decision)
        support[signature] += 1
        source_rewards[signature].append(float(result.metrics.reward))
        if signature not in candidates:
            candidates[signature] = {
                "signature": signature,
                "decision": _decision_payload(result.decision),
                "template": _candidate_template(result.decision),
                "source_prompt_id": state.prompt.prompt_id,
            }

    results = _run_recommendation_rollout(
        config,
        hardware=hardware,
        episodes=episodes,
        episode_offset=4_000_000,
        log_suffix="policy",
        act_fn=lambda state: trainer.act_online(state, deterministic=True)[0],
        on_result=_capture_candidate,
    )

    for signature, item in candidates.items():
        item["support"] = int(support[signature])
        item["source_mean_reward"] = mean(source_rewards[signature])
    return results, candidates


def _evaluate_fixed_candidate(
    config: FrameworkConfig,
    *,
    hardware: HardwareType,
    episodes: int,
    candidate: dict[str, object],
) -> dict[str, object]:
    template = candidate["template"]
    results = _run_recommendation_rollout(
        config,
        hardware=hardware,
        episodes=episodes,
        episode_offset=5_000_000,
        log_suffix="fixed",
        act_fn=lambda _state: deepcopy(template),
    )
    return {
        "signature": candidate["signature"],
        "decision": candidate["decision"],
        "support": candidate["support"],
        "source_prompt_id": candidate["source_prompt_id"],
        "source_mean_reward": candidate["source_mean_reward"],
        "evaluation": _summarize_results(results),
    }


def _run_recommendation_rollout(
    config: FrameworkConfig,
    *,
    hardware: HardwareType,
    episodes: int,
    episode_offset: int,
    log_suffix: str,
    act_fn,
    on_result=None,
) -> list[EpisodeResult]:
    env = AdaptiveQuantizationEnv(
        config,
        enable_logging=False,
    )
    previous_action = [0.0, 0.0, 0.0]
    results: list[EpisodeResult] = []
    try:
        for episode_index in range(episodes):
            state = env.reset(
                previous_action=previous_action,
                forced_hardware=hardware,
                phase="eval",
                episode_index=episode_offset + episode_index,
            )
            result = env.evaluate_current(
                act_fn(state),
                episode_index=episode_offset + episode_index,
                log_episode=False,
            )
            previous_action = _feedback_vector(config, result.decision)
            if on_result is not None:
                on_result(state, result)
            results.append(result)
    finally:
        env.logger.close()
    return results


def _summarize_results(results: list[EpisodeResult]) -> dict[str, object]:
    summary: dict[str, object] = summarize_episode_results(results)
    mode_histogram = Counter(result.decision.mode.value for result in results)
    average_bits = [
        mean(result.decision.effective_layer_bits)
        for result in results
        if result.decision.effective_layer_bits
    ]
    fallback_count = sum(1 for result in results if result.decision.fallback_applied)
    unstable_count = sum(1 for result in results if result.decision.unstable)
    summary["mean_average_bits"] = mean(average_bits)
    summary["fallback_rate"] = fallback_count / len(results) if results else 0.0
    summary["unstable_rate"] = unstable_count / len(results) if results else 0.0
    summary["mode_histogram"] = dict(mode_histogram)
    return summary


def _candidate_template(decision: QuantizationDecision) -> QuantizationDecision:
    return QuantizationDecision(
        mode=decision.mode,
        base_bit_width=decision.base_bit_width,
        group_bit_widths=list(decision.group_bit_widths),
        layer_bit_widths=list(decision.layer_bit_widths),
        scale_factor=float(decision.scale_factor),
        clipping_range=float(decision.clipping_range),
        precision_level=float(decision.precision_level),
        moe_variant_indices=list(decision.moe_variant_indices),
        moe_variant_names=list(decision.moe_variant_names),
        metadata=dict(decision.metadata),
    )


def _feedback_vector(config: FrameworkConfig, decision: QuantizationDecision) -> list[float]:
    return feedback_vector(
        decision,
        max_bits=max(config.discrete_bit_widths),
        scale_upper=config.scale_bounds[1],
        clip_upper=config.clip_bounds[1],
    )


def _decision_signature(decision: QuantizationDecision) -> str:
    payload = _decision_payload(decision)
    bits = payload.get("group_bit_widths") or payload.get("layer_bit_widths") or []
    bits_fragment = ",".join(str(bit) for bit in bits) if bits else "-"
    moe_fragment = ",".join(payload.get("moe_variant_names", [])) or "-"
    return (
        f"{payload['mode']}"
        f"|base={payload.get('base_bit_width')}"
        f"|bits={bits_fragment}"
        f"|scale={payload['scale_factor']:.2f}"
        f"|clip={payload['clipping_range']:.2f}"
        f"|precision={payload['precision_level']:.2f}"
        f"|moe={moe_fragment}"
    )


def _decision_payload(decision: QuantizationDecision) -> dict[str, Any]:
    return {
        "mode": decision.mode.value,
        "base_bit_width": decision.base_bit_width,
        "group_bit_widths": list(decision.group_bit_widths),
        "layer_bit_widths": list(decision.layer_bit_widths),
        "scale_factor": round(float(decision.scale_factor), 4),
        "clipping_range": round(float(decision.clipping_range), 4),
        "precision_level": round(float(decision.precision_level), 4),
        "moe_variant_names": list(decision.moe_variant_names),
    }


__all__ = ["recommend_quantization"]
