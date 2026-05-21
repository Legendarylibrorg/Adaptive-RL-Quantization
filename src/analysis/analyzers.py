"""Log/history analysis for pipeline reports (previously split across analysis/*.py)."""

from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, write_scatter_plot
from adaptive_quant.configuration.validation import validate_cli_path_argument
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.math_utils import mean
from analysis.log_records import (
    DEFAULT_ANALYSIS_PHASE,
    bucket_records_by_complexity,
    by_hardware,
    complexity_bucket_metrics,
    input_complexity,
    jsonl_analysis_setup,
    mean_effective_bits,
    mean_flag_rate,
    served_reward,
    summary_stats,
    training_step_reward,
    write_analysis_artifacts,
)


def _publish_analysis(
    output_root: Path,
    summary: dict[str, object],
    *,
    json_name: str,
    bar_charts: list[tuple[str, str, dict[str, float], str]] | None = None,
    scatter_charts: list[tuple[str, str, list[tuple[float, float]], str, str]] | None = None,
) -> dict[str, object]:
    write_analysis_artifacts(
        output_root,
        summary,
        json_name=json_name,
        bar_charts=bar_charts,
        scatter_charts=scatter_charts,
    )
    return summary


def analyze_hardware(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    reward_by_hardware = by_hardware(records, ("metrics", "reward"))
    latency_by_hardware = by_hardware(records, ("metrics", "latency_ms"))
    throughput_by_hardware = by_hardware(records, ("metrics", "throughput_tps"))
    perplexity_by_hardware = by_hardware(records, ("metrics", "perplexity"))
    rewards = list(reward_by_hardware.values())
    summary: dict[str, object] = {
        "log_path": log_path,
        "reward_by_hardware": reward_by_hardware,
        "latency_by_hardware": latency_by_hardware,
        "throughput_by_hardware": throughput_by_hardware,
        "perplexity_by_hardware": perplexity_by_hardware,
        "generalization_gap": (max(rewards) - min(rewards)) if rewards else 0.0,
    }
    return _publish_analysis(
        output_root,
        summary,
        json_name="hardware_generalization_summary.json",
        bar_charts=[
            (
                "hardware_generalization_reward.svg",
                "Policy Reward by Hardware",
                reward_by_hardware,
                "Reward",
            ),
            (
                "hardware_generalization_latency.svg",
                "Latency by Hardware",
                latency_by_hardware,
                "Latency (ms)",
            ),
        ],
    )


def analyze_inputs(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    points = [
        (input_complexity(record), mean_effective_bits(record.get("decision", {})))
        for record in records
    ]
    by_c = {
        name: complexity_bucket_metrics(bucket)
        for name, bucket in bucket_records_by_complexity(records).items()
    }
    summary: dict[str, object] = {"log_path": log_path, "by_complexity": by_c}
    return _publish_analysis(
        output_root,
        summary,
        json_name="input_adaptation_summary.json",
        bar_charts=[
            (
                "input_complexity_vs_bits.svg",
                "Average Precision by Input Complexity",
                {b: v["average_bits"] for b, v in by_c.items()},
                "Average effective bits",
            )
        ],
        scatter_charts=[
            (
                "input_adaptation_scatter.svg",
                "Complexity vs Precision",
                points,
                "Input complexity",
                "Average effective bits",
            )
        ],
    )


def analyze_moe_cache(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    cache_vs_latency: list[tuple[float, float]] = []
    entropy_vs_reward: list[tuple[float, float]] = []
    swap_costs, cache_misses = [], []
    for record in records:
        metrics = record.get("metrics", {})
        moe_context = record.get("moe_context") or {}
        cache_miss = float(metrics.get("cache_miss_count", 0.0))
        swap_costs.append(float(metrics.get("swap_cost_ms", 0.0)))
        cache_misses.append(cache_miss)
        cache_vs_latency.append((cache_miss, float(metrics.get("latency_ms", 0.0))))
        entropy_vs_reward.append(
            (float(moe_context.get("router_entropy", 0.0)), float(metrics.get("reward", 0.0)))
        )
    reward_by_hardware = by_hardware(records, ("metrics", "reward"))
    summary: dict[str, object] = {
        "log_path": log_path,
        "mean_swap_cost_ms": mean(swap_costs),
        "mean_cache_miss_count": mean(cache_misses),
        "reward_by_hardware": reward_by_hardware,
    }
    return _publish_analysis(
        output_root,
        summary,
        json_name="moe_cache_behavior_summary.json",
        bar_charts=[
            (
                "moe_cache_metrics.svg",
                "MoE Cache Metrics",
                {
                    "swap_cost_ms": float(summary["mean_swap_cost_ms"]),
                    "cache_miss_count": float(summary["mean_cache_miss_count"]),
                },
                "Average value",
            )
        ],
        scatter_charts=[
            (
                "moe_cache_miss_vs_latency.svg",
                "Cache Misses vs Latency",
                cache_vs_latency,
                "Cache miss count",
                "Latency (ms)",
            ),
            (
                "moe_router_entropy_vs_reward.svg",
                "Router Entropy vs Reward",
                entropy_vs_reward,
                "Router entropy",
                "Reward",
            ),
        ],
    )


def analyze_moe_experts(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    variant_usage: dict[str, float] = {}
    expert_frequency: dict[str, float] = {}
    sensitivity_vs_aggressiveness: list[tuple[float, float]] = []
    router_entropy_vals: list[float] = []
    aggressiveness_map = {"safe": 0.0, "balanced": 0.5, "aggressive": 1.0}
    for record in records:
        moe_context = record.get("moe_context") or {}
        experts = moe_context.get("experts") or []
        decision = record.get("decision") or {}
        variant_names = decision.get("moe_variant_names") or []
        router_entropy_vals.append(float(moe_context.get("router_entropy", 0.0)))
        for expert, variant_name in zip(experts, variant_names, strict=False):
            expert_key = f"expert_{int(expert.get('expert_index', 0))}"
            expert_frequency[expert_key] = expert_frequency.get(expert_key, 0.0) + 1.0
            variant_usage[variant_name] = variant_usage.get(variant_name, 0.0) + 1.0
            sensitivity_vs_aggressiveness.append(
                (float(expert.get("sensitivity", 0.0)), aggressiveness_map.get(variant_name, 0.5))
            )
    top_experts = dict(
        sorted(expert_frequency.items(), key=lambda item: item[1], reverse=True)[
            : min(8, len(expert_frequency))
        ]
    )
    summary: dict[str, object] = {
        "log_path": log_path,
        "mean_router_entropy": mean(router_entropy_vals),
        "variant_usage": variant_usage,
        "top_experts": top_experts,
        "mean_aggressiveness": mean([p[1] for p in sensitivity_vs_aggressiveness]),
        "selection_count": len(sensitivity_vs_aggressiveness),
    }
    return _publish_analysis(
        output_root,
        summary,
        json_name="moe_expert_behavior_summary.json",
        bar_charts=[
            ("moe_variant_usage.svg", "MoE Variant Usage", variant_usage, "Selections"),
            ("moe_top_experts.svg", "Most Active Experts", top_experts, "Selections"),
        ],
        scatter_charts=[
            (
                "moe_sensitivity_vs_aggressiveness.svg",
                "Expert Sensitivity vs Variant Aggressiveness",
                sensitivity_vs_aggressiveness,
                "Expert sensitivity",
                "Variant aggressiveness",
            )
        ],
    )


def analyze_quant(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    learned = [r for r in records if r.get("decision", {}).get("mode") == "learned"]
    scale_values = [float(r["decision"].get("scale_factor", 0.0)) for r in learned]
    clip_values = [float(r["decision"].get("clipping_range", 0.0)) for r in learned]
    precision_values = [float(r["decision"].get("precision_level", 0.0)) for r in learned]
    average_bits = [
        mean_effective_bits(d)
        for r in learned
        if (d := r.get("decision", {})).get("effective_layer_bits")
    ]
    summary: dict[str, object] = {
        "log_path": log_path,
        "learned_episode_count": len(learned),
        "scale_factor": summary_stats(scale_values),
        "clipping_range": summary_stats(clip_values),
        "precision_level": summary_stats(precision_values),
        "effective_bits_mean": mean(average_bits),
    }
    sf, cr, pl = summary["scale_factor"], summary["clipping_range"], summary["precision_level"]
    assert isinstance(sf, dict) and isinstance(cr, dict) and isinstance(pl, dict)
    return _publish_analysis(
        output_root,
        summary,
        json_name="quant_function_behavior_summary.json",
        bar_charts=[
            (
                "quant_function_parameters.svg",
                "Learned Quantization Parameters",
                {
                    "scale": sf["mean"],
                    "clip": cr["mean"],
                    "precision": pl["mean"],
                    "bits": float(summary["effective_bits_mean"]),
                },
                "Average value",
            )
        ],
    )


def analyze_training_dynamics(history_path: str, output_dir: str) -> dict[str, object]:
    source = Path(history_path)
    output_root = ensure_directory(output_dir)
    if not source.exists():
        empty_summary = {"history_path": history_path, "records": 0}
        write_json(str(output_root / "training_dynamics_summary.json"), empty_summary)
        return empty_summary
    records = read_json(source, label="Training history JSON")
    rewards = [training_step_reward(r) for r in records]
    points = [(float(r.get("step", 0.0)), training_step_reward(r)) for r in records]
    summary: dict[str, object] = {
        "history_path": history_path,
        "records": len(records),
        "mean_reward": mean(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
    }
    write_json(str(output_root / "training_dynamics_summary.json"), summary)
    write_scatter_plot(
        str(output_root / "training_reward_curve.svg"),
        "Training Reward Curve",
        points,
        "Update Step",
        "Reward",
    )
    return summary


def analyze_online(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = None,
) -> dict[str, object]:
    records, output_root = jsonl_analysis_setup(log_path, output_dir, phase=phase)
    reward_by_hardware = by_hardware(records, ("served_metrics", "reward"))
    complexity_reward_points = [(input_complexity(r), served_reward(r)) for r in records]
    summary: dict[str, object] = {
        "log_path": log_path,
        "records": len(records),
        "reward_by_hardware": reward_by_hardware,
        "candidate_accept_rate": mean_flag_rate(records, "accepted_candidate"),
        "online_update_rate": mean_flag_rate(records, "online_update_applied"),
        "rollback_count": sum(1 for r in records if r.get("drift_event") == "rollback"),
        "mean_served_reward": mean([served_reward(r) for r in records]),
    }
    return _publish_analysis(
        output_root,
        summary,
        json_name="online_learning_summary.json",
        bar_charts=[
            (
                "online_reward_by_hardware.svg",
                "Online Reward by Hardware",
                reward_by_hardware,
                "Reward",
            )
        ],
        scatter_charts=[
            (
                "online_complexity_vs_reward.svg",
                "Input Complexity vs Served Reward",
                complexity_reward_points,
                "Complexity",
                "Reward",
            )
        ],
    )


_CLI: dict[str, tuple[str, object, str, bool]] = {
    "hardware_generalization": (
        "python -m analysis hardware_generalization <log_path> <output_dir> [--phase eval|train|all]",
        analyze_hardware,
        "Wrote hardware analysis to",
        True,
    ),
    "input_adaptation": (
        "python -m analysis input_adaptation <log_path> <output_dir> [--phase eval|train|all]",
        analyze_inputs,
        "Wrote input adaptation analysis to",
        True,
    ),
    "moe_cache_behavior": (
        "python -m analysis moe_cache_behavior <log_path> <output_dir> [--phase eval|train|all]",
        analyze_moe_cache,
        "Wrote MoE cache analysis to",
        True,
    ),
    "moe_expert_behavior": (
        "python -m analysis moe_expert_behavior <log_path> <output_dir> [--phase eval|train|all]",
        analyze_moe_experts,
        "Wrote MoE expert analysis to",
        True,
    ),
    "quant_function_behavior": (
        "python -m analysis quant_function_behavior <log_path> <output_dir> [--phase eval|train|all]",
        analyze_quant,
        "Wrote quant function analysis to",
        True,
    ),
    "training_dynamics": (
        "python -m analysis training_dynamics <history_path> <output_dir>",
        analyze_training_dynamics,
        "Wrote training dynamics analysis to",
        False,
    ),
    "online_learning": (
        "python -m analysis online_learning <log_path> <output_dir>",
        analyze_online,
        "Wrote online analysis to",
        False,
    ),
}

CLI_COMMANDS = frozenset(_CLI)


def _parse_phase_argv(argv: list[str], usage: str) -> tuple[list[str], str | None]:
    phase: str | None = DEFAULT_ANALYSIS_PHASE
    positional: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--phase":
            if i + 1 >= len(argv):
                raise SystemExit(f"Usage: {usage}")
            value = argv[i + 1].strip().lower()
            phase = None if value == "all" else value
            i += 2
            continue
        if token.startswith("--phase="):
            value = token.split("=", 1)[1].strip().lower()
            phase = None if value == "all" else value
            i += 1
            continue
        positional.append(token)
        i += 1
    return positional, phase


def run_cli(key: str) -> None:
    usage, fn, msg, supports_phase = _CLI[key]
    argv = list(sys.argv[1:])
    if supports_phase:
        argv, phase = _parse_phase_argv(argv, usage)
    else:
        phase = None
    if len(argv) != 2:
        raise SystemExit(f"Usage: {usage}")
    for label, raw in (("log/history", argv[0]), ("output", argv[1])):
        try:
            validate_cli_path_argument(label, raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid {label} path: {exc}") from exc
    out = fn(argv[0], argv[1], phase=phase) if supports_phase else fn(argv[0], argv[1])
    print(f"{msg} {Path(argv[1]).resolve()}")
    print(out)


__all__ = [
    "CLI_COMMANDS",
    "analyze_hardware",
    "analyze_inputs",
    "analyze_moe_cache",
    "analyze_moe_experts",
    "analyze_online",
    "analyze_quant",
    "analyze_training_dynamics",
    "run_cli",
]
