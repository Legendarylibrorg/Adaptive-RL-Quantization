"""Log/history analysis for pipeline reports (previously split across analysis/*.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from adaptive_quant.analysis_utils import (
    ensure_directory,
    grouped_mean,
    write_bar_chart,
    write_scatter_plot,
)
from adaptive_quant.logging_utils import (
    enforce_local_read_limit,
    load_jsonl,
    write_json,
)
from adaptive_quant.math_utils import mean

DEFAULT_ANALYSIS_PHASE = "eval"


def _filter_phase(records: list[dict], phase: str | None) -> list[dict]:
    """Filter JSONL records by ``phase`` field.

    - ``phase=None`` keeps every record.
    - Otherwise keep only records whose ``phase`` equals the requested value.
      If *no* record carries a ``phase`` field (legacy logs), the records are
      returned unchanged so old logs remain analyzable.
    """
    if phase is None:
        return records
    if not any("phase" in record for record in records):
        return records
    return [record for record in records if record.get("phase") == phase]


def _mean_effective_bits(decision: dict) -> float:
    bits = decision.get("effective_layer_bits", [])
    return mean([float(b) for b in bits]) if bits else 0.0


def _training_step_reward(record: dict) -> float:
    return float(record.get("batch_reward", record.get("reward", 0.0)))


def _served_reward(record: dict) -> float:
    return float(record.get("served_metrics", {}).get("reward", 0.0))


def _input_complexity(record: dict) -> float:
    return float(record.get("input_features", {}).get("complexity_score", 0.0))


def _mean_flag_rate(records: list[dict], key: str) -> float:
    return mean([1.0 if r.get(key) else 0.0 for r in records])


def _by_hardware(records: list[dict], metric_path: tuple[str, ...]) -> dict[str, float]:
    return grouped_mean(records, "hardware_mode", metric_path)


def _jsonl_analysis_setup(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> tuple[list[dict], Path]:
    records = _filter_phase(load_jsonl(log_path), phase)
    return records, ensure_directory(output_dir)


def analyze_hardware(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    reward_by_hardware = _by_hardware(records, ("metrics", "reward"))
    latency_by_hardware = _by_hardware(records, ("metrics", "latency_ms"))
    throughput_by_hardware = _by_hardware(records, ("metrics", "throughput_tps"))
    perplexity_by_hardware = _by_hardware(records, ("metrics", "perplexity"))
    rewards = list(reward_by_hardware.values())
    summary: dict[str, object] = {
        "log_path": log_path,
        "reward_by_hardware": reward_by_hardware,
        "latency_by_hardware": latency_by_hardware,
        "throughput_by_hardware": throughput_by_hardware,
        "perplexity_by_hardware": perplexity_by_hardware,
        "generalization_gap": (max(rewards) - min(rewards)) if rewards else 0.0,
    }
    write_json(str(output_root / "hardware_generalization_summary.json"), summary)
    write_bar_chart(str(output_root / "hardware_generalization_reward.svg"), "Policy Reward by Hardware", reward_by_hardware, "Reward")
    write_bar_chart(str(output_root / "hardware_generalization_latency.svg"), "Latency by Hardware", latency_by_hardware, "Latency (ms)")
    return summary


def _complexity_bucket(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.70:
        return "medium"
    return "high"


def analyze_inputs(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    points: list[tuple[float, float]] = []
    for record in records:
        decision = record.get("decision", {})
        complexity = _input_complexity(record)
        average_bits = _mean_effective_bits(decision)
        points.append((complexity, average_bits))
        buckets[_complexity_bucket(complexity)].append(record)
    summary: dict[str, object] = {"log_path": log_path, "by_complexity": {}}
    for bucket_name, bucket_records in buckets.items():
        avg_bits, avg_perplexity, avg_reward = [], [], []
        for record in bucket_records:
            decision = record.get("decision", {})
            metrics = record.get("metrics", {})
            if decision.get("effective_layer_bits"):
                avg_bits.append(_mean_effective_bits(decision))
            if "perplexity" in metrics:
                avg_perplexity.append(float(metrics["perplexity"]))
            if "reward" in metrics:
                avg_reward.append(float(metrics["reward"]))
        summary["by_complexity"][bucket_name] = {
            "average_bits": mean(avg_bits),
            "average_perplexity": mean(avg_perplexity),
            "average_reward": mean(avg_reward),
            "count": len(bucket_records),
        }
    write_json(str(output_root / "input_adaptation_summary.json"), summary)
    by_c = summary["by_complexity"]
    assert isinstance(by_c, dict)
    write_bar_chart(
        str(output_root / "input_complexity_vs_bits.svg"),
        "Average Precision by Input Complexity",
        {b: v["average_bits"] for b, v in by_c.items()},
        "Average effective bits",
    )
    write_scatter_plot(str(output_root / "input_adaptation_scatter.svg"), "Complexity vs Precision", points, "Input complexity", "Average effective bits")
    return summary


def analyze_moe_cache(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    cache_vs_latency: list[tuple[float, float]] = []
    entropy_vs_reward: list[tuple[float, float]] = []
    swap_costs, cache_misses = [], []
    for record in records:
        metrics = record.get("metrics", {})
        moe_context = record.get("moe_context") or {}
        cache_miss = float(metrics.get("cache_miss_count", 0.0))
        latency = float(metrics.get("latency_ms", 0.0))
        reward = float(metrics.get("reward", 0.0))
        router_entropy = float(moe_context.get("router_entropy", 0.0))
        swap_costs.append(float(metrics.get("swap_cost_ms", 0.0)))
        cache_misses.append(cache_miss)
        cache_vs_latency.append((cache_miss, latency))
        entropy_vs_reward.append((router_entropy, reward))
    reward_by_hardware = _by_hardware(records, ("metrics", "reward"))
    summary: dict[str, object] = {
        "log_path": log_path,
        "mean_swap_cost_ms": mean(swap_costs),
        "mean_cache_miss_count": mean(cache_misses),
        "reward_by_hardware": reward_by_hardware,
    }
    write_json(str(output_root / "moe_cache_behavior_summary.json"), summary)
    write_bar_chart(
        str(output_root / "moe_cache_metrics.svg"),
        "MoE Cache Metrics",
        {"swap_cost_ms": summary["mean_swap_cost_ms"], "cache_miss_count": summary["mean_cache_miss_count"]},
        "Average value",
    )
    write_scatter_plot(str(output_root / "moe_cache_miss_vs_latency.svg"), "Cache Misses vs Latency", cache_vs_latency, "Cache miss count", "Latency (ms)")
    write_scatter_plot(str(output_root / "moe_router_entropy_vs_reward.svg"), "Router Entropy vs Reward", entropy_vs_reward, "Router entropy", "Reward")
    return summary


def analyze_moe_experts(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    variant_usage: dict[str, float] = {}
    expert_frequency: dict[str, float] = {}
    sensitivity_vs_aggressiveness: list[tuple[float, float]] = []
    router_entropy_vals: list[float] = []
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
            aggressiveness = {"safe": 0.0, "balanced": 0.5, "aggressive": 1.0}.get(variant_name, 0.5)
            sensitivity_vs_aggressiveness.append((float(expert.get("sensitivity", 0.0)), aggressiveness))
    top_experts = dict(sorted(expert_frequency.items(), key=lambda item: item[1], reverse=True)[: min(8, len(expert_frequency))])
    summary: dict[str, object] = {
        "log_path": log_path,
        "mean_router_entropy": mean(router_entropy_vals),
        "variant_usage": variant_usage,
        "top_experts": top_experts,
        "mean_aggressiveness": mean([p[1] for p in sensitivity_vs_aggressiveness]),
        "selection_count": len(sensitivity_vs_aggressiveness),
    }
    write_json(str(output_root / "moe_expert_behavior_summary.json"), summary)
    write_bar_chart(str(output_root / "moe_variant_usage.svg"), "MoE Variant Usage", variant_usage, "Selections")
    write_bar_chart(str(output_root / "moe_top_experts.svg"), "Most Active Experts", top_experts, "Selections")
    write_scatter_plot(
        str(output_root / "moe_sensitivity_vs_aggressiveness.svg"),
        "Expert Sensitivity vs Variant Aggressiveness",
        sensitivity_vs_aggressiveness,
        "Expert sensitivity",
        "Variant aggressiveness",
    )
    return summary


def analyze_quant(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> dict[str, object]:
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    learned = [r for r in records if r.get("decision", {}).get("mode") == "learned"]
    scale_values = [float(r["decision"].get("scale_factor", 0.0)) for r in learned]
    clip_values = [float(r["decision"].get("clipping_range", 0.0)) for r in learned]
    precision_values = [float(r["decision"].get("precision_level", 0.0)) for r in learned]
    average_bits = []
    for r in learned:
        d = r.get("decision", {})
        if d.get("effective_layer_bits"):
            average_bits.append(_mean_effective_bits(d))
    summary: dict[str, object] = {
        "log_path": log_path,
        "learned_episode_count": len(learned),
        "scale_factor": {"mean": mean(scale_values), "min": min(scale_values) if scale_values else 0.0, "max": max(scale_values) if scale_values else 0.0},
        "clipping_range": {"mean": mean(clip_values), "min": min(clip_values) if clip_values else 0.0, "max": max(clip_values) if clip_values else 0.0},
        "precision_level": {"mean": mean(precision_values), "min": min(precision_values) if precision_values else 0.0, "max": max(precision_values) if precision_values else 0.0},
        "effective_bits_mean": mean(average_bits),
    }
    write_json(str(output_root / "quant_function_behavior_summary.json"), summary)
    sf, cr, pl = summary["scale_factor"], summary["clipping_range"], summary["precision_level"]
    assert isinstance(sf, dict) and isinstance(cr, dict) and isinstance(pl, dict)
    write_bar_chart(
        str(output_root / "quant_function_parameters.svg"),
        "Learned Quantization Parameters",
        {"scale": sf["mean"], "clip": cr["mean"], "precision": pl["mean"], "bits": summary["effective_bits_mean"]},
        "Average value",
    )
    return summary


def analyze_training_dynamics(history_path: str, output_dir: str) -> dict[str, object]:
    source = Path(history_path)
    output_root = ensure_directory(output_dir)
    if not source.exists():
        summary = {"history_path": history_path, "records": 0}
        write_json(str(output_root / "training_dynamics_summary.json"), summary)
        return summary
    enforce_local_read_limit(source, label="Training history JSON")
    records = json.loads(source.read_text(encoding="utf-8"))
    rewards = [_training_step_reward(r) for r in records]
    points = [(float(r.get("step", 0.0)), _training_step_reward(r)) for r in records]
    summary: dict[str, object] = {
        "history_path": history_path,
        "records": len(records),
        "mean_reward": mean(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
    }
    write_json(str(output_root / "training_dynamics_summary.json"), summary)
    write_scatter_plot(str(output_root / "training_reward_curve.svg"), "Training Reward Curve", points, "Update Step", "Reward")
    return summary


def analyze_online(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = None,
) -> dict[str, object]:
    """Online telemetry has its own ``served_metrics`` shape and no train/eval split,
    so phase filtering defaults to ``None`` here."""
    records, output_root = _jsonl_analysis_setup(log_path, output_dir, phase=phase)
    reward_by_hardware = _by_hardware(records, ("served_metrics", "reward"))
    accept_rate = _mean_flag_rate(records, "accepted_candidate")
    update_rate = _mean_flag_rate(records, "online_update_applied")
    rollback_count = sum(1 for r in records if r.get("drift_event") == "rollback")
    complexity_reward_points = [(_input_complexity(r), _served_reward(r)) for r in records]
    summary: dict[str, object] = {
        "log_path": log_path,
        "records": len(records),
        "reward_by_hardware": reward_by_hardware,
        "candidate_accept_rate": accept_rate,
        "online_update_rate": update_rate,
        "rollback_count": rollback_count,
        "mean_served_reward": mean([_served_reward(r) for r in records]),
    }
    write_json(str(output_root / "online_learning_summary.json"), summary)
    write_bar_chart(str(output_root / "online_reward_by_hardware.svg"), "Online Reward by Hardware", reward_by_hardware, "Reward")
    write_scatter_plot(str(output_root / "online_complexity_vs_reward.svg"), "Input Complexity vs Served Reward", complexity_reward_points, "Complexity", "Reward")
    return summary


_CLI: dict[str, tuple[str, object, str, bool]] = {
    "hardware_generalization": (
        "python3 analysis/hardware_generalization.py <log_path> <output_dir> [--phase eval|train|all]",
        analyze_hardware,
        "Wrote hardware analysis to",
        True,
    ),
    "input_adaptation": (
        "python3 analysis/input_adaptation.py <log_path> <output_dir> [--phase eval|train|all]",
        analyze_inputs,
        "Wrote input adaptation analysis to",
        True,
    ),
    "moe_cache_behavior": (
        "python3 analysis/moe_cache_behavior.py <log_path> <output_dir> [--phase eval|train|all]",
        analyze_moe_cache,
        "Wrote MoE cache analysis to",
        True,
    ),
    "moe_expert_behavior": (
        "python3 analysis/moe_expert_behavior.py <log_path> <output_dir> [--phase eval|train|all]",
        analyze_moe_experts,
        "Wrote MoE expert analysis to",
        True,
    ),
    "quant_function_behavior": (
        "python3 analysis/quant_function_behavior.py <log_path> <output_dir> [--phase eval|train|all]",
        analyze_quant,
        "Wrote quant function analysis to",
        True,
    ),
    "training_dynamics": (
        "python3 analysis/training_dynamics.py <history_path> <output_dir>",
        analyze_training_dynamics,
        "Wrote training dynamics analysis to",
        False,
    ),
    "online_learning": (
        "python3 analysis/online_learning.py <log_path> <output_dir>",
        analyze_online,
        "Wrote online analysis to",
        False,
    ),
}


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
        if any(c in raw for c in "\n\r\x00"):
            raise SystemExit(f"Invalid characters in {label} path.")
    out = fn(argv[0], argv[1], phase=phase) if supports_phase else fn(argv[0], argv[1])
    print(f"{msg} {Path(argv[1]).resolve()}")
    print(out)


__all__ = [
    "analyze_hardware",
    "analyze_inputs",
    "analyze_moe_cache",
    "analyze_moe_experts",
    "analyze_online",
    "analyze_quant",
    "analyze_training_dynamics",
    "run_cli",
]
