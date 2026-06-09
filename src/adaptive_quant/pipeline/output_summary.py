"""Headline metrics and report helpers — keep ``*_summary.json`` and reports readable."""

from __future__ import annotations

import math
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import format_display


def resolve_analysis_log_path(config: FrameworkConfig, suffix: str) -> str:
    """Prefer benchmark variant logs ``{run_name}_{suffix}.jsonl``, else primary log."""
    specialized = Path(config.log_dir) / f"{config.run_name}_{suffix}.jsonl"
    if specialized.is_file():
        return str(specialized)
    primary = Path(config.primary_log_path())
    if primary.is_file():
        return str(primary)
    return str(specialized)


def _fmt(value: object, *, digits: int = 3) -> str:
    return format_display(value, style="report", digits=digits)


def slim_analysis_section(section: object) -> dict[str, object]:
    if not isinstance(section, dict):
        return {}
    slim: dict[str, object] = {}
    if "log_path" in section:
        slim["log_path"] = section["log_path"]
    for key in (
        "generalization_gap",
        "reward_by_hardware",
        "latency_by_hardware",
        "throughput_by_hardware",
        "perplexity_by_hardware",
        "by_complexity",
        "mean_swap_cost_ms",
        "mean_cache_miss_count",
        "mean_reward",
        "final_reward",
        "episodes",
    ):
        if key in section:
            slim[key] = section[key]
    return slim


def slim_analysis_for_summary(
    analysis: dict[str, object],
    config: FrameworkConfig,
) -> dict[str, object]:
    """Drop chart-ready blobs from ``*_summary.json``; full JSON lives under ``analysis_dir``."""
    root = Path(config.analysis_dir) / config.run_name
    slim: dict[str, object] = {"root": str(root)}
    for name, section in analysis.items():
        if not isinstance(section, dict):
            continue
        entry = slim_analysis_section(section)
        subdir = {
            "hardware": "hardware",
            "input": "inputs",
            "quant_function": "quant",
            "training_dynamics": "training",
            "moe_experts": "moe_experts",
            "moe_cache": "moe_cache",
        }.get(name, name)
        entry["artifacts_dir"] = str(root / subdir)
        slim[name] = entry
    return slim


def analysis_takeaway_lines(analysis: dict[str, object]) -> list[str]:
    lines: list[str] = []
    hardware = analysis.get("hardware")
    if isinstance(hardware, dict):
        gap = hardware.get("generalization_gap")
        if gap is not None:
            lines.append(
                f"- Hardware generalization gap (reward spread): **{_fmt(gap)}** "
                "(lower is more uniform across hardware modes)"
            )
        rewards = hardware.get("reward_by_hardware")
        if isinstance(rewards, dict) and rewards:
            best = max(rewards.items(), key=lambda item: float(item[1]))
            worst = min(rewards.items(), key=lambda item: float(item[1]))
            lines.append(
                f"- Best hardware mode for reward: `{best[0]}` ({_fmt(best[1])}); "
                f"weakest: `{worst[0]}` ({_fmt(worst[1])})"
            )

    inputs = analysis.get("input")
    if isinstance(inputs, dict):
        by_c = inputs.get("by_complexity")
        if isinstance(by_c, dict) and by_c:
            bits = {
                bucket: float((metrics or {}).get("average_bits", 0.0))
                for bucket, metrics in by_c.items()
                if isinstance(metrics, dict)
            }
            if bits:
                low, high = (
                    min(bits.items(), key=lambda x: x[1]),
                    max(bits.items(), key=lambda x: x[1]),
                )
                lines.append(
                    f"- Input complexity vs precision: `{low[0]}` → {_fmt(low[1])} bits, "
                    f"`{high[0]}` → {_fmt(high[1])} bits"
                )

    training = analysis.get("training_dynamics")
    if isinstance(training, dict) and training.get("mean_reward") is not None:
        lines.append(
            f"- Training dynamics mean step reward: **{_fmt(training.get('mean_reward'))}**"
        )

    moe_cache = analysis.get("moe_cache")
    if isinstance(moe_cache, dict) and moe_cache.get("mean_cache_miss_count") is not None:
        lines.append(
            f"- MoE cache: mean misses **{_fmt(moe_cache.get('mean_cache_miss_count'))}**, "
            f"swap cost **{_fmt(moe_cache.get('mean_swap_cost_ms'))} ms**"
        )

    return lines or ["- (no analysis takeaways — check logs under `outputs/analysis/`)"]


def benchmark_metric_rows(
    benchmark_summary: dict[str, object],
) -> list[list[str]]:
    """Flatten head-to-head benchmark eval metrics into table rows."""
    rows: list[list[str]] = []
    sections = (
        ("static_vs_dynamic", ("static", "dynamic")),
        ("discrete_vs_learned", ("discrete", "learned")),
    )
    metrics = (
        "mean_reward",
        "mean_latency_ms",
        "mean_throughput_tps",
        "mean_stability_penalty",
    )
    for section_name, variants in sections:
        section = benchmark_summary.get(section_name)
        if not isinstance(section, dict):
            continue
        evaluation = section.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        for metric in metrics:
            cells = [f"{section_name}.{metric}"]
            for variant in variants:
                bucket = evaluation.get(variant)
                if isinstance(bucket, dict):
                    cells.append(_fmt(bucket.get(metric)))
                else:
                    cells.append("—")
            if len(cells) == 3:
                rows.append(cells)
        for delta_name, delta_value in section.items():
            if delta_name in {"train", "evaluation"}:
                continue
            rows.append([f"{section_name}.{delta_name}", _fmt(delta_value), "—", "—"])
    single = benchmark_summary.get("single_vs_multi")
    if isinstance(single, dict) and single.get("generalization_gap_improvement") is not None:
        rows.append(
            [
                "single_vs_multi.gap_improvement",
                _fmt(single.get("generalization_gap_improvement")),
                "—",
                "—",
            ]
        )
    return rows


def recommendation_decision_block(payload: dict[str, object]) -> dict[str, object]:
    adaptive = payload.get("adaptive_policy")
    recommended = payload.get("recommended_quant")
    adaptive_reward = (
        float((adaptive or {}).get("mean_reward", float("nan")))
        if isinstance(adaptive, dict)
        else float("nan")
    )
    fixed_reward = float("nan")
    signature: str | None = None
    if isinstance(recommended, dict):
        signature = str(recommended.get("signature", "")) or None
        evaluation = recommended.get("evaluation")
        if isinstance(evaluation, dict):
            fixed_reward = float(evaluation.get("mean_reward", float("nan")))

    use_adaptive = True
    delta: float | None = None
    if signature and math.isfinite(adaptive_reward) and math.isfinite(fixed_reward):
        delta = fixed_reward - adaptive_reward
        use_adaptive = fixed_reward <= adaptive_reward

    if use_adaptive:
        rationale = (
            "Use the trained adaptive policy on the target hardware "
            "(fixed candidate did not beat adaptive on mean reward)."
        )
        deploy = "adaptive_policy"
    else:
        rationale = (
            f"Deploy fixed quant `{signature}` — mean reward {_fmt(fixed_reward)} "
            f"vs adaptive {_fmt(adaptive_reward)} (Δ {_fmt(delta)})."
        )
        deploy = signature or "fixed_quant"

    block: dict[str, object] = {
        "deploy": deploy,
        "use_adaptive_policy": use_adaptive,
        "rationale": rationale,
    }
    if delta is not None and delta == delta:
        block["reward_delta_vs_adaptive"] = delta
    return block


__all__ = [
    "analysis_takeaway_lines",
    "benchmark_metric_rows",
    "recommendation_decision_block",
    "resolve_analysis_log_path",
    "slim_analysis_for_summary",
]
