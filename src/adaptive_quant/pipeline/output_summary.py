"""Headline metrics and report helpers — keep ``*_summary.json`` and reports readable."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def experiment_config_summary(config: FrameworkConfig) -> dict[str, object]:
    """Small config fingerprint for multiseed/sweep aggregate JSON (not full ``asdict``)."""
    return {
        "run_name": config.run_name,
        "backend": config.backend,
        "training_backend": config.training_backend,
        "training_episodes": config.training_episodes,
        "evaluation_episodes": config.evaluation_episodes,
        "quant_mode": config.quant_mode,
        "hardware_modes": list(config.hardware_modes),
        "moe_enabled": config.moe_enabled,
        "seed": config.seed,
    }


def headline_summary_for_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Curated summary subset for paper-bundle headline metrics (skip config blobs)."""
    curated: dict[str, Any] = {}
    for section in ("train", "evaluation", "bootstrap_train", "online"):
        block = summary.get(section)
        if isinstance(block, dict):
            curated[section] = block
    benchmarks = summary.get("benchmarks")
    if isinstance(benchmarks, dict):
        curated["benchmarks"] = {
            key: benchmarks[key]
            for key in ("single_vs_multi", "static_vs_dynamic", "discrete_vs_learned")
            if key in benchmarks
        }
    recommendation = summary.get("recommendation")
    if isinstance(recommendation, dict):
        slim_rec: dict[str, Any] = {}
        if "target_hardware" in recommendation:
            slim_rec["target_hardware"] = recommendation["target_hardware"]
        adaptive = recommendation.get("adaptive_policy")
        if isinstance(adaptive, dict):
            slim_rec["adaptive_policy"] = {
                key: adaptive[key] for key in ("mean_reward",) if key in adaptive
            }
        fixed = recommendation.get("recommended_quant")
        if isinstance(fixed, dict):
            slim_rec["recommended_quant"] = {
                "signature": fixed.get("signature"),
                "evaluation": fixed.get("evaluation")
                if isinstance(fixed.get("evaluation"), dict)
                else None,
            }
        decision = recommendation.get("decision")
        if isinstance(decision, dict):
            slim_rec["decision"] = decision
        curated["recommendation"] = slim_rec
    analysis = summary.get("analysis")
    if isinstance(analysis, dict):
        curated["analysis"] = {
            name: slim_analysis_section(section)
            for name, section in analysis.items()
            if isinstance(section, dict)
        }
    return curated


def slim_online_analysis_for_summary(online_analysis: dict[str, object]) -> dict[str, object]:
    """Drop chart paths from online summary JSON; figures live under ``analysis_dir``."""
    slim: dict[str, object] = {}
    for key in (
        "log_path",
        "records",
        "reward_by_hardware",
        "candidate_accept_rate",
        "online_update_rate",
        "rollback_count",
        "mean_served_reward",
    ):
        if key in online_analysis:
            slim[key] = online_analysis[key]
    return slim


def online_analysis_takeaway_lines(online_analysis: dict[str, object]) -> list[str]:
    lines: list[str] = []
    if online_analysis.get("mean_served_reward") is not None:
        lines.append(f"- Mean served reward: **{_fmt(online_analysis.get('mean_served_reward'))}**")
    if online_analysis.get("candidate_accept_rate") is not None:
        lines.append(
            f"- Candidate accept rate: **{_fmt(online_analysis.get('candidate_accept_rate'))}**"
        )
    if online_analysis.get("online_update_rate") is not None:
        lines.append(f"- Online update rate: **{_fmt(online_analysis.get('online_update_rate'))}**")
    rollback = online_analysis.get("rollback_count")
    if rollback is not None:
        lines.append(f"- Rollbacks: **{int(rollback)}**")
    rewards = online_analysis.get("reward_by_hardware")
    if isinstance(rewards, dict) and rewards:
        best = max(rewards.items(), key=lambda item: float(item[1]))
        lines.append(f"- Best hardware for served reward: `{best[0]}` ({_fmt(best[1])})")
    return lines or ["- (no online analysis takeaways — check `outputs/analysis/<run>/online/`)"]


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


def build_research_artifact_index(
    config: FrameworkConfig,
    artifacts: Mapping[str, Any],
) -> dict[str, str | None]:
    """Stable artifact map for research-grade navigation in summaries and reports."""
    paper_bundle = artifacts.get("paper_bundle")
    bundle_dir: str | None = None
    if isinstance(paper_bundle, Mapping):
        raw = paper_bundle.get("paper_bundle_dir")
        bundle_dir = str(raw) if raw else None
    return {
        "summary_json": config.summary_path(),
        "report_md": _artifact_path(artifacts, "report"),
        "checkpoint": _artifact_path(artifacts, "final_checkpoint"),
        "recommendation_json": _artifact_path(artifacts, "recommendation"),
        "training_history": _artifact_path(artifacts, "training_history"),
        "exported_gguf": _artifact_path(artifacts, "exported_gguf"),
        "paper_bundle_dir": bundle_dir,
        "analysis_dir": f"{config.analysis_dir}/{config.run_name}/",
    }


def _artifact_path(artifacts: Mapping[str, Any], key: str) -> str | None:
    value = artifacts.get(key)
    if value is None:
        return None
    return str(value)


def gguf_export_report_lines(gguf_export: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(gguf_export, dict):
        return ["- GGUF export disabled."]
    if not gguf_export.get("enabled"):
        return ["- GGUF export disabled (`llama_cpp_gguf_export_enabled=false`)."]
    if gguf_export.get("error"):
        return [f"- GGUF export failed: `{gguf_export.get('error')}`"]
    output_path = gguf_export.get("output_path")
    if output_path:
        return [
            f"- exported GGUF: `{output_path}`",
            f"- quant type: `{gguf_export.get('quant_type')}`",
            f"- source: `{gguf_export.get('source_path')}`",
        ]
    return ["- GGUF export did not produce an output path."]


__all__ = [
    "analysis_takeaway_lines",
    "benchmark_metric_rows",
    "build_research_artifact_index",
    "experiment_config_summary",
    "gguf_export_report_lines",
    "headline_summary_for_metrics",
    "online_analysis_takeaway_lines",
    "recommendation_decision_block",
    "resolve_analysis_log_path",
    "slim_analysis_for_summary",
    "slim_online_analysis_for_summary",
]
