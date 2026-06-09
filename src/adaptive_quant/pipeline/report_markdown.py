from __future__ import annotations

import json
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import md_table, write_text_file
from adaptive_quant.math_utils import format_display
from adaptive_quant.pipeline.output_summary import analysis_takeaway_lines, benchmark_metric_rows


def fmt_report_num(value: object, *, digits: int = 2) -> str:
    return format_display(value, style="report", digits=digits)


def maybe_report_link(report_dir: Path, rel_path: Path) -> str:
    abs_path = report_dir / rel_path
    if abs_path.exists():
        return f"[{rel_path.name}]({rel_path.as_posix()})"
    return f"`{rel_path.as_posix()}` (missing)"


def md_code_json(obj: object) -> str:
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```"


def _dict_table_rows(data: dict[str, object] | None) -> list[list[str]]:
    if not isinstance(data, dict):
        return []
    return [[str(key), fmt_report_num(value)] for key, value in sorted(data.items())]


def write_research_report_markdown(
    config: FrameworkConfig,
    *,
    git_commit: str | None,
    train_summary: dict[str, object],
    eval_summary: dict[str, object],
    benchmark_summary: dict[str, object],
    gpu_profile_report: dict[str, object] | None,
    preflight_report: dict[str, object] | None,
    vram_report: dict[str, object] | None,
    analysis: dict[str, object],
    history_path: str | None,
    checkpoint_path: str | None,
    recommendation_summary: dict[str, object] | None,
) -> str | None:
    if not config.write_research_report:
        return None
    report_path = config.report_path()
    target = Path(report_path)
    report_dir = target.parent

    def _analysis_fig_links() -> list[str]:
        analysis_root = Path(config.analysis_dir) / config.run_name
        rel_root = Path("..") / "analysis" / config.run_name
        candidates = [
            ("hardware reward", analysis_root / "hardware" / "hardware_generalization_reward.svg"),
            (
                "hardware latency",
                analysis_root / "hardware" / "hardware_generalization_latency.svg",
            ),
            ("input complexity vs bits", analysis_root / "inputs" / "input_complexity_vs_bits.svg"),
            ("input adaptation scatter", analysis_root / "inputs" / "input_adaptation_scatter.svg"),
            ("quant function params", analysis_root / "quant" / "quant_function_parameters.svg"),
            ("training reward curve", analysis_root / "training" / "training_reward_curve.svg"),
        ]
        if config.moe_enabled:
            candidates.extend(
                [
                    (
                        "MoE variant usage",
                        analysis_root / "moe_experts" / "moe_variant_usage.svg",
                    ),
                    (
                        "MoE cache metrics",
                        analysis_root / "moe_cache" / "moe_cache_metrics.svg",
                    ),
                ]
            )
        lines: list[str] = []
        for label, abs_path in candidates:
            rel_path = rel_root / abs_path.relative_to(analysis_root)
            lines.append(f"- {label}: {maybe_report_link(report_dir, rel_path)}")
        return lines

    eval_metrics = [
        ("mean_reward", eval_summary.get("mean_reward")),
        ("mean_latency_ms", eval_summary.get("mean_latency_ms")),
        ("mean_throughput_tps", eval_summary.get("mean_throughput_tps")),
        ("mean_memory_mb", eval_summary.get("mean_memory_mb")),
        ("mean_perplexity", eval_summary.get("mean_perplexity")),
        ("mean_stability_penalty", eval_summary.get("mean_stability_penalty")),
    ]
    eval_rows = [[k, fmt_report_num(v)] for k, v in eval_metrics]
    recommendation = recommendation_summary if isinstance(recommendation_summary, dict) else None
    recommended_quant = (
        recommendation.get("recommended_quant") if isinstance(recommendation, dict) else None
    )
    decision = recommendation.get("decision") if isinstance(recommendation, dict) else None

    bench = benchmark_summary
    single_vs_multi = bench.get("single_vs_multi") if isinstance(bench, dict) else None
    static_vs_dynamic = bench.get("static_vs_dynamic") if isinstance(bench, dict) else None
    discrete_vs_learned = bench.get("discrete_vs_learned") if isinstance(bench, dict) else None

    key_results_lines: list[str] = []
    if isinstance(single_vs_multi, dict):
        key_results_lines.append(
            "- universal policy gap improvement: "
            f"**{fmt_report_num(single_vs_multi.get('generalization_gap_improvement'))}** "
            "(lower gap is better)"
        )
    if isinstance(static_vs_dynamic, dict):
        evaluation = static_vs_dynamic.get("evaluation", {})
        if isinstance(evaluation, dict):
            s = evaluation.get("static", {})
            d = evaluation.get("dynamic", {})
            if isinstance(s, dict) and isinstance(d, dict):
                key_results_lines.append(
                    "- static → dynamic reward: "
                    f"{fmt_report_num(s.get('mean_reward'))} → {fmt_report_num(d.get('mean_reward'))}"
                )
                key_results_lines.append(
                    "- static → dynamic stability penalty: "
                    f"{fmt_report_num(s.get('mean_stability_penalty'))} → {fmt_report_num(d.get('mean_stability_penalty'))}"
                )
    if isinstance(discrete_vs_learned, dict):
        evaluation = discrete_vs_learned.get("evaluation", {})
        if isinstance(evaluation, dict):
            learned_metrics = evaluation.get("learned", {})
            discrete_metrics = evaluation.get("discrete", {})
            if isinstance(discrete_metrics, dict) and isinstance(learned_metrics, dict):
                key_results_lines.append(
                    "- discrete → learned reward: "
                    f"{fmt_report_num(discrete_metrics.get('mean_reward'))} → {fmt_report_num(learned_metrics.get('mean_reward'))}"
                )
                key_results_lines.append(
                    "- discrete → learned latency (ms): "
                    f"{fmt_report_num(discrete_metrics.get('mean_latency_ms'))} → {fmt_report_num(learned_metrics.get('mean_latency_ms'))}"
                )

    benchmark_rows = benchmark_metric_rows(benchmark_summary)
    benchmark_table_lines = (
        md_table(["comparison", "left", "right"], benchmark_rows)
        if benchmark_rows
        else ["_no benchmark comparisons available_"]
    )

    analysis_root = Path("..") / "analysis" / config.run_name
    analysis_json_links = [
        f"- {name}: {maybe_report_link(report_dir, analysis_root / sub / json_file)}"
        for name, sub, json_file in (
            ("hardware", "hardware", "hardware_generalization_summary.json"),
            ("inputs", "inputs", "input_adaptation_summary.json"),
            ("quant", "quant", "quant_function_behavior_summary.json"),
            ("training", "training", "training_dynamics_summary.json"),
        )
    ]

    lines = [
        f"# {config.run_name}",
        "",
        "## Overview",
        f"- backend: `{config.training_backend}`",
        f"- training_host_label: `{config.training_host_label or 'unspecified'}`",
        f"- quant_mode: `{config.quant_mode}`",
        f"- moe_enabled: `{config.moe_enabled}`",
        f"- hardware_modes: `{', '.join(config.hardware_modes)}`",
        f"- git_commit: `{git_commit or 'unknown'}`",
        "",
        "## Key results (from this run)",
        *(key_results_lines or ["- (no benchmark summary available)"]),
        "",
        "## Training",
        f"- episodes: `{train_summary.get('episodes')}`",
        f"- mean_reward: `{fmt_report_num(train_summary.get('mean_reward'))}`",
        f"- best_reward: `{fmt_report_num(train_summary.get('best_reward'))}`",
        f"- final_reward: `{fmt_report_num(train_summary.get('final_reward'))}`",
        f"- history: `{history_path or 'not written'}`",
        f"- checkpoint: `{checkpoint_path or 'not written'}`",
        "",
        "## Evaluation",
        *md_table(["metric", "value"], eval_rows),
        "",
        "## Recommendation",
    ]
    if isinstance(decision, dict):
        lines.extend(
            [
                f"- **deploy:** `{decision.get('deploy', 'n/a')}`",
                f"- use_adaptive_policy: `{decision.get('use_adaptive_policy')}`",
                f"- rationale: {decision.get('rationale', 'n/a')}",
            ]
        )
        if decision.get("reward_delta_vs_adaptive") is not None:
            lines.append(
                f"- reward_delta_vs_adaptive: `{fmt_report_num(decision.get('reward_delta_vs_adaptive'))}`"
            )
    lines.extend(
        [
            f"- target_hardware: `{recommendation.get('target_hardware') if recommendation else 'n/a'}`",
            f"- adaptive_policy_reward: `{fmt_report_num((recommendation or {}).get('adaptive_policy', {}).get('mean_reward') if recommendation else None)}`",
            f"- recommended_fixed_quant: `{(recommended_quant or {}).get('signature', 'n/a')}`",
            f"- recommended_fixed_reward: `{fmt_report_num((recommended_quant or {}).get('evaluation', {}).get('mean_reward') if isinstance(recommended_quant, dict) else None)}`",
            f"- full recommendation JSON: `{config.recommendation_path()}`",
            "",
            "## Benchmark comparisons",
            "",
            *benchmark_table_lines,
            "",
            "## GPU / VRAM / Preflight",
        ]
    )
    if gpu_profile_report:
        lines.extend(
            [
                "",
                "### GPU profile",
                *md_table(["field", "value"], _dict_table_rows(gpu_profile_report)),
            ]
        )
    if preflight_report:
        lines.extend(
            ["", "### Preflight", *md_table(["field", "value"], _dict_table_rows(preflight_report))]
        )
    lines.extend(
        [
            "",
            f"- continuous_training: `{config.continuous_training}`",
            f"- max_training_episodes: `{config.max_training_episodes:,}`",
            f"- replay_buffer_capacity: `{config.replay_buffer_capacity:,}`",
        ]
    )
    if vram_report:
        lines.extend(["", *md_table(["vram metric", "value"], _dict_table_rows(vram_report))])
    else:
        lines.append("- vram: `n/a (cpu backend)`")
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "### Takeaways",
            *analysis_takeaway_lines(analysis),
            "",
            "### Figures",
            *_analysis_fig_links(),
            "",
            "### Analysis JSON (on disk)",
            *analysis_json_links,
            "",
            f"Full analysis tree: `{config.analysis_dir}/{config.run_name}/`",
        ]
    )
    write_text_file(report_path, "\n".join(lines) + "\n")
    return str(target)


__all__ = ["fmt_report_num", "maybe_report_link", "md_code_json", "write_research_report_markdown"]
