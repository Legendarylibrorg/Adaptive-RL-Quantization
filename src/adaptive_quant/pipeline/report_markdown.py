from __future__ import annotations

import json
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import md_table, write_text_file


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

    def _md_code_json(obj: object) -> str:
        return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```"

    def _fmt_num(value: object, *, digits: int = 2) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, (int, float)):
            return f"{value:.{digits}f}"
        return str(value)

    def _maybe_link(rel_path: Path) -> str:
        abs_path = report_dir / rel_path
        if abs_path.exists():
            return f"[{rel_path.name}]({rel_path.as_posix()})"
        return f"`{rel_path.as_posix()}` (missing)"

    def _analysis_fig_links() -> list[str]:
        analysis_root = Path(config.analysis_dir) / config.run_name
        rel_root = Path("..") / "analysis" / config.run_name
        candidates = [
            ("hardware reward", analysis_root / "hardware" / "hardware_generalization_reward.svg"),
            ("hardware latency", analysis_root / "hardware" / "hardware_generalization_latency.svg"),
            ("input complexity vs bits", analysis_root / "inputs" / "input_complexity_vs_bits.svg"),
            ("input adaptation scatter", analysis_root / "inputs" / "input_adaptation_scatter.svg"),
            ("quant function params", analysis_root / "quant" / "quant_function_parameters.svg"),
            ("training reward curve", analysis_root / "training" / "training_reward_curve.svg"),
        ]
        lines: list[str] = []
        for label, abs_path in candidates:
            rel_path = rel_root / abs_path.relative_to(analysis_root)
            lines.append(f"- {label}: {_maybe_link(rel_path)}")
        return lines

    eval_metrics = [
        ("mean_reward", eval_summary.get("mean_reward")),
        ("mean_latency_ms", eval_summary.get("mean_latency_ms")),
        ("mean_throughput_tps", eval_summary.get("mean_throughput_tps")),
        ("mean_memory_mb", eval_summary.get("mean_memory_mb")),
        ("mean_perplexity", eval_summary.get("mean_perplexity")),
        ("mean_stability_penalty", eval_summary.get("mean_stability_penalty")),
    ]
    eval_rows = [[k, _fmt_num(v)] for k, v in eval_metrics]
    recommendation = recommendation_summary if isinstance(recommendation_summary, dict) else None
    recommended_quant = recommendation.get("recommended_quant") if isinstance(recommendation, dict) else None

    bench = benchmark_summary
    single_vs_multi = bench.get("single_vs_multi") if isinstance(bench, dict) else None
    static_vs_dynamic = bench.get("static_vs_dynamic") if isinstance(bench, dict) else None
    discrete_vs_learned = bench.get("discrete_vs_learned") if isinstance(bench, dict) else None

    key_results_lines: list[str] = []
    if isinstance(single_vs_multi, dict):
        key_results_lines.append(
            "- universal policy gap improvement: "
            f"**{_fmt_num(single_vs_multi.get('generalization_gap_improvement'))}** "
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
                    f"{_fmt_num(s.get('mean_reward'))} → {_fmt_num(d.get('mean_reward'))}"
                )
                key_results_lines.append(
                    "- static → dynamic stability penalty: "
                    f"{_fmt_num(s.get('mean_stability_penalty'))} → {_fmt_num(d.get('mean_stability_penalty'))}"
                )
    if isinstance(discrete_vs_learned, dict):
        evaluation = discrete_vs_learned.get("evaluation", {})
        if isinstance(evaluation, dict):
            learned_metrics = evaluation.get("learned", {})
            discrete_metrics = evaluation.get("discrete", {})
            if isinstance(discrete_metrics, dict) and isinstance(learned_metrics, dict):
                key_results_lines.append(
                    "- discrete → learned reward: "
                    f"{_fmt_num(discrete_metrics.get('mean_reward'))} → {_fmt_num(learned_metrics.get('mean_reward'))}"
                )
                key_results_lines.append(
                    "- discrete → learned latency (ms): "
                    f"{_fmt_num(discrete_metrics.get('mean_latency_ms'))} → {_fmt_num(learned_metrics.get('mean_latency_ms'))}"
                )

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
        f"- mean_reward: `{_fmt_num(train_summary.get('mean_reward'))}`",
        f"- best_reward: `{_fmt_num(train_summary.get('best_reward'))}`",
        f"- final_reward: `{_fmt_num(train_summary.get('final_reward'))}`",
        f"- history: `{history_path or 'not written'}`",
        f"- checkpoint: `{checkpoint_path or 'not written'}`",
        "",
        "## Evaluation",
        *md_table(["metric", "value"], eval_rows),
        "",
        "## Recommendation",
        f"- target_hardware: `{recommendation.get('target_hardware') if recommendation else 'n/a'}`",
        f"- detected_hardware: `{recommendation.get('detected_hardware') if recommendation else 'n/a'}`",
        f"- adaptive_policy_reward: `{_fmt_num((recommendation or {}).get('adaptive_policy', {}).get('mean_reward') if recommendation else None)}`",
        f"- recommended_fixed_quant: `{(recommended_quant or {}).get('signature', 'n/a')}`",
        f"- recommended_fixed_reward: `{_fmt_num((recommended_quant or {}).get('evaluation', {}).get('mean_reward') if isinstance(recommended_quant, dict) else None)}`",
        "",
        "## Benchmarks",
        "",
        "### Single-hardware vs multi-hardware",
        _md_code_json(single_vs_multi) if single_vs_multi is not None else "_not run_",
        "",
        "### Static vs dynamic",
        _md_code_json(static_vs_dynamic) if static_vs_dynamic is not None else "_not run_",
        "",
        "### Discrete vs learned",
        _md_code_json(discrete_vs_learned) if discrete_vs_learned is not None else "_not run_",
        "",
        "## GPU / VRAM / Preflight",
        f"- gpu profile: `{gpu_profile_report or 'n/a'}`",
        f"- preflight: `{preflight_report or 'n/a'}`",
        f"- continuous_training: `{config.continuous_training}`",
        f"- max_training_episodes: `{config.max_training_episodes:,}`",
        f"- replay_buffer_capacity: `{config.replay_buffer_capacity:,}`",
        *(
            [
                f"- vram_allocated_mb: `{vram_report.get('vram_allocated_mb', 'n/a')}`",
                f"- vram_reserved_mb: `{vram_report.get('vram_reserved_mb', 'n/a')}`",
                f"- replay_buffer_mb: `{vram_report.get('replay_buffer_mb', 'n/a')}`",
                f"- replay_buffer_entries: `{vram_report.get('replay_buffer_entries', 'n/a')}`",
            ]
            if vram_report
            else ["- vram: `n/a (cpu backend)`"]
        ),
        "",
        "## Analysis",
        "### Figures",
        *_analysis_fig_links(),
        "",
        "### Analysis summaries",
        _md_code_json(analysis),
    ]
    write_text_file(report_path, "\n".join(lines) + "\n")
    return str(target)
