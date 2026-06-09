from __future__ import annotations

from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from adaptive_quant.logging_utils import md_table, write_json, write_text_file
from adaptive_quant.online_learning import OnlineLearningLoop, build_request_stream
from adaptive_quant.pipeline.output_summary import (
    online_analysis_takeaway_lines,
    slim_online_analysis_for_summary,
)
from adaptive_quant.pipeline.report_markdown import fmt_report_num, maybe_report_link
from adaptive_quant.pipeline.research_contract import build_research_contract
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.research_pipeline import maybe_save_final_checkpoint, write_training_history
from adaptive_quant.security_audit import build_security_audit_record
from adaptive_quant.security_bypass import enforce_security_bypass_policy
from adaptive_quant.trainer import build_trainer


def run_online_pipeline(
    config: FrameworkConfig,
    *,
    request_count: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    summary_path = config.summary_path()
    trainer = build_trainer(config)
    git_commit = git_commit_hash()
    loop: OnlineLearningLoop | None = None
    pipeline_error: Exception | None = None
    bootstrap_summary: dict[str, object] = {}
    online_summary: dict[str, object] = {}
    eval_summary: dict[str, object] = {}
    online_analysis: dict[str, object] = {}
    history_path: str | None = None
    checkpoint_path: str | None = None
    report_path: str | None = None

    enforce_security_bypass_policy(context="online pipeline")

    try:
        bootstrap_summary = trainer.train()
        loop = OnlineLearningLoop(config, trainer=trainer)
        online_summary = loop.run_stream(build_request_stream(config, request_count=request_count))
        eval_summary = trainer.evaluate()

        analysis_root = f"{config.analysis_dir}/{config.run_name}"
        from analysis.analyzers import analyze_online

        online_analysis = analyze_online(config.online_telemetry_path(), f"{analysis_root}/online")
        history_path = write_training_history(config, trainer)
        checkpoint_path = maybe_save_final_checkpoint(config, trainer)
        report_path = _write_online_report(
            config,
            git_commit=git_commit,
            summary_path=summary_path,
            bootstrap_summary=bootstrap_summary,
            online_summary=online_summary,
            eval_summary=eval_summary,
            online_analysis=online_analysis,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        pipeline_error = exc
    finally:
        if loop is not None:
            loop.close()
        trainer.close()

    if pipeline_error is not None:
        raise pipeline_error

    summary = {
        "config": config_to_flat_dict(config),
        "git_commit": git_commit,
        "research": build_research_contract(
            config,
            git_commit=git_commit,
            pipeline="online_adaptation",
            phases=["bootstrap_train", "online_stream", "evaluate", "analysis", "report"],
        ),
        "security_audit": build_security_audit_record(
            config,
            cli_startup_overrides=cli_startup_overrides,
        ),
        "bootstrap_train": bootstrap_summary,
        "online": online_summary,
        "evaluation": eval_summary,
        "analysis": {
            "online_learning": slim_online_analysis_for_summary(online_analysis),
        },
        "artifacts": {
            "training_history": history_path,
            "final_checkpoint": checkpoint_path,
            "online_detail": config.online_summary_path(),
            "online_telemetry": config.online_telemetry_path(),
            "online_replay": config.online_replay_path(),
            "report": report_path,
        },
    }
    write_json(summary_path, summary)
    return summary


def run_online_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    request_count: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
    footer_mode: str = "full",
) -> dict[str, object]:
    from adaptive_quant.run_footer import print_online_footer

    summary = run_online_pipeline(
        config,
        request_count=request_count,
        cli_startup_overrides=cli_startup_overrides,
    )
    print_online_footer(config, summary, mode=footer_mode)
    return summary


def _write_online_report(
    config: FrameworkConfig,
    *,
    git_commit: str | None,
    summary_path: str,
    bootstrap_summary: dict[str, object],
    online_summary: dict[str, object],
    eval_summary: dict[str, object],
    online_analysis: dict[str, object],
    history_path: str | None,
    checkpoint_path: str | None,
) -> str | None:
    if not config.write_research_report:
        return None

    report_path = config.report_path()
    target = Path(report_path)
    report_dir = target.parent
    analysis_root = Path(config.analysis_dir) / config.run_name / "online"
    rel_analysis_root = Path("..") / "analysis" / config.run_name / "online"

    def _analysis_links() -> list[str]:
        candidates = [
            ("online reward by hardware", analysis_root / "online_reward_by_hardware.svg"),
            ("online complexity vs reward", analysis_root / "online_complexity_vs_reward.svg"),
        ]
        lines: list[str] = []
        for label, abs_path in candidates:
            rel_path = rel_analysis_root / abs_path.relative_to(analysis_root)
            lines.append(f"- {label}: {maybe_report_link(report_dir, rel_path)}")
        return lines

    bootstrap_rows = [
        [key, fmt_report_num(bootstrap_summary.get(key))]
        for key in ("mean_reward", "min_reward", "max_reward", "episodes", "updates")
        if key in bootstrap_summary
    ]
    online_rows = [
        [key, fmt_report_num(online_summary.get(key))]
        for key in (
            "requests",
            "mean_served_reward",
            "mean_candidate_reward",
            "exploration_rate_observed",
            "canary_rate_observed",
            "candidate_accept_rate",
            "total_updates",
            "total_rollbacks",
            "replay_size",
        )
        if key in online_summary
    ]
    eval_rows = [
        [key, fmt_report_num(eval_summary.get(key))]
        for key in (
            "mean_reward",
            "mean_latency_ms",
            "mean_throughput_tps",
            "mean_memory_mb",
            "mean_perplexity",
        )
        if key in eval_summary
    ]
    lines = [
        "# Online Adaptation Report",
        "",
        "## Overview",
        f"- run_name: `{config.run_name}`",
        f"- git_commit: `{git_commit or 'unknown'}`",
        f"- training_backend: `{config.training_backend}`",
        f"- summary_json: `{summary_path}`",
        f"- online_detail_json: `{config.online_summary_path()}`",
        f"- telemetry_jsonl: `{config.online_telemetry_path()}`",
        f"- replay_jsonl: `{config.online_replay_path()}`",
        f"- history: `{history_path or 'not written'}`",
        f"- checkpoint: `{checkpoint_path or 'not written'}`",
        "",
        "## Bootstrap",
        *(md_table(["metric", "value"], bootstrap_rows) if bootstrap_rows else ["_not written_"]),
        "",
        "## Online",
        *(md_table(["metric", "value"], online_rows) if online_rows else ["_not written_"]),
        "",
        "## Evaluation",
        *(md_table(["metric", "value"], eval_rows) if eval_rows else ["_not written_"]),
        "",
        "## Analysis",
        "### Figures",
        *_analysis_links(),
        "",
        "### Takeaways",
        *online_analysis_takeaway_lines(online_analysis),
    ]
    write_text_file(report_path, "\n".join(lines) + "\n")
    return str(target)


__all__ = ["run_online_pipeline", "run_online_pipeline_entrypoint"]
