"""Consistent end-of-run CLI output for research entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import format_display


def _metric_rows(
    prefix: str, data: Mapping[str, Any], keys: tuple[str, ...]
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for k in keys:
        if k not in data:
            continue
        rows.append((f"{prefix}_{k}", format_display(data[k], style="footer")))
    return rows


def print_cli_block(title: str, rows: list[tuple[str, str]]) -> None:
    width = max((len(a) for a, _ in rows), default=0)
    print()
    print(f"── {title} ──")
    for a, b in rows:
        print(f"  {a:<{width}}  {b}")


def print_pipeline_footer(
    config: FrameworkConfig,
    summary: Mapping[str, Any],
    *,
    mode: str = "full",
) -> None:
    """End-of-run paths and headline metrics. ``mode``: full | minimal | none."""
    if mode == "none":
        return
    summary_path = config.summary_path()
    if mode == "minimal":
        print(f"[done] {config.run_name}  →  {summary_path}")
        return

    artifacts = summary.get("artifacts")
    art: dict[str, Any] = artifacts if isinstance(artifacts, dict) else {}

    rows: list[tuple[str, str]] = [
        ("run_name", config.run_name),
        ("training_backend", str(config.training_backend)),
        ("summary_json", summary_path),
    ]
    rp = art.get("report")
    if rp:
        rows.append(("report_md", str(rp)))
    rows.append(("analysis_dir", f"{config.analysis_dir}/{config.run_name}/"))

    th = art.get("training_history")
    if th:
        rows.append(("training_history", str(th)))
    ck = art.get("final_checkpoint")
    if ck:
        rows.append(("checkpoint", str(ck)))
    rq = art.get("recommendation")
    if rq:
        rows.append(("recommendation_json", str(rq)))

    train = summary.get("train")
    if isinstance(train, dict):
        rows.extend(_metric_rows("train", train, ("mean_reward", "episodes", "updates")))
    ev = summary.get("evaluation")
    if isinstance(ev, dict):
        rows.extend(
            _metric_rows("eval", ev, ("mean_reward", "mean_latency_ms", "mean_throughput_tps"))
        )
    recommendation = summary.get("recommendation")
    if isinstance(recommendation, dict):
        rows.append(
            (
                "target_hardware",
                format_display(recommendation.get("target_hardware"), style="footer"),
            )
        )
        fixed = recommendation.get("recommended_quant")
        if isinstance(fixed, dict):
            rows.append(
                ("recommended_quant", format_display(fixed.get("signature"), style="footer"))
            )
            evaluation = fixed.get("evaluation")
            if isinstance(evaluation, dict):
                rows.append(
                    (
                        "recommended_reward",
                        format_display(evaluation.get("mean_reward"), style="footer"),
                    )
                )
        decision = recommendation.get("decision")
        if isinstance(decision, dict) and decision.get("deploy"):
            rows.append(("deploy", format_display(decision.get("deploy"), style="footer")))

    print_cli_block("Run complete", rows)


def print_sweep_footer(
    *,
    sweep_run_name: str,
    objective: str,
    direction: str,
    trial_count: int,
    best_trial: object | None,
    aggregate_json: str,
    report_md: str,
) -> None:
    rows: list[tuple[str, str]] = [
        ("sweep_run", sweep_run_name),
        ("objective", f"{objective} ({direction})"),
        ("trials", str(trial_count)),
        ("aggregate_json", aggregate_json),
        ("aggregate_report", report_md),
    ]
    if best_trial is not None:
        plan = getattr(best_trial, "plan", None)
        objective_value = getattr(best_trial, "objective_value", None)
        if plan is not None:
            rows.append(("best_trial", f"#{plan.trial_id} ({plan.run_name_suffix})"))
        if objective_value is not None:
            rows.append(("best_objective", format_display(objective_value, style="footer")))
    print_cli_block("Sweep complete", rows)


def print_multiseed_footer(
    *,
    multiseed_run_name: str,
    seeds: list[int],
    aggregate_json: str,
    report_md: str,
    per_seed_summary_paths: list[str],
) -> None:
    rows: list[tuple[str, str]] = [
        ("multiseed_run", multiseed_run_name),
        ("seeds", ", ".join(str(s) for s in seeds)),
        ("aggregate_json", aggregate_json),
        ("aggregate_report", report_md),
        ("per_seed_summaries", f"{len(per_seed_summary_paths)} file(s)"),
    ]
    print_cli_block("Multiseed complete", rows)


def print_online_footer(
    config: FrameworkConfig,
    summary: Mapping[str, Any],
    *,
    mode: str = "full",
) -> None:
    if mode == "none":
        return
    summary_path = config.summary_path()
    if mode == "minimal":
        print(f"[done] {config.run_name}  →  {summary_path}")
        return

    bootstrap = summary.get("bootstrap_train")
    online = summary.get("online")
    evaluation = summary.get("evaluation")
    artifacts = summary.get("artifacts")
    bootstrap_map = bootstrap if isinstance(bootstrap, Mapping) else {}
    online_map = online if isinstance(online, Mapping) else {}
    evaluation_map = evaluation if isinstance(evaluation, Mapping) else {}
    artifact_map = artifacts if isinstance(artifacts, Mapping) else {}

    rows: list[tuple[str, str]] = [
        ("run_name", config.run_name),
        ("training_backend", str(config.training_backend)),
        (
            "bootstrap_mean_reward",
            format_display(bootstrap_map.get("mean_reward", 0.0), style="footer"),
        ),
        ("online_requests", str(online_map.get("requests", ""))),
        (
            "online_mean_served_reward",
            format_display(online_map.get("mean_served_reward", 0.0), style="footer"),
        ),
        ("online_total_updates", str(online_map.get("total_updates", 0))),
        (
            "eval_mean_reward",
            format_display(evaluation_map.get("mean_reward", 0.0), style="footer"),
        ),
        ("summary_json", summary_path),
        ("online_detail_json", str(artifact_map.get("online_detail", ""))),
        ("telemetry_jsonl", str(artifact_map.get("online_telemetry", ""))),
        ("replay_jsonl", str(artifact_map.get("online_replay", ""))),
        ("analysis_dir", f"{config.analysis_dir}/{config.run_name}/"),
    ]
    history_path = artifact_map.get("training_history")
    if history_path:
        rows.append(("training_history", str(history_path)))
    checkpoint_path = artifact_map.get("final_checkpoint")
    if checkpoint_path:
        rows.append(("checkpoint", str(checkpoint_path)))
    report_path = artifact_map.get("report")
    if report_path:
        rows.append(("report_md", str(report_path)))
    print_cli_block("Online run complete", rows)


def print_calibration_footer(
    *,
    run_name: str,
    out_path: str,
    calibration: Mapping[str, Any],
) -> None:
    bits: list[str] = []
    for hw, fits in sorted(calibration.items()):
        if not isinstance(fits, Mapping):
            continue
        lat = fits.get("latency_multiplier")
        thr = fits.get("throughput_multiplier")
        mem = fits.get("memory_multiplier")
        if lat is not None and thr is not None and mem is not None:
            bits.append(
                f"{hw}: latency×{format_display(lat, style='footer')} "
                f"throughput×{format_display(thr, style='footer')} "
                f"mem×{format_display(mem, style='footer')}"
            )
    rows: list[tuple[str, str]] = [
        ("run_name", run_name),
        ("calibration_json", out_path),
        ("fit_multipliers", "; ".join(bits) if bits else "(see JSON)"),
    ]
    print_cli_block("Calibration complete", rows)


__all__ = [
    "print_calibration_footer",
    "print_cli_block",
    "print_multiseed_footer",
    "print_online_footer",
    "print_pipeline_footer",
    "print_sweep_footer",
]
