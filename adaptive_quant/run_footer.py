"""Consistent end-of-run CLI output for research entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adaptive_quant.configuration import FrameworkConfig


def _fmt_scalar(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _metric_rows(prefix: str, data: Mapping[str, Any], keys: tuple[str, ...]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for k in keys:
        if k not in data:
            continue
        rows.append((f"{prefix}_{k}", _fmt_scalar(data[k])))
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
    summary_path = f"{config.benchmark_dir}/{config.run_name}_summary.json"
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

    train = summary.get("train")
    if isinstance(train, dict):
        rows.extend(_metric_rows("train", train, ("mean_reward", "episodes", "updates")))
    ev = summary.get("evaluation")
    if isinstance(ev, dict):
        rows.extend(
            _metric_rows("eval", ev, ("mean_reward", "mean_latency_ms", "mean_throughput_tps"))
        )

    print_cli_block("Run complete", rows)


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
    *,
    summary_path: str,
    bootstrap: Mapping[str, Any],
    online: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    online_detail = f"{config.benchmark_dir}/{config.run_name}_online_summary.json"
    rows: list[tuple[str, str]] = [
        ("run_name", config.run_name),
        ("bootstrap_mean_reward", _fmt_scalar(bootstrap.get("mean_reward", 0.0))),
        ("online_requests", str(online.get("requests", ""))),
        ("online_mean_served_reward", _fmt_scalar(online.get("mean_served_reward", 0.0))),
        ("online_total_updates", str(online.get("total_updates", 0))),
        ("eval_mean_reward", _fmt_scalar(evaluation.get("mean_reward", 0.0))),
        ("summary_json", summary_path),
        ("online_detail_json", online_detail),
        ("analysis_dir", f"{config.analysis_dir}/{config.run_name}/"),
    ]
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
                f"{hw}: latency×{_fmt_scalar(lat)} throughput×{_fmt_scalar(thr)} mem×{_fmt_scalar(mem)}"
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
]
