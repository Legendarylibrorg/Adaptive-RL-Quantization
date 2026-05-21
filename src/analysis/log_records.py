"""Shared JSONL / episode record helpers for ``analysis.analyzers``."""

from __future__ import annotations

from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, grouped_mean, write_bar_chart, write_scatter_plot
from adaptive_quant.features import complexity_bucket
from adaptive_quant.logging_utils import load_jsonl, write_json
from adaptive_quant.math_utils import mean

DEFAULT_ANALYSIS_PHASE = "eval"


def filter_phase(records: list[dict], phase: str | None) -> list[dict]:
    if phase is None:
        return records
    if not any("phase" in record for record in records):
        return records
    return [record for record in records if record.get("phase") == phase]


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def mean_effective_bits(decision: dict) -> float:
    bits = decision.get("effective_layer_bits", [])
    return mean([float(b) for b in bits]) if bits else 0.0


def training_step_reward(record: dict) -> float:
    return float(record.get("batch_reward", record.get("reward", 0.0)))


def served_reward(record: dict) -> float:
    return float(record.get("served_metrics", {}).get("reward", 0.0))


def input_complexity(record: dict) -> float:
    return float(record.get("input_features", {}).get("complexity_score", 0.0))


def mean_flag_rate(records: list[dict], key: str) -> float:
    return mean([1.0 if r.get(key) else 0.0 for r in records])


def by_hardware(records: list[dict], metric_path: tuple[str, ...]) -> dict[str, float]:
    return grouped_mean(records, "hardware_mode", metric_path)


def jsonl_analysis_setup(
    log_path: str,
    output_dir: str,
    *,
    phase: str | None = DEFAULT_ANALYSIS_PHASE,
) -> tuple[list[dict], Path]:
    records = filter_phase(load_jsonl(log_path), phase)
    return records, ensure_directory(output_dir)


def bucket_records_by_complexity(records: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    for record in records:
        buckets[complexity_bucket(input_complexity(record))].append(record)
    return buckets


def write_analysis_artifacts(
    output_root: Path,
    summary: dict[str, object],
    *,
    json_name: str,
    bar_charts: list[tuple[str, str, dict[str, float], str]] | None = None,
    scatter_charts: list[tuple[str, str, list[tuple[float, float]], str, str]] | None = None,
) -> None:
    write_json(str(output_root / json_name), summary)
    for filename, title, values, y_label in bar_charts or ():
        write_bar_chart(str(output_root / filename), title, values, y_label)
    for filename, title, points, x_label, y_label in scatter_charts or ():
        write_scatter_plot(str(output_root / filename), title, points, x_label, y_label)


def complexity_bucket_metrics(bucket_records: list[dict]) -> dict[str, float | int]:
    avg_bits, avg_perplexity, avg_reward = [], [], []
    for record in bucket_records:
        decision = record.get("decision", {})
        metrics = record.get("metrics", {})
        if decision.get("effective_layer_bits"):
            avg_bits.append(mean_effective_bits(decision))
        if "perplexity" in metrics:
            avg_perplexity.append(float(metrics["perplexity"]))
        if "reward" in metrics:
            avg_reward.append(float(metrics["reward"]))
    return {
        "average_bits": mean(avg_bits),
        "average_perplexity": mean(avg_perplexity),
        "average_reward": mean(avg_reward),
        "count": len(bucket_records),
    }


__all__ = [
    "DEFAULT_ANALYSIS_PHASE",
    "bucket_records_by_complexity",
    "by_hardware",
    "complexity_bucket_metrics",
    "filter_phase",
    "input_complexity",
    "jsonl_analysis_setup",
    "mean_effective_bits",
    "mean_flag_rate",
    "served_reward",
    "summary_stats",
    "training_step_reward",
    "write_analysis_artifacts",
]
