from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, load_jsonl, write_bar_chart, write_json, write_scatter_plot
from adaptive_quant.math_utils import mean


def _complexity_bucket(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.70:
        return "medium"
    return "high"


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    points: list[tuple[float, float]] = []

    for record in records:
        input_features = record.get("input_features", {})
        decision = record.get("decision", {})
        complexity = float(input_features.get("complexity_score", 0.0))
        effective_bits = decision.get("effective_layer_bits", [])
        average_bits = mean([float(bit) for bit in effective_bits]) if effective_bits else 0.0
        points.append((complexity, average_bits))
        buckets[_complexity_bucket(complexity)].append(record)

    summary = {
        "log_path": log_path,
        "by_complexity": {},
    }
    for bucket_name, bucket_records in buckets.items():
        avg_bits = []
        avg_perplexity = []
        avg_reward = []
        for record in bucket_records:
            decision = record.get("decision", {})
            metrics = record.get("metrics", {})
            bits = decision.get("effective_layer_bits", [])
            if bits:
                avg_bits.append(mean([float(bit) for bit in bits]))
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
    write_bar_chart(
        str(output_root / "input_complexity_vs_bits.svg"),
        "Average Precision by Input Complexity",
        {bucket: values["average_bits"] for bucket, values in summary["by_complexity"].items()},
        "Average effective bits",
    )
    write_scatter_plot(
        str(output_root / "input_adaptation_scatter.svg"),
        "Complexity vs Precision",
        points,
        "Input complexity",
        "Average effective bits",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/input_adaptation.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote input adaptation analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)

