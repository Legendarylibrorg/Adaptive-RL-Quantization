from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, grouped_mean, load_jsonl, write_bar_chart, write_json, write_scatter_plot
from adaptive_quant.math_utils import mean


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    reward_by_hardware = grouped_mean(records, "hardware_mode", ("served_metrics", "reward"))
    accept_rate = mean([1.0 if record.get("accepted_candidate") else 0.0 for record in records])
    update_rate = mean([1.0 if record.get("online_update_applied") else 0.0 for record in records])
    rollback_count = sum(1 for record in records if record.get("drift_event") == "rollback")
    complexity_reward_points = [
        (
            float(record.get("input_features", {}).get("complexity_score", 0.0)),
            float(record.get("served_metrics", {}).get("reward", 0.0)),
        )
        for record in records
    ]

    summary = {
        "log_path": log_path,
        "records": len(records),
        "reward_by_hardware": reward_by_hardware,
        "candidate_accept_rate": accept_rate,
        "online_update_rate": update_rate,
        "rollback_count": rollback_count,
        "mean_served_reward": mean([float(record.get("served_metrics", {}).get("reward", 0.0)) for record in records]),
    }
    write_json(str(output_root / "online_learning_summary.json"), summary)
    write_bar_chart(str(output_root / "online_reward_by_hardware.svg"), "Online Reward by Hardware", reward_by_hardware, "Reward")
    write_scatter_plot(
        str(output_root / "online_complexity_vs_reward.svg"),
        "Input Complexity vs Served Reward",
        complexity_reward_points,
        "Complexity",
        "Reward",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/online_learning.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote online analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)
