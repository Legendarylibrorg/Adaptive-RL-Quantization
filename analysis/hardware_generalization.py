from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, grouped_mean, load_jsonl, write_bar_chart, write_json


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    reward_by_hardware = grouped_mean(records, "hardware_mode", ("metrics", "reward"))
    latency_by_hardware = grouped_mean(records, "hardware_mode", ("metrics", "latency_ms"))
    throughput_by_hardware = grouped_mean(records, "hardware_mode", ("metrics", "throughput_tps"))
    perplexity_by_hardware = grouped_mean(records, "hardware_mode", ("metrics", "perplexity"))

    rewards = list(reward_by_hardware.values())
    summary = {
        "log_path": log_path,
        "reward_by_hardware": reward_by_hardware,
        "latency_by_hardware": latency_by_hardware,
        "throughput_by_hardware": throughput_by_hardware,
        "perplexity_by_hardware": perplexity_by_hardware,
        "generalization_gap": (max(rewards) - min(rewards)) if rewards else 0.0,
    }
    write_json(str(output_root / "hardware_generalization_summary.json"), summary)
    write_bar_chart(str(output_root / "hardware_generalization_reward.svg"), "Policy Reward by Hardware", reward_by_hardware, "Reward")
    write_bar_chart(str(output_root / "hardware_generalization_latency.svg"), "Latency by Hardware", latency_by_hardware, "Latency (ms)")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/hardware_generalization.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote hardware analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)

