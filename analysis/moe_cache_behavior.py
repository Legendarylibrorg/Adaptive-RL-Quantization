from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, grouped_mean, load_jsonl, write_bar_chart, write_json, write_scatter_plot
from adaptive_quant.math_utils import mean


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    cache_vs_latency: list[tuple[float, float]] = []
    entropy_vs_reward: list[tuple[float, float]] = []
    swap_costs = []
    cache_misses = []
    for record in records:
        metrics = record.get("metrics", {})
        moe_context = record.get("moe_context") or {}
        cache_miss = float(metrics.get("cache_miss_count", 0.0))
        latency = float(metrics.get("latency_ms", 0.0))
        reward = float(metrics.get("reward", 0.0))
        router_entropy = float(moe_context.get("router_entropy", 0.0))
        swap_cost = float(metrics.get("swap_cost_ms", 0.0))
        cache_vs_latency.append((cache_miss, latency))
        entropy_vs_reward.append((router_entropy, reward))
        swap_costs.append(swap_cost)
        cache_misses.append(cache_miss)

    reward_by_hardware = grouped_mean(records, "hardware_mode", ("metrics", "reward"))
    summary = {
        "log_path": log_path,
        "mean_swap_cost_ms": mean(swap_costs),
        "mean_cache_miss_count": mean(cache_misses),
        "reward_by_hardware": reward_by_hardware,
    }
    write_json(str(output_root / "moe_cache_behavior_summary.json"), summary)
    write_bar_chart(
        str(output_root / "moe_cache_metrics.svg"),
        "MoE Cache Metrics",
        {"swap_cost_ms": summary["mean_swap_cost_ms"], "cache_miss_count": summary["mean_cache_miss_count"]},
        "Average value",
    )
    write_scatter_plot(
        str(output_root / "moe_cache_miss_vs_latency.svg"),
        "Cache Misses vs Latency",
        cache_vs_latency,
        "Cache miss count",
        "Latency (ms)",
    )
    write_scatter_plot(
        str(output_root / "moe_router_entropy_vs_reward.svg"),
        "Router Entropy vs Reward",
        entropy_vs_reward,
        "Router entropy",
        "Reward",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/moe_cache_behavior.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote MoE cache analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)
