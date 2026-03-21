from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, load_jsonl, write_bar_chart, write_json, write_scatter_plot
from adaptive_quant.math_utils import mean


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    variant_usage: dict[str, float] = {}
    expert_frequency: dict[str, float] = {}
    sensitivity_vs_aggressiveness: list[tuple[float, float]] = []
    router_entropy: list[float] = []

    for record in records:
        moe_context = record.get("moe_context") or {}
        experts = moe_context.get("experts") or []
        decision = record.get("decision") or {}
        variant_names = decision.get("moe_variant_names") or []
        router_entropy.append(float(moe_context.get("router_entropy", 0.0)))
        for expert, variant_name in zip(experts, variant_names):
            expert_key = f"expert_{int(expert.get('expert_index', 0))}"
            expert_frequency[expert_key] = expert_frequency.get(expert_key, 0.0) + 1.0
            variant_usage[variant_name] = variant_usage.get(variant_name, 0.0) + 1.0
            aggressiveness = {"safe": 0.0, "balanced": 0.5, "aggressive": 1.0}.get(variant_name, 0.5)
            sensitivity_vs_aggressiveness.append((float(expert.get("sensitivity", 0.0)), aggressiveness))

    top_experts = dict(sorted(expert_frequency.items(), key=lambda item: item[1], reverse=True)[: min(8, len(expert_frequency))])
    summary = {
        "log_path": log_path,
        "mean_router_entropy": mean(router_entropy),
        "variant_usage": variant_usage,
        "top_experts": top_experts,
        "mean_aggressiveness": mean([point[1] for point in sensitivity_vs_aggressiveness]),
        "selection_count": len(sensitivity_vs_aggressiveness),
    }
    write_json(str(output_root / "moe_expert_behavior_summary.json"), summary)
    write_bar_chart(str(output_root / "moe_variant_usage.svg"), "MoE Variant Usage", variant_usage, "Selections")
    write_bar_chart(str(output_root / "moe_top_experts.svg"), "Most Active Experts", top_experts, "Selections")
    write_scatter_plot(
        str(output_root / "moe_sensitivity_vs_aggressiveness.svg"),
        "Expert Sensitivity vs Variant Aggressiveness",
        sensitivity_vs_aggressiveness,
        "Expert sensitivity",
        "Variant aggressiveness",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/moe_expert_behavior.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote MoE expert analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)
