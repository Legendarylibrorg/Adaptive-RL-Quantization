from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig


def run_research_analysis(config: FrameworkConfig, history_path: str | None) -> dict[str, object]:
    from analysis.analyzers import (
        analyze_hardware,
        analyze_inputs,
        analyze_moe_cache,
        analyze_moe_experts,
        analyze_quant,
        analyze_training_dynamics,
    )

    analysis_root = f"{config.analysis_dir}/{config.run_name}"
    analysis: dict[str, object] = {
        "hardware": analyze_hardware(
            f"{config.log_dir}/{config.run_name}_multi_hw.jsonl", f"{analysis_root}/hardware"
        ),
        "input": analyze_inputs(
            f"{config.log_dir}/{config.run_name}_dynamic.jsonl", f"{analysis_root}/inputs"
        ),
        "quant_function": analyze_quant(
            f"{config.log_dir}/{config.run_name}_learned.jsonl", f"{analysis_root}/quant"
        ),
    }
    if config.moe_enabled:
        analysis["moe_experts"] = analyze_moe_experts(
            f"{config.log_dir}/{config.run_name}.jsonl", f"{analysis_root}/moe_experts"
        )
        analysis["moe_cache"] = analyze_moe_cache(
            f"{config.log_dir}/{config.run_name}.jsonl", f"{analysis_root}/moe_cache"
        )
    if history_path is not None:
        analysis["training_dynamics"] = analyze_training_dynamics(
            history_path, f"{analysis_root}/training"
        )
    return analysis
