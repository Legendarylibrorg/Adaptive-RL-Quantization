"""Machine-readable pipeline topology — how config flows through layers to artifacts."""

from __future__ import annotations

from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.sections import (
    artifact_layout,
    default_route_catalog_path,
    default_route_models_dir,
)


def _artifact_paths_for_topology(config: FrameworkConfig) -> list[str]:
    layout = artifact_layout(config.outputs_dir)
    return [
        layout["log_dir"],
        layout["benchmark_dir"],
        layout["analysis_dir"],
        layout["checkpoint_dir"],
        layout["report_dir"],
        f"{config.outputs_dir.rstrip('/\\')}/paper_bundles",
        layout["gguf_export_dir"],
        str(Path(default_route_catalog_path(config.outputs_dir)).parent),
        default_route_models_dir(config.outputs_dir),
    ]


def infer_simulator_engine(config: FrameworkConfig) -> str | None:
    if config.backend != "simulator":
        return None
    from adaptive_quant.rust_cli import rust_simulator_available

    if rust_simulator_available(config):
        return "rust_cli"
    return "python"


def build_pipeline_topology(config: FrameworkConfig) -> dict[str, object]:
    """Describe research pipeline layers and active components for ``research.topology``."""
    sim_engine = infer_simulator_engine(config)
    measurement: list[str] = []
    if config.backend == "simulator":
        measurement.append(f"SimulatorBackend({sim_engine or 'python'})")
    elif config.backend == "llama_cpp":
        measurement.append("LlamaCppBackend")
    if config.router_enabled:
        measurement.append("in_run_router")
    if config.llama_cpp_gguf_export_enabled:
        measurement.append("gguf_export(llama.cpp quantize)")

    return {
        "orchestrator": "python",
        "entrypoints": _active_entrypoints(config),
        "layers": {
            "configuration": ["FrameworkConfig", "presets", "JSON/TOML"],
            "orchestration": [
                "research_pipeline",
                "online_pipeline",
                "multiseed",
                "sweep",
            ],
            "learning": _learning_components(config),
            "measurement": measurement,
            "analysis": ["src/analysis/analyzers", "pipeline/output_summary"],
            "artifacts": _artifact_paths_for_topology(config),
        },
        "active": {
            "backend": config.backend,
            "training_backend": config.training_backend,
            "simulator_engine": sim_engine,
            "moe_enabled": config.moe_enabled,
            "router_enabled": config.router_enabled,
            "gguf_export_enabled": bool(config.llama_cpp_gguf_export_enabled),
            "rust_simulator_enabled": bool(config.rust_simulator_enabled),
        },
    }


def _active_entrypoints(config: FrameworkConfig) -> list[str]:
    if config.online_learning:
        return ["adaptive-rl-quant-online"]
    if config.moe_enabled:
        return ["adaptive-rl-quant-moe"]
    if config.training_backend == "pytorch":
        return ["adaptive-rl-quant-pytorch"]
    return ["adaptive-rl-quant"]


def _learning_components(config: FrameworkConfig) -> list[str]:
    if config.training_backend == "pytorch":
        base = ["TorchTrainer", "torch_policy"]
    else:
        base = ["Trainer", "UniversalQuantizationPolicy"]
    if config.router_enabled:
        base.append("routing.Router")
    return base


__all__ = ["build_pipeline_topology", "infer_simulator_engine"]
