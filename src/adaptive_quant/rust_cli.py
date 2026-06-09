"""Invoke optional Rust CLI accelerators (Python orchestrator stays canonical)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.types import BackendMetricDict, EpisodeState, QuantizationDecision

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUST_BINARY_NAMES = (
    "adaptive-rl-quant-rust",
    "adaptive-rl-quant-rust.exe",
)


class RustCliError(RuntimeError):
    """Rust CLI subprocess failed or returned invalid JSON."""


def resolve_rust_cli_binary(config: FrameworkConfig) -> str | None:
    if config.rust_cli_binary:
        path = Path(config.rust_cli_binary)
        return str(path) if path.is_file() else None
    for rel in (
        "rust/target/release/adaptive-rl-quant-rust",
        "rust/target/release/adaptive-rl-quant-rust.exe",
        "rust/target/debug/adaptive-rl-quant-rust",
        "rust/target/debug/adaptive-rl-quant-rust.exe",
    ):
        candidate = _REPO_ROOT / rel
        if candidate.is_file():
            return str(candidate)
    return None


def rust_simulator_available(config: FrameworkConfig) -> bool:
    return bool(
        config.rust_simulator_enabled
        and config.backend == "simulator"
        and not config.moe_enabled
        and resolve_rust_cli_binary(config) is not None
    )


def _payload_for_eval(
    state: EpisodeState,
    decision: QuantizationDecision,
    config: FrameworkConfig,
) -> dict[str, Any]:
    hw = state.hardware_profile
    calibration = config.sim_calibration if isinstance(config.sim_calibration, dict) else {}
    return {
        "hardware": {
            "hardware_type": hw.hardware_type.value,
            "compute_factor": hw.compute_factor,
            "throughput_bias": hw.throughput_bias,
            "latency_bias": hw.latency_bias,
            "memory_budget_mb": hw.memory_budget_mb,
            "preferred_bits": hw.preferred_bits,
            "kernel_uniformity_preference": hw.kernel_uniformity_preference,
        },
        "input_features": {
            "prompt_length": state.input_features.prompt_length,
            "complexity_score": state.input_features.complexity_score,
        },
        "sensitivity": {
            "layer_stats": list(state.sensitivity.layer_stats),
        },
        "decision": {
            "mode": decision.mode.value,
            "scale_factor": decision.scale_factor,
            "clipping_range": decision.clipping_range,
            "effective_layer_bits": list(decision.effective_layer_bits),
        },
        "calibration": calibration,
    }


def run_rust_sim_eval(
    config: FrameworkConfig,
    state: EpisodeState,
    decision: QuantizationDecision,
    *,
    binary: str | None = None,
) -> BackendMetricDict:
    """Run ``adaptive-rl-quant-rust sim-eval``; return parsed metrics JSON."""
    cli = binary or resolve_rust_cli_binary(config)
    if not cli:
        raise RustCliError("Rust CLI binary not found; run scripts/build_rust.sh")
    payload = json.dumps(_payload_for_eval(state, decision, config), separators=(",", ":"))
    try:
        completed = subprocess.run(
            [cli, "sim-eval"],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RustCliError(str(exc)) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RustCliError(stderr or f"rust sim-eval exited {completed.returncode}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RustCliError(f"invalid JSON from rust sim-eval: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RustCliError("rust sim-eval did not return a JSON object")
    return parsed  # type: ignore[return-value]


__all__ = [
    "RustCliError",
    "resolve_rust_cli_binary",
    "run_rust_sim_eval",
    "rust_simulator_available",
]
