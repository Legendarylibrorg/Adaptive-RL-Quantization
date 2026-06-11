"""Invoke optional Rust CLI accelerators (Python orchestrator stays canonical)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import validate_optional_filesystem_path
from adaptive_quant.repo_paths import default_rust_binary_paths, find_repo_root
from adaptive_quant.types import BackendMetricDict, EpisodeState, QuantizationDecision

_RUST_CLI_ENV = "ADAPTIVE_RL_RUST_CLI"
_BUILD_HINT = "Build with ./scripts/build_rust.sh from the repo root, set rust_cli_binary, or install adaptive-rl-quant-rust on PATH."


class RustCliError(RuntimeError):
    """Rust CLI subprocess failed or returned invalid JSON."""


def resolve_rust_cli_binary(config: FrameworkConfig) -> str | None:
    """Resolve the Rust CLI binary (explicit config → env → PATH → repo target/)."""
    if config.rust_cli_binary:
        path = Path(config.rust_cli_binary).expanduser()
        return str(path.resolve()) if path.is_file() else None

    env_cli = os.environ.get(_RUST_CLI_ENV, "").strip()
    if env_cli:
        validate_optional_filesystem_path(_RUST_CLI_ENV, env_cli)
        path = Path(env_cli).expanduser()
        if path.is_file():
            return str(path.resolve())

    which = shutil.which("adaptive-rl-quant-rust")
    if which:
        return which

    repo = find_repo_root()
    if repo is not None:
        for candidate in default_rust_binary_paths(repo):
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def rust_simulator_available(config: FrameworkConfig) -> bool:
    return bool(
        config.rust_simulator_enabled
        and config.backend == "simulator"
        and not config.moe_enabled
        and resolve_rust_cli_binary(config) is not None
    )


def rust_cli_status(config: FrameworkConfig) -> dict[str, object]:
    """Diagnostic block for research summaries and env reports."""
    binary = resolve_rust_cli_binary(config)
    repo = find_repo_root()
    return {
        "enabled": config.rust_simulator_enabled,
        "available": rust_simulator_available(config),
        "resolved_binary": binary,
        "repo_root": str(repo) if repo is not None else None,
        "build_script": "scripts/build_rust.sh",
        "default_target": "rust/target/release/adaptive-rl-quant-rust",
    }


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


def _finalize_rust_metrics(
    metrics: BackendMetricDict,
    state: EpisodeState,
    decision: QuantizationDecision,
) -> BackendMetricDict:
    """Align Rust output with Python simulator metadata fields."""
    metrics.setdefault("swap_cost_ms", 0.0)
    metrics.setdefault("cache_miss_count", 0.0)
    metrics["variant_churn"] = float(decision.metadata.get("moe_variant_churn", 0.0))
    metrics.setdefault("simulator_engine", "rust_cli")
    metrics.setdefault("latency_source", "simulator")
    metrics.setdefault("throughput_source", "simulator")
    metrics.setdefault("memory_source", "simulator")
    metrics.setdefault("perplexity_source", "simulator")
    if "tokens_processed" not in metrics or "latency_ms_per_token" not in metrics:
        from adaptive_quant.backends.protocol import per_token_latency_fields

        metrics.update(per_token_latency_fields(state, float(metrics.get("latency_ms", 0.0))))
    return metrics


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
        raise RustCliError(f"Rust CLI binary not found. {_BUILD_HINT}")
    payload = json.dumps(_payload_for_eval(state, decision, config), separators=(",", ":"))
    try:
        completed = subprocess.run(
            [cli, "sim-eval"],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(config.rust_cli_timeout_s),
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
    return _finalize_rust_metrics(parsed, state, decision)


__all__ = [
    "RustCliError",
    "resolve_rust_cli_binary",
    "run_rust_sim_eval",
    "rust_cli_status",
    "rust_simulator_available",
]
