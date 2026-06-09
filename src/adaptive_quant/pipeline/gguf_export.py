"""Optional GGUF export via llama.cpp ``quantize`` after policy training."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.model_routes import QUANT_BITS

_BIT_WIDTH_TO_QUANT: dict[int, str] = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    5: "Q5_K_M",
    6: "Q6_K",
    8: "Q8_0",
}

_QUANTIZE_NAME_CANDIDATES = (
    "llama-quantize",
    "llama-quantize.exe",
    "quantize",
    "quantize.exe",
)


def resolve_gguf_export_source(config: FrameworkConfig) -> str:
    source = config.llama_cpp_gguf_export_source or config.llama_cpp_model
    if not source:
        raise ValueError("GGUF export requires llama_cpp_gguf_export_source or llama_cpp_model")
    return str(source)


def resolve_gguf_quant_type(
    config: FrameworkConfig,
    recommendation: dict[str, object] | None,
) -> str:
    """Pick llama.cpp quant label from config default or recommendation bit width."""
    default = str(config.llama_cpp_gguf_export_quant_type).strip().upper()
    if not isinstance(recommendation, dict):
        return default
    fixed = recommendation.get("recommended_quant")
    decision = recommendation.get("decision")
    deploy = decision.get("deploy") if isinstance(decision, dict) else None
    if deploy == "adaptive_policy":
        return default
    if isinstance(fixed, dict):
        signature = str(fixed.get("signature", ""))
        match = re.search(r"base=(\d+)", signature)
        if match:
            bits = int(match.group(1))
            return _BIT_WIDTH_TO_QUANT.get(bits, default)
    return default


def derive_quantize_binary(config: FrameworkConfig) -> str:
    if config.llama_cpp_gguf_quantize_binary:
        return str(config.llama_cpp_gguf_quantize_binary)
    binary = config.llama_cpp_binary
    if not binary:
        raise ValueError("GGUF export requires llama_cpp_gguf_quantize_binary or llama_cpp_binary")
    parent = Path(binary).resolve().parent
    for name in _QUANTIZE_NAME_CANDIDATES:
        candidate = parent / name
        if candidate.is_file():
            return str(candidate)
    return str(parent / "llama-quantize")


def export_gguf(
    config: FrameworkConfig,
    recommendation: dict[str, object] | None,
) -> dict[str, Any]:
    """Run llama.cpp quantize to produce an exported GGUF from a source model."""
    source_path = resolve_gguf_export_source(config)
    quant_type = resolve_gguf_quant_type(config, recommendation)
    if quant_type not in QUANT_BITS:
        raise ValueError(f"Unsupported GGUF quant type {quant_type!r}")

    output_path = config.gguf_export_path(quant_type=quant_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_binary = derive_quantize_binary(config)

    argv = [quantize_binary]
    if config.llama_cpp_gguf_export_allow_requantize:
        argv.append("--allow-requantize")
    argv.extend([source_path, output_path, quant_type])

    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=max(30.0, float(config.llama_cpp_timeout_s) * 4.0),
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"GGUF export failed (exit {completed.returncode}): {stderr[:500] or argv}"
        )

    return {
        "enabled": True,
        "output_path": output_path,
        "source_path": source_path,
        "quant_type": quant_type,
        "quantize_binary": quantize_binary,
        "command": argv,
    }


def maybe_export_gguf(
    config: FrameworkConfig,
    recommendation: dict[str, object] | None,
) -> dict[str, Any]:
    if not config.llama_cpp_gguf_export_enabled:
        return {"enabled": False, "skipped": True, "reason": "llama_cpp_gguf_export_enabled=false"}
    try:
        return export_gguf(config, recommendation)
    except Exception as exc:
        return {
            "enabled": True,
            "skipped": False,
            "error": str(exc),
            "quant_type": resolve_gguf_quant_type(config, recommendation),
        }


__all__ = [
    "derive_quantize_binary",
    "export_gguf",
    "maybe_export_gguf",
    "resolve_gguf_export_source",
    "resolve_gguf_quant_type",
]
