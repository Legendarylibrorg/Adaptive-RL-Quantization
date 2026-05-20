"""Measurement backends (simulator, llama.cpp) and registration hooks."""

from __future__ import annotations

from adaptive_quant.backends.llama_cpp import (
    LlamaCppBackend,
    extract_numeric,
    parse_llama_cpp_metrics,
    require_llama_cpp_paths,
    run_llama_cpp_measurement,
)
from adaptive_quant.backends.protocol import Backend, per_token_latency_fields
from adaptive_quant.backends.quality import ExternalQualityScores, apply_external_quality
from adaptive_quant.backends.registry import build_backend, register_backend
from adaptive_quant.backends.simulator import SimulatorBackend

__all__ = [
    "Backend",
    "ExternalQualityScores",
    "LlamaCppBackend",
    "SimulatorBackend",
    "apply_external_quality",
    "build_backend",
    "extract_numeric",
    "parse_llama_cpp_metrics",
    "per_token_latency_fields",
    "register_backend",
    "require_llama_cpp_paths",
    "run_llama_cpp_measurement",
]
