"""Stable import path ``adaptive_quant.backend`` (implementation in ``adaptive_quant.backends``)."""

from __future__ import annotations

import subprocess

from adaptive_quant.backends import (
    Backend,
    ExternalQualityScores,
    LlamaCppBackend,
    SimulatorBackend,
    apply_external_quality,
    build_backend,
    parse_llama_cpp_metrics,
    per_token_latency_fields,
    register_backend,
    require_llama_cpp_paths,
    run_llama_cpp_measurement,
)
from adaptive_quant.backends import llama_cpp as _llama_cpp

_extract_numeric = _llama_cpp._extract_numeric

__all__ = [
    "Backend",
    "ExternalQualityScores",
    "LlamaCppBackend",
    "SimulatorBackend",
    "_extract_numeric",
    "apply_external_quality",
    "build_backend",
    "parse_llama_cpp_metrics",
    "per_token_latency_fields",
    "register_backend",
    "require_llama_cpp_paths",
    "run_llama_cpp_measurement",
    "subprocess",
]
