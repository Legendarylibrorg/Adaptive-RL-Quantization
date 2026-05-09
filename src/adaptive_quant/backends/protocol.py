from __future__ import annotations

from typing import Protocol

from adaptive_quant.types import BackendMetricDict, EpisodeState, QuantizationDecision


class Backend(Protocol):
    """Evaluation backend interface (simulator or llama.cpp)."""

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict: ...


def per_token_latency_fields(state: EpisodeState, latency_ms: float) -> dict[str, float]:
    """Normalize wall-clock latency by prompt length for logging and optional reward."""
    tokens = float(max(1, state.input_features.prompt_length))
    return {
        "tokens_processed": tokens,
        "latency_ms_per_token": float(latency_ms) / tokens,
    }
