from __future__ import annotations

from dataclasses import replace

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.features import summarize_precision_needs
from adaptive_quant.math_utils import clamp, mean, variance
from adaptive_quant.types import EpisodeState, QuantMode, QuantizationDecision


def precision_level_to_bits(precision_level: float, config: FrameworkConfig) -> float:
    min_bits = min(config.discrete_bit_widths)
    max_bits = max(config.discrete_bit_widths)
    return min_bits + clamp(precision_level, *config.precision_bounds) * (max_bits - min_bits)


def safe_fallback_decision(config: FrameworkConfig) -> QuantizationDecision:
    return QuantizationDecision(
        mode=QuantMode.DISCRETE,
        base_bit_width=config.safe_default_bits,
        scale_factor=1.0,
        clipping_range=1.0,
        precision_level=0.5,
        metadata={"fallback_reason": "safe_default"},
    )


def _expand_group_bits(group_bits: list[int], num_layers: int) -> list[float]:
    if not group_bits:
        return []
    layers_per_group = max(1, num_layers // len(group_bits))
    expanded: list[float] = []
    for bit_width in group_bits:
        expanded.extend([float(bit_width)] * layers_per_group)
    while len(expanded) < num_layers:
        expanded.append(float(group_bits[-1]))
    return expanded[:num_layers]


def _normalize_bit_width(bit_width: int | None, config: FrameworkConfig) -> int:
    allowed = sorted(config.discrete_bit_widths)
    if bit_width is None:
        return config.safe_default_bits
    return min(allowed, key=lambda candidate: abs(candidate - bit_width))


def _dynamic_bits(base_bit_width: int, state: EpisodeState, config: FrameworkConfig) -> list[float]:
    min_bits = min(config.discrete_bit_widths)
    max_bits = max(config.discrete_bit_widths)
    complexity = state.input_features.complexity_score
    layer_bits: list[float] = []
    for layer_stat in state.sensitivity.layer_stats:
        adjustment = 2.2 * (complexity - 0.45) + 1.7 * (layer_stat - 0.55)
        layer_bits.append(clamp(base_bit_width + adjustment, min_bits, max_bits))
    return layer_bits


def _learned_bits(decision: QuantizationDecision, state: EpisodeState, config: FrameworkConfig) -> list[float]:
    min_bits = min(config.discrete_bit_widths)
    max_bits = max(config.discrete_bit_widths)
    learned_span = (max_bits - min_bits) * 0.75
    base_bits = min_bits + clamp(decision.precision_level, *config.precision_bounds) * learned_span
    scale_factor = clamp(decision.scale_factor, *config.scale_bounds)
    clipping_range = clamp(decision.clipping_range, *config.clip_bounds)
    precision_need = summarize_precision_needs(state.input_features, state.sensitivity)

    layer_bits: list[float] = []
    for layer_index, layer_stat in enumerate(state.sensitivity.layer_stats):
        sensitivity_push = 1.05 * (layer_stat - 0.55) + 0.80 * (precision_need - 0.50)
        scale_push = (scale_factor - 1.0) * 0.45
        clipping_push = (clipping_range - 1.0) * 0.35
        depth_bias = 0.12 if layer_index >= len(state.sensitivity.layer_stats) // 2 else -0.04
        layer_bits.append(
            clamp(
                base_bits + sensitivity_push + scale_push + clipping_push + depth_bias,
                min_bits,
                max_bits,
            )
        )
    return layer_bits


def finalize_decision(decision: QuantizationDecision, state: EpisodeState, config: FrameworkConfig) -> QuantizationDecision:
    finalized = replace(decision)
    finalized.scale_factor = clamp(finalized.scale_factor, *config.scale_bounds)
    finalized.clipping_range = clamp(finalized.clipping_range, *config.clip_bounds)
    finalized.precision_level = clamp(finalized.precision_level, *config.precision_bounds)

    if finalized.mode == QuantMode.DISCRETE:
        bit_width = _normalize_bit_width(finalized.base_bit_width, config)
        finalized.base_bit_width = bit_width
        finalized.effective_layer_bits = [float(bit_width)] * config.num_layers
    elif finalized.mode == QuantMode.GROUPED:
        normalized = [_normalize_bit_width(bit_width, config) for bit_width in finalized.group_bit_widths]
        finalized.group_bit_widths = normalized
        finalized.effective_layer_bits = _expand_group_bits(normalized, config.num_layers)
    elif finalized.mode == QuantMode.PER_LAYER:
        if not finalized.layer_bit_widths:
            finalized.layer_bit_widths = [config.safe_default_bits] * config.num_layers
        normalized = [_normalize_bit_width(bit_width, config) for bit_width in finalized.layer_bit_widths]
        while len(normalized) < config.num_layers:
            normalized.append(normalized[-1])
        finalized.layer_bit_widths = normalized[: config.num_layers]
        finalized.effective_layer_bits = [float(bit_width) for bit_width in finalized.layer_bit_widths]
    elif finalized.mode == QuantMode.DYNAMIC:
        bit_width = _normalize_bit_width(finalized.base_bit_width, config)
        finalized.base_bit_width = bit_width
        finalized.effective_layer_bits = _dynamic_bits(bit_width, state, config)
    elif finalized.mode == QuantMode.LEARNED:
        finalized.effective_layer_bits = _learned_bits(finalized, state, config)
    else:
        raise ValueError(f"Unsupported decision mode: {finalized.mode}")

    finalized.metadata = dict(finalized.metadata)
    finalized.metadata["average_bits"] = mean(finalized.effective_layer_bits)
    finalized.metadata["bit_variance"] = variance(finalized.effective_layer_bits)

    out_of_bounds = any(bit < min(config.discrete_bit_widths) or bit > max(config.discrete_bit_widths) for bit in finalized.effective_layer_bits)
    extremely_fragmented = variance(finalized.effective_layer_bits) > 4.0
    if out_of_bounds or extremely_fragmented:
        fallback = safe_fallback_decision(config)
        fallback = finalize_decision(fallback, state, config) if fallback.mode != QuantMode.DISCRETE or not fallback.effective_layer_bits else fallback
        fallback.fallback_applied = True
        fallback.unstable = True
        fallback.metadata["fallback_reason"] = "quantization_safety"
        return fallback

    return finalized


def quantize_values(values: list[float], decision: QuantizationDecision, config: FrameworkConfig) -> list[float]:
    if not values:
        return []
    scale = clamp(decision.scale_factor, *config.scale_bounds)
    clip = clamp(decision.clipping_range, *config.clip_bounds)
    average_bits = mean(decision.effective_layer_bits) if decision.effective_layer_bits else config.safe_default_bits
    levels = max(2, (2 ** int(round(average_bits))) - 1)

    quantized: list[float] = []
    for value in values:
        clipped = clamp(value, -clip, clip)
        normalized = clipped / scale
        rounded = round(normalized * levels) / levels
        quantized.append(rounded * scale)
    return quantized
