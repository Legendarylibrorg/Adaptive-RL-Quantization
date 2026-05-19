from __future__ import annotations

from dataclasses import replace

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.features import summarize_precision_needs
from adaptive_quant.math_utils import clamp, mean, variance
from adaptive_quant.types import EpisodeState, QuantizationDecision, QuantMode


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
    return _pad_or_truncate(expanded, num_layers, fill=float(group_bits[-1]))


def _pad_or_truncate(values: list, length: int, *, fill) -> list:
    if length <= 0:
        return []
    if len(values) >= length:
        return values[:length]
    return values + [fill] * (length - len(values))


def _nearest_allowed_bit_width(
    bit_width: int | None,
    allowed: list[int],
    *,
    default: int,
) -> int:
    if bit_width is None:
        return default
    return min(allowed, key=lambda candidate: abs(candidate - bit_width))


def _allowed_bit_widths(config: FrameworkConfig) -> list[int]:
    return sorted(config.discrete_bit_widths)


def nearest_allowed_discrete_bit_width(value: float | int, config: FrameworkConfig) -> int:
    allowed = _allowed_bit_widths(config)
    if not allowed:
        return int(config.safe_default_bits)
    return min(allowed, key=lambda candidate: abs(float(candidate) - float(value)))


def _normalize_bit_width(bit_width: int | None, allowed: list[int], *, default: int) -> int:
    if bit_width is None:
        return default
    return _nearest_allowed_bit_width(bit_width, allowed, default=default)


def _dynamic_bits(
    base_bit_width: int,
    state: EpisodeState,
    *,
    min_bits: int,
    max_bits: int,
) -> list[float]:
    complexity = state.input_features.complexity_score
    layer_bits: list[float] = []
    for layer_stat in state.sensitivity.layer_stats:
        adjustment = 2.2 * (complexity - 0.45) + 1.7 * (layer_stat - 0.55)
        layer_bits.append(clamp(base_bit_width + adjustment, min_bits, max_bits))
    return layer_bits


def _learned_bits(
    decision: QuantizationDecision,
    state: EpisodeState,
    config: FrameworkConfig,
    *,
    min_bits: int,
    max_bits: int,
) -> list[float]:
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


def finalize_decision(
    decision: QuantizationDecision, state: EpisodeState, config: FrameworkConfig
) -> QuantizationDecision:
    finalized = replace(decision)
    finalized.scale_factor = clamp(finalized.scale_factor, *config.scale_bounds)
    finalized.clipping_range = clamp(finalized.clipping_range, *config.clip_bounds)
    finalized.precision_level = clamp(finalized.precision_level, *config.precision_bounds)

    allowed = _allowed_bit_widths(config)
    min_bits = allowed[0] if allowed else int(config.safe_default_bits)
    max_bits = allowed[-1] if allowed else int(config.safe_default_bits)

    if finalized.mode == QuantMode.DISCRETE:
        bit_width = _normalize_bit_width(
            finalized.base_bit_width, allowed, default=config.safe_default_bits
        )
        finalized.base_bit_width = bit_width
        finalized.effective_layer_bits = [float(bit_width)] * config.num_layers
    elif finalized.mode == QuantMode.GROUPED:
        normalized = [
            _normalize_bit_width(bit_width, allowed, default=config.safe_default_bits)
            for bit_width in finalized.group_bit_widths
        ]
        finalized.group_bit_widths = normalized
        finalized.effective_layer_bits = _expand_group_bits(normalized, config.num_layers)
    elif finalized.mode == QuantMode.PER_LAYER:
        if not finalized.layer_bit_widths:
            finalized.layer_bit_widths = [config.safe_default_bits] * config.num_layers
        normalized = [
            _normalize_bit_width(bit_width, allowed, default=config.safe_default_bits)
            for bit_width in finalized.layer_bit_widths
        ]
        normalized = _pad_or_truncate(normalized, config.num_layers, fill=normalized[-1])
        finalized.layer_bit_widths = normalized
        finalized.effective_layer_bits = [
            float(bit_width) for bit_width in finalized.layer_bit_widths
        ]
    elif finalized.mode == QuantMode.DYNAMIC:
        bit_width = _normalize_bit_width(
            finalized.base_bit_width, allowed, default=config.safe_default_bits
        )
        finalized.base_bit_width = bit_width
        finalized.effective_layer_bits = _dynamic_bits(
            bit_width, state, min_bits=min_bits, max_bits=max_bits
        )
    elif finalized.mode == QuantMode.LEARNED:
        finalized.effective_layer_bits = _learned_bits(
            finalized, state, config, min_bits=min_bits, max_bits=max_bits
        )
    else:
        raise ValueError(f"Unsupported decision mode: {finalized.mode}")

    finalized.metadata = dict(finalized.metadata)
    finalized.metadata["average_bits"] = mean(finalized.effective_layer_bits)
    finalized.metadata["bit_variance"] = variance(finalized.effective_layer_bits)
    _finalize_moe_selection(finalized, state, config)

    out_of_bounds = any(bit < min_bits or bit > max_bits for bit in finalized.effective_layer_bits)
    extremely_fragmented = variance(finalized.effective_layer_bits) > 4.0
    if out_of_bounds or extremely_fragmented:
        fallback = safe_fallback_decision(config)
        fallback = (
            finalize_decision(fallback, state, config)
            if fallback.mode != QuantMode.DISCRETE or not fallback.effective_layer_bits
            else fallback
        )
        fallback.fallback_applied = True
        fallback.unstable = True
        fallback.metadata["fallback_reason"] = "quantization_safety"
        return fallback

    return finalized


def _finalize_moe_selection(
    decision: QuantizationDecision, state: EpisodeState, config: FrameworkConfig
) -> None:
    if not config.moe_enabled or state.moe_context is None:
        decision.moe_variant_indices = []
        decision.moe_variant_names = []
        return

    default_index = config.default_moe_variant_index()
    variant_count = config.moe_variant_count()
    fixed_index = (
        config.moe_variant_index(config.moe_fixed_variant) if config.moe_fixed_variant else None
    )
    if fixed_index is not None:
        normalized_indices = [fixed_index] * len(state.moe_context.experts)
    else:
        normalized_indices = list(decision.moe_variant_indices[: len(state.moe_context.experts)])
        normalized_indices = _pad_or_truncate(
            normalized_indices,
            len(state.moe_context.experts),
            fill=default_index,
        )

    names: list[str] = []
    for slot, expert in enumerate(state.moe_context.experts):
        index = max(0, min(variant_count - 1, int(normalized_indices[slot])))
        if expert.available_variants_mask:
            available = [mask > 0.0 for mask in expert.available_variants_mask]
            if index >= len(available) or not available[index]:
                valid_indices = [
                    candidate for candidate, allowed in enumerate(available) if allowed
                ]
                index = (
                    default_index
                    if default_index in valid_indices
                    else (valid_indices[0] if valid_indices else default_index)
                )
        normalized_indices[slot] = index
        names.append(config.moe_variant_names[index])

    _apply_moe_safety(normalized_indices, state, config, default_index)
    names = [config.moe_variant_names[index] for index in normalized_indices]

    decision.moe_variant_indices = normalized_indices
    decision.moe_variant_names = names
    average_aggressiveness = (
        mean([index / max(1, variant_count - 1) for index in normalized_indices])
        if normalized_indices
        else 0.0
    )
    variant_churn = (
        mean(
            [abs(index - default_index) / max(1, variant_count - 1) for index in normalized_indices]
        )
        if normalized_indices
        else 0.0
    )
    decision.metadata["moe_enabled"] = True
    decision.metadata["moe_selected_variants"] = names
    decision.metadata["moe_average_aggressiveness"] = average_aggressiveness
    decision.metadata["moe_variant_churn"] = variant_churn
    decision.metadata["moe_fixed_variant"] = config.moe_fixed_variant


def _apply_moe_safety(
    indices: list[int], state: EpisodeState, config: FrameworkConfig, default_index: int
) -> None:
    if not indices or state.moe_context is None:
        return

    aggressive_index = config.aggressive_moe_variant_index()
    if (
        aggressive_index is not None
        and config.moe_max_aggressive_experts >= 0
        and aggressive_index < config.moe_variant_count()
    ):
        aggressive_slots = [slot for slot, index in enumerate(indices) if index == aggressive_index]
        if len(aggressive_slots) > config.moe_max_aggressive_experts:
            ranked = sorted(
                aggressive_slots,
                key=lambda slot: state.moe_context.experts[slot].router_probability,
                reverse=True,
            )
            keep = set(ranked[: config.moe_max_aggressive_experts])
            for slot in aggressive_slots:
                if slot not in keep:
                    indices[slot] = default_index

    while _predicted_moe_swap_cost(indices, state, config) > config.moe_max_swap_cost_ms:
        adjusted = False
        for slot, expert in sorted(
            enumerate(state.moe_context.experts),
            key=lambda item: (item[1].resident_on_device, item[1].router_probability),
        ):
            if indices[slot] > default_index and expert.resident_on_device < 0.5:
                indices[slot] = default_index
                adjusted = True
                break
        if adjusted:
            continue
        for slot, expert in sorted(
            enumerate(state.moe_context.experts),
            key=lambda item: item[1].router_probability,
        ):
            if indices[slot] != 0 and expert.resident_on_device < 0.5:
                indices[slot] = 0
                adjusted = True
                break
        if not adjusted:
            break


def _predicted_moe_swap_cost(
    indices: list[int], state: EpisodeState, config: FrameworkConfig
) -> float:
    if state.moe_context is None or not indices:
        return 0.0
    total = 0.0
    for expert, index in zip(state.moe_context.experts, indices, strict=True):
        aggressiveness = index / max(1, config.moe_variant_count() - 1)
        if expert.resident_on_device < 0.5:
            total += (
                (1.2 + 3.4 * aggressiveness)
                * (0.75 + expert.router_probability)
                * (1.10 - 0.35 * expert.hotness)
            )
    return total
