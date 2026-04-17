from __future__ import annotations

import math
import re

from adaptive_quant.math_utils import clamp, deterministic_float, mean, norm, variance
from adaptive_quant.types import InputFeatures, LayerSensitivity, PromptSample

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _token_ids(tokens: list[str]) -> list[float]:
    return [deterministic_float(token, 0.0, 1.0) for token in tokens]


def _embedding_vector(tokens: list[str], dimensions: int = 8) -> list[float]:
    if not tokens:
        return [0.0] * dimensions
    values = [0.0] * dimensions
    for token in tokens:
        for index in range(dimensions):
            values[index] += deterministic_float(f"{token}:{index}", -1.0, 1.0)
    return [value / len(tokens) for value in values]


def extract_input_features(prompt: PromptSample) -> InputFeatures:
    tokens = tokenize(prompt.text)
    if not tokens:
        return InputFeatures(0, 0.0, 0.0, 0.0, 0.0)

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    entropy = 0.0
    total = len(tokens)
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-9, 2)
    max_entropy = math.log(max(len(counts), 2), 2)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    token_ids = _token_ids(tokens)
    token_variance = clamp(variance(token_ids) * 12.0, 0.0, 1.0)

    embedding = _embedding_vector(tokens)
    embedding_norm = clamp(norm(embedding) / math.sqrt(len(embedding)), 0.0, 1.5)

    length_score = clamp(len(tokens) / 80.0, 0.0, 1.4)
    complexity = clamp(
        0.35 * min(length_score, 1.0)
        + 0.30 * normalized_entropy
        + 0.20 * token_variance
        + 0.15 * min(embedding_norm, 1.0),
        0.0,
        1.25,
    )
    return InputFeatures(
        prompt_length=len(tokens),
        token_entropy=normalized_entropy,
        token_variance=token_variance,
        embedding_norm=embedding_norm,
        complexity_score=complexity,
    )


def estimate_layer_sensitivity(prompt: PromptSample, input_features: InputFeatures, num_layers: int) -> LayerSensitivity:
    domain_bias = deterministic_float(f"domain:{prompt.domain}", -0.08, 0.08)
    attention = clamp(0.45 + 0.40 * input_features.token_entropy + 0.15 * input_features.complexity_score + domain_bias, 0.0, 1.4)
    ffn = clamp(0.42 + 0.38 * input_features.token_variance + 0.18 * input_features.embedding_norm + domain_bias, 0.0, 1.4)

    layer_stats: list[float] = []
    for layer_index in range(num_layers):
        phase = math.sin((layer_index + 1) * 0.8 + input_features.complexity_score * 2.2)
        layer_bias = deterministic_float(f"{prompt.prompt_id}:{layer_index}", -0.10, 0.10)
        layer_value = clamp(
            0.40
            + 0.25 * input_features.complexity_score
            + 0.15 * attention
            + 0.10 * ffn
            + 0.12 * phase
            + layer_bias,
            0.0,
            1.5,
        )
        layer_stats.append(layer_value)
    return LayerSensitivity(attention_sensitivity=attention, ffn_sensitivity=ffn, layer_stats=layer_stats)


def summarize_precision_needs(input_features: InputFeatures, sensitivity: LayerSensitivity) -> float:
    combined = [
        input_features.complexity_score,
        input_features.token_entropy,
        input_features.token_variance,
        sensitivity.attention_sensitivity,
        sensitivity.ffn_sensitivity,
        mean(sensitivity.layer_stats),
    ]
    return clamp(mean(combined), 0.0, 1.4)

