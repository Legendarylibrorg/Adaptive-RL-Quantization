"""Policy network heads and checkpoint (de)serialization helpers."""

from __future__ import annotations

import math
import random

from adaptive_quant.math_utils import (
    argmax,
    clamp,
    dot,
    gaussian_sample,
    sample_categorical,
    softmax,
    stable_sigmoid,
)


def _random_matrix(
    rows: int, cols: int, rng: random.Random, scale: float = 0.08
) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


class CategoricalHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random) -> None:
        self.weights = _random_matrix(output_dim, input_dim, rng)
        self.bias = [0.0] * output_dim

    def logits(self, state_vector: list[float]) -> list[float]:
        return [
            dot(weights, state_vector) + bias
            for weights, bias in zip(self.weights, self.bias, strict=True)
        ]

    def sample(
        self, state_vector: list[float], rng: random.Random, deterministic: bool = False
    ) -> tuple[int, list[float]]:
        probabilities = softmax(self.logits(state_vector))
        if deterministic:
            return argmax(probabilities), probabilities
        return sample_categorical(probabilities, rng), probabilities

    def update(
        self,
        state_vector: list[float],
        selected_index: int,
        probabilities: list[float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        for row_index, row in enumerate(self.weights):
            coefficient = (
                (1.0 if row_index == selected_index else 0.0) - probabilities[row_index]
            ) * advantage
            for column_index, value in enumerate(state_vector):
                row[column_index] += learning_rate * coefficient * value
            self.bias[row_index] += learning_rate * coefficient


class GaussianHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random, stddev: float) -> None:
        self.weights = _random_matrix(output_dim, input_dim, rng)
        self.bias = [0.0] * output_dim
        self.stddev = stddev

    def means(self, state_vector: list[float]) -> list[float]:
        return [
            dot(weights, state_vector) + bias
            for weights, bias in zip(self.weights, self.bias, strict=True)
        ]

    def sample(
        self,
        state_vector: list[float],
        rng: random.Random,
        bounds: list[tuple[float, float]],
        deterministic: bool = False,
    ) -> tuple[list[float], list[float], list[float]]:
        raw_means = self.means(state_vector)
        if deterministic:
            raw_samples = list(raw_means)
        else:
            raw_samples = [
                gaussian_sample(mean_value, self.stddev, rng) for mean_value in raw_means
            ]
        mapped = [
            _map_to_bounds(stable_sigmoid(sample), lower, upper)
            for sample, (lower, upper) in zip(raw_samples, bounds, strict=True)
        ]
        return mapped, raw_samples, raw_means

    def update(
        self,
        state_vector: list[float],
        raw_samples: list[float],
        raw_means: list[float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        variance = max(self.stddev * self.stddev, 1e-6)
        for row_index, row in enumerate(self.weights):
            coefficient = ((raw_samples[row_index] - raw_means[row_index]) / variance) * advantage
            for column_index, value in enumerate(state_vector):
                row[column_index] += learning_rate * coefficient * value
            self.bias[row_index] += learning_rate * coefficient


class ValueHead:
    def __init__(self, input_dim: int, rng: random.Random) -> None:
        self.weights = [rng.uniform(-0.05, 0.05) for _ in range(input_dim)]
        self.bias = 0.0

    def predict(self, state_vector: list[float]) -> float:
        return dot(self.weights, state_vector) + self.bias

    def update(self, state_vector: list[float], target: float, learning_rate: float) -> None:
        prediction = self.predict(state_vector)
        error = target - prediction
        for index, value in enumerate(state_vector):
            self.weights[index] += learning_rate * error * value
        self.bias += learning_rate * error
def _map_to_bounds(value: float, lower: float, upper: float) -> float:
    return lower + (upper - lower) * clamp(value, 0.0, 1.0)


def _serialize_rng_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_serialize_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_serialize_rng_state(item) for item in value]
    return value


def _deserialize_rng_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_deserialize_rng_state(item) for item in value)
    return value


def _serialize_categorical_head(head: CategoricalHead) -> dict[str, object]:
    return {
        "weights": [list(row) for row in head.weights],
        "bias": list(head.bias),
    }


def _categorical_head_shape(payload: object, *, label: str) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    weights = payload.get("weights")
    bias = payload.get("bias")
    if not isinstance(weights, list) or not isinstance(bias, list):
        raise TypeError(f"{label} payload must contain list weights and bias")
    if len(weights) != len(bias):
        raise ValueError(f"{label} payload has {len(weights)} rows but {len(bias)} bias values")
    row_width: int | None = None
    for index, row in enumerate(weights):
        if not isinstance(row, list):
            raise TypeError(f"{label} row {index} must be a list")
        if row_width is None:
            row_width = len(row)
        elif len(row) != row_width:
            raise ValueError(f"{label} rows must all have the same width")
    return len(weights), (row_width or 0)


def _validate_categorical_head_payload(
    label: str,
    payload: object,
    *,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    output_dim, input_dim = _categorical_head_shape(payload, label=label)
    if input_dim != expected_input_dim or output_dim != expected_output_dim:
        raise ValueError(
            f"{label} checkpoint shape mismatch: expected "
            f"{expected_output_dim}x{expected_input_dim}, got {output_dim}x{input_dim}"
        )


def _validate_head_payload_sequence(
    label: str,
    payload: object,
    *,
    expected_count: int,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    if not isinstance(payload, list):
        raise TypeError(f"{label} payload must be a list")
    if len(payload) != expected_count:
        raise ValueError(
            f"{label} checkpoint count mismatch: expected {expected_count}, got {len(payload)}"
        )
    for index, item in enumerate(payload):
        _validate_categorical_head_payload(
            f"{label}[{index}]",
            item,
            expected_input_dim=expected_input_dim,
            expected_output_dim=expected_output_dim,
        )


def _finite_float(value: object, *, label: str) -> float:
    """Coerce a checkpoint-encoded number to ``float`` while rejecting NaN/Inf.

    Loading a JSON-encoded ``"Infinity"`` / ``"NaN"`` would otherwise inject
    poison values into policy weights and silently break training downstream;
    untrusted checkpoints can therefore use this to wedge a session.
    """
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value).__name__}")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{label} must be finite, got {f!r}")
    return f


def _restore_categorical_head(head: CategoricalHead, payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("categorical head payload must be a dict")
    head.weights = [
        [_finite_float(value, label=f"weights[{i}][{j}]") for j, value in enumerate(row)]
        for i, row in enumerate(payload["weights"])
    ]
    head.bias = [
        _finite_float(value, label=f"bias[{i}]") for i, value in enumerate(payload["bias"])
    ]


def _categorical_head_from_payload(payload: object) -> CategoricalHead:
    if not isinstance(payload, dict):
        raise TypeError("categorical head payload must be a dict")
    input_dim = len(payload["weights"][0]) if payload["weights"] else 0
    output_dim = len(payload["weights"])
    head = CategoricalHead(input_dim, output_dim, random.Random(0))
    _restore_categorical_head(head, payload)
    return head


def _serialize_gaussian_head(head: GaussianHead) -> dict[str, object]:
    return {
        "weights": [list(row) for row in head.weights],
        "bias": list(head.bias),
        "stddev": float(head.stddev),
    }


def _validate_gaussian_head_payload(
    label: str,
    payload: object,
    *,
    expected_input_dim: int,
    expected_output_dim: int,
) -> None:
    output_dim, input_dim = _categorical_head_shape(payload, label=label)
    if input_dim != expected_input_dim or output_dim != expected_output_dim:
        raise ValueError(
            f"{label} checkpoint shape mismatch: expected "
            f"{expected_output_dim}x{expected_input_dim}, got {output_dim}x{input_dim}"
        )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    stddev = payload.get("stddev")
    if not isinstance(stddev, (int, float)) or isinstance(stddev, bool):
        raise TypeError(f"{label} stddev must be numeric")


def _gaussian_head_from_payload(payload: object) -> GaussianHead:
    if not isinstance(payload, dict):
        raise TypeError("gaussian head payload must be a dict")
    input_dim = len(payload["weights"][0]) if payload["weights"] else 0
    output_dim = len(payload["weights"])
    stddev = _finite_float(payload["stddev"], label="gaussian.stddev")
    if stddev < 0.0:
        raise ValueError(f"gaussian.stddev must be >= 0, got {stddev!r}")
    head = GaussianHead(input_dim, output_dim, random.Random(0), stddev)
    head.weights = [
        [_finite_float(value, label=f"gaussian.weights[{i}][{j}]") for j, value in enumerate(row)]
        for i, row in enumerate(payload["weights"])
    ]
    head.bias = [
        _finite_float(value, label=f"gaussian.bias[{i}]") for i, value in enumerate(payload["bias"])
    ]
    return head


def _serialize_value_head(head: ValueHead) -> dict[str, object]:
    return {
        "weights": list(head.weights),
        "bias": float(head.bias),
    }


def _validate_value_head_payload(label: str, payload: object, *, expected_input_dim: int) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict")
    weights = payload.get("weights")
    bias = payload.get("bias")
    if not isinstance(weights, list):
        raise TypeError(f"{label} weights must be a list")
    if len(weights) != expected_input_dim:
        raise ValueError(
            f"{label} checkpoint width mismatch: expected {expected_input_dim}, got {len(weights)}"
        )
    if not isinstance(bias, (int, float)) or isinstance(bias, bool):
        raise TypeError(f"{label} bias must be numeric")


def _value_head_from_payload(payload: object) -> ValueHead:
    if not isinstance(payload, dict):
        raise TypeError("value head payload must be a dict")
    head = ValueHead(len(payload["weights"]), random.Random(0))
    head.weights = [
        _finite_float(value, label=f"value.weights[{i}]")
        for i, value in enumerate(payload["weights"])
    ]
    head.bias = _finite_float(payload["bias"], label="value.bias")
    return head
