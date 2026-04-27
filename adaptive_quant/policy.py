"""Quantization **policy**: maps env state → ``QuantizationDecision`` (modes, bits, learned knobs, MoE indices)."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import (
    argmax,
    clamp,
    dot,
    gaussian_sample,
    sample_categorical,
    softmax,
    stable_sigmoid,
)
from adaptive_quant.types import EpisodeState, QuantizationDecision, QuantMode


def _random_matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.08) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


@dataclass
class PolicyTrace:
    state_vector: list[float]
    value_prediction: float
    mode_trace: dict | None
    action_traces: list[dict]


class CategoricalHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random) -> None:
        self.weights = _random_matrix(output_dim, input_dim, rng)
        self.bias = [0.0] * output_dim

    def logits(self, state_vector: list[float]) -> list[float]:
        return [dot(weights, state_vector) + bias for weights, bias in zip(self.weights, self.bias)]

    def sample(self, state_vector: list[float], rng: random.Random, deterministic: bool = False) -> tuple[int, list[float]]:
        probabilities = softmax(self.logits(state_vector))
        if deterministic:
            return argmax(probabilities), probabilities
        return sample_categorical(probabilities, rng), probabilities

    def update(self, state_vector: list[float], selected_index: int, probabilities: list[float], advantage: float, learning_rate: float) -> None:
        for row_index, row in enumerate(self.weights):
            coefficient = ((1.0 if row_index == selected_index else 0.0) - probabilities[row_index]) * advantage
            for column_index, value in enumerate(state_vector):
                row[column_index] += learning_rate * coefficient * value
            self.bias[row_index] += learning_rate * coefficient


class GaussianHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random, stddev: float) -> None:
        self.weights = _random_matrix(output_dim, input_dim, rng)
        self.bias = [0.0] * output_dim
        self.stddev = stddev

    def means(self, state_vector: list[float]) -> list[float]:
        return [dot(weights, state_vector) + bias for weights, bias in zip(self.weights, self.bias)]

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
            raw_samples = [gaussian_sample(mean_value, self.stddev, rng) for mean_value in raw_means]
        mapped = [_map_to_bounds(stable_sigmoid(sample), lower, upper) for sample, (lower, upper) in zip(raw_samples, bounds)]
        return mapped, raw_samples, raw_means

    def update(self, state_vector: list[float], raw_samples: list[float], raw_means: list[float], advantage: float, learning_rate: float) -> None:
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


class UniversalQuantizationPolicy:
    """Bandit-friendly policy: state vector → ``QuantizationDecision`` across configured quant modes (incl. learned + MoE)."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed + 101)
        self.supported_modes = config.supported_modes()
        self.ordered_hardware = config.ordered_hardware()
        self.state_dim = config.state_vector_dim()
        self.mode_head = CategoricalHead(self.state_dim, len(self.supported_modes), self.rng)
        self.discrete_head = CategoricalHead(self.state_dim, len(config.discrete_bit_widths), self.rng)
        self.group_heads = [CategoricalHead(self.state_dim, len(config.discrete_bit_widths), self.rng) for _ in range(config.num_groups)]
        self.layer_heads = [CategoricalHead(self.state_dim, len(config.discrete_bit_widths), self.rng) for _ in range(config.num_layers)]
        self.learned_head = GaussianHead(self.state_dim, 3, self.rng, config.continuous_stddev)
        self.learned_head.bias = [-0.10, -0.20, -0.45]
        self.moe_heads = [CategoricalHead(self.state_dim, config.moe_variant_count(), self.rng) for _ in range(config.moe_top_k)] if config.moe_enabled else []
        self.value_head = ValueHead(self.state_dim, self.rng)

    def act(self, state: EpisodeState, deterministic: bool = False) -> tuple[QuantizationDecision, PolicyTrace]:
        state_vector = state.to_vector(self.ordered_hardware)
        value_prediction = self.value_head.predict(state_vector)
        quant_mode = self.config.resolved_quant_mode()

        mode_trace = None
        selected_mode = quant_mode
        if quant_mode == QuantMode.HYBRID:
            selected_index, probabilities = self.mode_head.sample(state_vector, self.rng, deterministic=deterministic)
            selected_mode = self.supported_modes[selected_index]
            mode_trace = {
                "selected_index": selected_index,
                "probabilities": probabilities,
            }

        traces: list[dict] = []
        if selected_mode in {QuantMode.DISCRETE, QuantMode.DYNAMIC}:
            bit_index, probabilities = self.discrete_head.sample(state_vector, self.rng, deterministic=deterministic)
            bit_width = self.config.discrete_bit_widths[bit_index]
            decision = QuantizationDecision(
                mode=selected_mode,
                base_bit_width=bit_width,
                scale_factor=1.0,
                clipping_range=1.0,
                precision_level=(bit_width - min(self.config.discrete_bit_widths))
                / (max(self.config.discrete_bit_widths) - min(self.config.discrete_bit_widths)),
                metadata={"head": "discrete"},
            )
            traces.append({"head": "discrete", "selected_index": bit_index, "probabilities": probabilities})
        elif selected_mode == QuantMode.GROUPED:
            group_bits: list[int] = []
            for group_index, head in enumerate(self.group_heads):
                bit_index, probabilities = head.sample(state_vector, self.rng, deterministic=deterministic)
                group_bits.append(self.config.discrete_bit_widths[bit_index])
                traces.append({"head": "group", "slot": group_index, "selected_index": bit_index, "probabilities": probabilities})
            decision = QuantizationDecision(mode=selected_mode, group_bit_widths=group_bits, metadata={"head": "grouped"})
        elif selected_mode == QuantMode.PER_LAYER:
            layer_bits: list[int] = []
            for layer_index, head in enumerate(self.layer_heads):
                bit_index, probabilities = head.sample(state_vector, self.rng, deterministic=deterministic)
                layer_bits.append(self.config.discrete_bit_widths[bit_index])
                traces.append({"head": "layer", "slot": layer_index, "selected_index": bit_index, "probabilities": probabilities})
            decision = QuantizationDecision(mode=selected_mode, layer_bit_widths=layer_bits, metadata={"head": "per_layer"})
        elif selected_mode == QuantMode.LEARNED:
            bounds = [
                self.config.scale_bounds,
                self.config.clip_bounds,
                self.config.precision_bounds,
            ]
            samples, raw_samples, raw_means = self.learned_head.sample(state_vector, self.rng, bounds, deterministic=deterministic)
            decision = QuantizationDecision(
                mode=selected_mode,
                scale_factor=samples[0],
                clipping_range=samples[1],
                precision_level=samples[2],
                metadata={"head": "learned"},
            )
            traces.append({"head": "learned", "raw_samples": raw_samples, "raw_means": raw_means})
        else:
            raise ValueError(f"Unsupported selected mode: {selected_mode}")

        if self.config.moe_enabled and state.moe_context is not None:
            variant_indices: list[int] = []
            variant_names: list[str] = []
            for slot, head in enumerate(self.moe_heads[: len(state.moe_context.experts)]):
                variant_index, probabilities = head.sample(state_vector, self.rng, deterministic=deterministic)
                variant_indices.append(variant_index)
                variant_names.append(self.config.moe_variant_names[variant_index])
                traces.append({"head": "moe", "slot": slot, "selected_index": variant_index, "probabilities": probabilities})
            decision.moe_variant_indices = variant_indices
            decision.moe_variant_names = variant_names
            decision.metadata["moe_head"] = "packed_expert_bank"

        decision.metadata["selected_mode"] = selected_mode.value
        trace = PolicyTrace(
            state_vector=state_vector,
            value_prediction=value_prediction,
            mode_trace=mode_trace,
            action_traces=traces,
        )
        return decision, trace

    def update(self, trace: PolicyTrace, reward: float) -> None:
        advantage = reward - trace.value_prediction
        learning_rate = self.config.learning_rate

        def update_categorical(head: CategoricalHead, action_trace: dict) -> None:
            head.update(
                trace.state_vector,
                action_trace["selected_index"],
                action_trace["probabilities"],
                advantage,
                learning_rate,
            )

        if trace.mode_trace is not None:
            self.mode_head.update(
                trace.state_vector,
                trace.mode_trace["selected_index"],
                trace.mode_trace["probabilities"],
                advantage,
                learning_rate,
            )

        for action_trace in trace.action_traces:
            head_name = action_trace["head"]
            if head_name == "discrete":
                update_categorical(self.discrete_head, action_trace)
            elif head_name == "group":
                update_categorical(self.group_heads[action_trace["slot"]], action_trace)
            elif head_name == "layer":
                update_categorical(self.layer_heads[action_trace["slot"]], action_trace)
            elif head_name == "learned":
                self.learned_head.update(
                    trace.state_vector,
                    action_trace["raw_samples"],
                    action_trace["raw_means"],
                    advantage,
                    learning_rate,
                )
            elif head_name == "moe":
                update_categorical(self.moe_heads[action_trace["slot"]], action_trace)
        self.value_head.update(trace.state_vector, reward, self.config.value_learning_rate)

    def snapshot(self) -> dict[str, object]:
        return {
            "rng_state": self.rng.getstate(),
            "mode_head": copy.deepcopy(self.mode_head),
            "discrete_head": copy.deepcopy(self.discrete_head),
            "group_heads": copy.deepcopy(self.group_heads),
            "layer_heads": copy.deepcopy(self.layer_heads),
            "learned_head": copy.deepcopy(self.learned_head),
            "moe_heads": copy.deepcopy(self.moe_heads),
            "value_head": copy.deepcopy(self.value_head),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.rng.setstate(snapshot["rng_state"])
        self.mode_head = copy.deepcopy(snapshot["mode_head"])
        self.discrete_head = copy.deepcopy(snapshot["discrete_head"])
        self.group_heads = copy.deepcopy(snapshot["group_heads"])
        self.layer_heads = copy.deepcopy(snapshot["layer_heads"])
        self.learned_head = copy.deepcopy(snapshot["learned_head"])
        self.moe_heads = copy.deepcopy(snapshot.get("moe_heads", self.moe_heads))
        self.value_head = copy.deepcopy(snapshot["value_head"])

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "rng_state": _serialize_rng_state(self.rng.getstate()),
            "mode_head": _serialize_categorical_head(self.mode_head),
            "discrete_head": _serialize_categorical_head(self.discrete_head),
            "group_heads": [_serialize_categorical_head(head) for head in self.group_heads],
            "layer_heads": [_serialize_categorical_head(head) for head in self.layer_heads],
            "learned_head": _serialize_gaussian_head(self.learned_head),
            "moe_heads": [_serialize_categorical_head(head) for head in self.moe_heads],
            "value_head": _serialize_value_head(self.value_head),
        }

    def restore_checkpoint_state(self, payload: dict[str, object]) -> None:
        _validate_categorical_head_payload(
            "mode_head",
            payload["mode_head"],
            expected_input_dim=self.state_dim,
            expected_output_dim=len(self.supported_modes),
        )
        _validate_categorical_head_payload(
            "discrete_head",
            payload["discrete_head"],
            expected_input_dim=self.state_dim,
            expected_output_dim=len(self.config.discrete_bit_widths),
        )
        _validate_head_payload_sequence(
            "group_heads",
            payload["group_heads"],
            expected_count=len(self.group_heads),
            expected_input_dim=self.state_dim,
            expected_output_dim=len(self.config.discrete_bit_widths),
        )
        _validate_head_payload_sequence(
            "layer_heads",
            payload["layer_heads"],
            expected_count=len(self.layer_heads),
            expected_input_dim=self.state_dim,
            expected_output_dim=len(self.config.discrete_bit_widths),
        )
        _validate_gaussian_head_payload(
            "learned_head",
            payload["learned_head"],
            expected_input_dim=self.state_dim,
            expected_output_dim=3,
        )
        _validate_head_payload_sequence(
            "moe_heads",
            payload.get("moe_heads", []),
            expected_count=len(self.moe_heads),
            expected_input_dim=self.state_dim,
            expected_output_dim=self.config.moe_variant_count(),
        )
        _validate_value_head_payload(
            "value_head",
            payload["value_head"],
            expected_input_dim=self.state_dim,
        )
        self.rng.setstate(_deserialize_rng_state(payload["rng_state"]))
        _restore_categorical_head(self.mode_head, payload["mode_head"])
        _restore_categorical_head(self.discrete_head, payload["discrete_head"])
        self.group_heads = [_categorical_head_from_payload(item) for item in payload["group_heads"]]
        self.layer_heads = [_categorical_head_from_payload(item) for item in payload["layer_heads"]]
        self.learned_head = _gaussian_head_from_payload(payload["learned_head"])
        self.moe_heads = [_categorical_head_from_payload(item) for item in payload.get("moe_heads", [])]
        self.value_head = _value_head_from_payload(payload["value_head"])


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
    head.bias = [_finite_float(value, label=f"bias[{i}]") for i, value in enumerate(payload["bias"])]


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
        _finite_float(value, label=f"gaussian.bias[{i}]")
        for i, value in enumerate(payload["bias"])
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
