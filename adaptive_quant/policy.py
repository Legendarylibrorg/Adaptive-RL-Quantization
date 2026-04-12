"""Quantization **policy**: maps env state → ``QuantizationDecision`` (modes, bits, learned knobs, MoE indices)."""

from __future__ import annotations

import copy
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
        if trace.mode_trace is not None:
            self.mode_head.update(
                trace.state_vector,
                trace.mode_trace["selected_index"],
                trace.mode_trace["probabilities"],
                advantage,
                self.config.learning_rate,
            )

        for action_trace in trace.action_traces:
            head_name = action_trace["head"]
            if head_name == "discrete":
                self.discrete_head.update(
                    trace.state_vector,
                    action_trace["selected_index"],
                    action_trace["probabilities"],
                    advantage,
                    self.config.learning_rate,
                )
            elif head_name == "group":
                self.group_heads[action_trace["slot"]].update(
                    trace.state_vector,
                    action_trace["selected_index"],
                    action_trace["probabilities"],
                    advantage,
                    self.config.learning_rate,
                )
            elif head_name == "layer":
                self.layer_heads[action_trace["slot"]].update(
                    trace.state_vector,
                    action_trace["selected_index"],
                    action_trace["probabilities"],
                    advantage,
                    self.config.learning_rate,
                )
            elif head_name == "learned":
                self.learned_head.update(
                    trace.state_vector,
                    action_trace["raw_samples"],
                    action_trace["raw_means"],
                    advantage,
                    self.config.learning_rate,
                )
            elif head_name == "moe":
                self.moe_heads[action_trace["slot"]].update(
                    trace.state_vector,
                    action_trace["selected_index"],
                    action_trace["probabilities"],
                    advantage,
                    self.config.learning_rate,
                )
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


def _map_to_bounds(value: float, lower: float, upper: float) -> float:
    return lower + (upper - lower) * clamp(value, 0.0, 1.0)
