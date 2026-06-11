"""Quantization **policy**: maps env state → ``QuantizationDecision`` (modes, bits, learned knobs, MoE indices)."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import discrete_precision_level
from adaptive_quant.policy_heads import (
    CategoricalHead,
    GaussianHead,
    ValueHead,
    _categorical_head_from_payload,
    _deserialize_rng_state,
    _gaussian_head_from_payload,
    _restore_categorical_head,
    _serialize_categorical_head,
    _serialize_gaussian_head,
    _serialize_rng_state,
    _serialize_value_head,
    _validate_categorical_head_payload,
    _validate_gaussian_head_payload,
    _validate_head_payload_sequence,
    _validate_value_head_payload,
    _value_head_from_payload,
)
from adaptive_quant.types import EpisodeState, QuantizationDecision, QuantMode

__all__ = [
    "CategoricalHead",
    "GaussianHead",
    "PolicyTrace",
    "UniversalQuantizationPolicy",
    "ValueHead",
    "_categorical_head_from_payload",
    "_gaussian_head_from_payload",
    "_value_head_from_payload",
]


@dataclass
class PolicyTrace:
    state_vector: list[float]
    value_prediction: float
    mode_trace: dict | None
    action_traces: list[dict]


class UniversalQuantizationPolicy:
    """Bandit-friendly policy: state vector → ``QuantizationDecision`` across configured quant modes (incl. learned + MoE)."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed + 101)
        self.supported_modes = config.supported_modes()
        self.ordered_hardware = config.ordered_hardware()
        self.state_dim = config.state_vector_dim()
        self.mode_head = CategoricalHead(self.state_dim, len(self.supported_modes), self.rng)
        self.discrete_head = CategoricalHead(
            self.state_dim, len(config.discrete_bit_widths), self.rng
        )
        self.group_heads = [
            CategoricalHead(self.state_dim, len(config.discrete_bit_widths), self.rng)
            for _ in range(config.num_groups)
        ]
        self.layer_heads = [
            CategoricalHead(self.state_dim, len(config.discrete_bit_widths), self.rng)
            for _ in range(config.num_layers)
        ]
        self.learned_head = GaussianHead(self.state_dim, 3, self.rng, config.continuous_stddev)
        self.learned_head.bias = [-0.10, -0.20, -0.45]
        self.moe_heads = (
            [
                CategoricalHead(self.state_dim, config.moe_variant_count(), self.rng)
                for _ in range(config.moe_top_k)
            ]
            if config.moe_enabled
            else []
        )
        self.value_head = ValueHead(self.state_dim, self.rng)

    def act(
        self, state: EpisodeState, deterministic: bool = False
    ) -> tuple[QuantizationDecision, PolicyTrace]:
        state_vector = state.to_vector(self.ordered_hardware)
        value_prediction = self.value_head.predict(state_vector)
        quant_mode = self.config.resolved_quant_mode()

        mode_trace = None
        selected_mode = quant_mode
        if quant_mode == QuantMode.HYBRID:
            selected_index, probabilities = self.mode_head.sample(
                state_vector, self.rng, deterministic=deterministic
            )
            selected_mode = self.supported_modes[selected_index]
            mode_trace = {
                "selected_index": selected_index,
                "probabilities": probabilities,
            }

        traces: list[dict] = []
        if selected_mode in {QuantMode.DISCRETE, QuantMode.DYNAMIC}:
            bit_index, probabilities = self.discrete_head.sample(
                state_vector, self.rng, deterministic=deterministic
            )
            bit_width = self.config.discrete_bit_widths[bit_index]
            decision = QuantizationDecision(
                mode=selected_mode,
                base_bit_width=bit_width,
                scale_factor=1.0,
                clipping_range=1.0,
                precision_level=discrete_precision_level(
                    bit_width, self.config.discrete_bit_widths
                ),
                metadata={"head": "discrete"},
            )
            traces.append(
                {"head": "discrete", "selected_index": bit_index, "probabilities": probabilities}
            )
        elif selected_mode == QuantMode.GROUPED:
            group_bits: list[int] = []
            for group_index, head in enumerate(self.group_heads):
                bit_index, probabilities = head.sample(
                    state_vector, self.rng, deterministic=deterministic
                )
                group_bits.append(self.config.discrete_bit_widths[bit_index])
                traces.append(
                    {
                        "head": "group",
                        "slot": group_index,
                        "selected_index": bit_index,
                        "probabilities": probabilities,
                    }
                )
            decision = QuantizationDecision(
                mode=selected_mode, group_bit_widths=group_bits, metadata={"head": "grouped"}
            )
        elif selected_mode == QuantMode.PER_LAYER:
            layer_bits: list[int] = []
            for layer_index, head in enumerate(self.layer_heads):
                bit_index, probabilities = head.sample(
                    state_vector, self.rng, deterministic=deterministic
                )
                layer_bits.append(self.config.discrete_bit_widths[bit_index])
                traces.append(
                    {
                        "head": "layer",
                        "slot": layer_index,
                        "selected_index": bit_index,
                        "probabilities": probabilities,
                    }
                )
            decision = QuantizationDecision(
                mode=selected_mode, layer_bit_widths=layer_bits, metadata={"head": "per_layer"}
            )
        elif selected_mode == QuantMode.LEARNED:
            bounds = [
                self.config.scale_bounds,
                self.config.clip_bounds,
                self.config.precision_bounds,
            ]
            samples, raw_samples, raw_means = self.learned_head.sample(
                state_vector, self.rng, bounds, deterministic=deterministic
            )
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
                variant_index, probabilities = head.sample(
                    state_vector, self.rng, deterministic=deterministic
                )
                variant_indices.append(variant_index)
                variant_names.append(self.config.moe_variant_names[variant_index])
                traces.append(
                    {
                        "head": "moe",
                        "slot": slot,
                        "selected_index": variant_index,
                        "probabilities": probabilities,
                    }
                )
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
        if isinstance(snapshot.get("mode_head"), dict):
            self.restore_checkpoint_state(snapshot)
            return
        self._validate_live_snapshot(snapshot)
        self.rng.setstate(snapshot["rng_state"])
        self.mode_head = copy.deepcopy(snapshot["mode_head"])
        self.discrete_head = copy.deepcopy(snapshot["discrete_head"])
        self.group_heads = copy.deepcopy(snapshot["group_heads"])
        self.layer_heads = copy.deepcopy(snapshot["layer_heads"])
        self.learned_head = copy.deepcopy(snapshot["learned_head"])
        self.moe_heads = copy.deepcopy(snapshot.get("moe_heads", self.moe_heads))
        self.value_head = copy.deepcopy(snapshot["value_head"])

    def _validate_live_snapshot(self, snapshot: dict[str, object]) -> None:
        required = (
            "rng_state",
            "mode_head",
            "discrete_head",
            "group_heads",
            "layer_heads",
            "learned_head",
            "value_head",
        )
        for key in required:
            if key not in snapshot:
                raise ValueError(f"policy snapshot missing {key!r}")
        mode_head = snapshot["mode_head"]
        if not isinstance(mode_head, CategoricalHead):
            raise TypeError("policy snapshot mode_head must be a CategoricalHead instance")
        if len(mode_head.weights) != len(self.supported_modes):
            raise ValueError("policy snapshot mode_head output dimension mismatch")
        discrete_head = snapshot["discrete_head"]
        if not isinstance(discrete_head, CategoricalHead):
            raise TypeError("policy snapshot discrete_head must be a CategoricalHead instance")
        if len(discrete_head.weights) != len(self.config.discrete_bit_widths):
            raise ValueError("policy snapshot discrete_head output dimension mismatch")
        group_heads = snapshot["group_heads"]
        if not isinstance(group_heads, list) or len(group_heads) != len(self.group_heads):
            raise ValueError("policy snapshot group_heads length mismatch")
        layer_heads = snapshot["layer_heads"]
        if not isinstance(layer_heads, list) or len(layer_heads) != len(self.layer_heads):
            raise ValueError("policy snapshot layer_heads length mismatch")
        moe_heads = snapshot.get("moe_heads", self.moe_heads)
        if not isinstance(moe_heads, list) or len(moe_heads) != len(self.moe_heads):
            raise ValueError("policy snapshot moe_heads length mismatch")

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
        self.moe_heads = [
            _categorical_head_from_payload(item) for item in payload.get("moe_heads", [])
        ]
        self.value_head = _value_head_from_payload(payload["value_head"])
