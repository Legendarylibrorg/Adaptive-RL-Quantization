from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HardwareType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    LOW_RESOURCE = "low_resource"


class QuantMode(str, Enum):
    DISCRETE = "discrete"
    GROUPED = "grouped"
    PER_LAYER = "per_layer"
    DYNAMIC = "dynamic"
    LEARNED = "learned"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class HardwareProfile:
    hardware_type: HardwareType
    name: str
    compute_factor: float
    throughput_bias: float
    latency_bias: float
    memory_budget_mb: float
    preferred_bits: float
    kernel_uniformity_preference: float
    ngl: int

    def one_hot(self, ordered_hardware: list[HardwareType]) -> list[float]:
        return [1.0 if self.hardware_type == hardware_type else 0.0 for hardware_type in ordered_hardware]


@dataclass(frozen=True)
class PromptSample:
    prompt_id: str
    text: str
    domain: str


@dataclass(frozen=True)
class InputFeatures:
    prompt_length: int
    token_entropy: float
    token_variance: float
    embedding_norm: float
    complexity_score: float

    def to_vector(self) -> list[float]:
        return [
            min(self.prompt_length / 256.0, 1.5),
            self.token_entropy,
            self.token_variance,
            self.embedding_norm,
            self.complexity_score,
        ]


@dataclass(frozen=True)
class LayerSensitivity:
    attention_sensitivity: float
    ffn_sensitivity: float
    layer_stats: list[float]

    def to_vector(self) -> list[float]:
        return [self.attention_sensitivity, self.ffn_sensitivity, *self.layer_stats]


@dataclass(frozen=True)
class EpisodeState:
    hardware_profile: HardwareProfile
    prompt: PromptSample
    input_features: InputFeatures
    sensitivity: LayerSensitivity
    previous_action: list[float]

    def to_vector(self, ordered_hardware: list[HardwareType]) -> list[float]:
        return [
            *self.hardware_profile.one_hot(ordered_hardware),
            *self.input_features.to_vector(),
            *self.sensitivity.to_vector(),
            *self.previous_action,
        ]


@dataclass
class QuantizationDecision:
    mode: QuantMode
    base_bit_width: int | None = None
    group_bit_widths: list[int] = field(default_factory=list)
    layer_bit_widths: list[int] = field(default_factory=list)
    scale_factor: float = 1.0
    clipping_range: float = 1.0
    precision_level: float = 0.5
    effective_layer_bits: list[float] = field(default_factory=list)
    fallback_applied: bool = False
    unstable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def feedback_vector(self, max_bits: int, scale_upper: float, clip_upper: float) -> list[float]:
        average_bits = sum(self.effective_layer_bits) / len(self.effective_layer_bits) if self.effective_layer_bits else 0.0
        return [
            average_bits / max_bits if max_bits else 0.0,
            self.scale_factor / scale_upper if scale_upper else 0.0,
            self.clipping_range / clip_upper if clip_upper else 0.0,
        ]


@dataclass
class EpisodeMetrics:
    latency_ms: float
    throughput_tps: float
    perplexity: float
    memory_mb: float
    stability_penalty: float
    reward: float


@dataclass
class EpisodeResult:
    state: EpisodeState
    decision: QuantizationDecision
    metrics: EpisodeMetrics

