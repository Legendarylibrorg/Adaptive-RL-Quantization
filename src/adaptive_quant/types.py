from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from adaptive_quant.math_utils import mean


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
        return [
            1.0 if self.hardware_type == hardware_type else 0.0
            for hardware_type in ordered_hardware
        ]


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
class MoEExpertState:
    expert_index: int
    router_probability: float
    sensitivity: float
    resident_on_device: float
    hotness: float
    available_variants_mask: list[float]

    def to_vector(self, num_experts: int) -> list[float]:
        expert_scale = max(1, num_experts - 1)
        return [
            self.expert_index / expert_scale,
            self.router_probability,
            self.sensitivity,
            self.resident_on_device,
            self.hotness,
            *self.available_variants_mask,
        ]


@dataclass(frozen=True)
class MoEContext:
    router_entropy: float
    active_expert_count: float
    cache_pressure: float
    estimated_swap_cost_ms: float
    experts: list[MoEExpertState]
    top_k: int
    variant_count: int
    num_experts: int

    def to_vector(self) -> list[float]:
        vector = [
            self.router_entropy,
            self.active_expert_count / max(1, self.top_k),
            self.cache_pressure,
            min(self.estimated_swap_cost_ms / 20.0, 2.0),
        ]
        slot_width = 5 + self.variant_count
        for slot in range(self.top_k):
            if slot < len(self.experts):
                vector.extend(self.experts[slot].to_vector(self.num_experts))
            else:
                vector.extend([0.0] * slot_width)
        return vector


@dataclass(frozen=True)
class EpisodeState:
    hardware_profile: HardwareProfile
    prompt: PromptSample
    input_features: InputFeatures
    sensitivity: LayerSensitivity
    previous_action: list[float]
    moe_context: MoEContext | None = None

    def to_vector(self, ordered_hardware: list[HardwareType]) -> list[float]:
        vector = [
            *self.hardware_profile.one_hot(ordered_hardware),
            *self.input_features.to_vector(),
            *self.sensitivity.to_vector(),
            *self.previous_action,
        ]
        if self.moe_context is not None:
            vector.extend(self.moe_context.to_vector())
        return vector


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
    moe_variant_indices: list[int] = field(default_factory=list)
    moe_variant_names: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    unstable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def feedback_vector(self, max_bits: int, scale_upper: float, clip_upper: float) -> list[float]:
        average_bits = mean(self.effective_layer_bits)
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
    tokens_processed: float = 0.0
    latency_ms_per_token: float = 0.0
    swap_cost_ms: float = 0.0
    cache_miss_count: float = 0.0
    variant_churn: float = 0.0
    latency_source: str = ""
    throughput_source: str = ""
    memory_source: str = ""
    perplexity_source: str = ""


@dataclass
class EpisodeResult:
    state: EpisodeState
    decision: QuantizationDecision
    metrics: EpisodeMetrics


@dataclass(frozen=True)
class OnlineRequest:
    prompt_text: str
    hardware: HardwareType = HardwareType.GPU
    prompt_id: str | None = None
    prompt_domain: str = "online"


class BackendMetricRequired(TypedDict):
    latency_ms: float
    throughput_tps: float
    perplexity: float
    memory_mb: float


class BackendMetricDict(BackendMetricRequired, total=False):
    """Backend evaluation outputs used to compute rewards and build EpisodeMetrics.

    Required keys are the cross-backend contract; optional keys may be absent depending on backend.
    This is implemented without ``typing.NotRequired`` so the module remains importable on older
    Python interpreters even if they are not supported for full runs.
    """

    tokens_processed: float
    latency_ms_per_token: float
    swap_cost_ms: float
    cache_miss_count: float
    variant_churn: float
    latency_source: str
    throughput_source: str
    memory_source: str
    perplexity_source: str
