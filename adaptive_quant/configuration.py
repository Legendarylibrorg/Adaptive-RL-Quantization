from __future__ import annotations

from dataclasses import dataclass, field, replace

from adaptive_quant.types import HardwareType, QuantMode


@dataclass
class RewardWeights:
    alpha_latency: float = 0.020
    beta_throughput: float = 0.060
    gamma_perplexity: float = 0.850
    delta_memory: float = 0.002
    epsilon_instability: float = 1.000


@dataclass
class FrameworkConfig:
    training_backend: str = "python"
    multi_hardware: bool = True
    dynamic_quant: bool = True
    learned_quant: bool = True
    quant_mode: str = QuantMode.HYBRID.value
    hardware_modes: tuple[str, ...] = ("gpu", "cpu", "low_resource")
    discrete_bit_widths: tuple[int, ...] = (2, 4, 8)
    num_groups: int = 4
    num_layers: int = 8
    training_episodes: int = 240
    evaluation_episodes: int = 60
    benchmark_training_episodes: int | None = None
    benchmark_evaluation_episodes: int | None = None
    learning_rate: float = 0.035
    value_learning_rate: float = 0.020
    continuous_stddev: float = 0.18
    entropy_bonus: float = 0.005
    stability_probe_count: int = 3
    instability_threshold: float = 2.5
    safe_default_bits: int = 4
    scale_bounds: tuple[float, float] = (0.45, 1.85)
    clip_bounds: tuple[float, float] = (0.55, 2.40)
    precision_bounds: tuple[float, float] = (0.0, 1.0)
    outputs_dir: str = "outputs"
    log_dir: str = "outputs/logs"
    benchmark_dir: str = "outputs/benchmarks"
    analysis_dir: str = "outputs/analysis"
    run_name: str = "adaptive_universal_policy"
    cache_prompt_features: bool = True
    log_every_n_episodes: int = 1
    backend: str = "simulator"
    llama_cpp_binary: str | None = None
    llama_cpp_model: str | None = None
    llama_cpp_threads: int = 8
    llama_cpp_context: int = 2048
    torch_device: str = "cuda"
    torch_dtype: str = "bfloat16"
    torch_compile: bool = True
    torch_amp: bool = True
    torch_tf32: bool = True
    torch_hidden_dim: int = 768
    torch_mlp_depth: int = 4
    torch_learning_rate: float = 3e-4
    torch_weight_decay: float = 1e-4
    torch_batch_episodes: int = 256
    torch_minibatch_size: int = 128
    torch_update_epochs: int = 6
    torch_ppo_clip: float = 0.2
    torch_value_coef: float = 0.5
    torch_entropy_coef: float = 0.01
    torch_max_grad_norm: float = 1.0
    torch_fused_optimizer: bool = True
    torch_preflight: bool = True
    torch_preflight_batch_size: int = 4096
    torch_preflight_warmup_steps: int = 10
    torch_preflight_steps: int = 40
    torch_preflight_min_free_memory_gb: float = 8.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    seed: int = 13

    def resolved_quant_mode(self) -> QuantMode:
        return QuantMode(self.quant_mode)

    def ordered_hardware(self) -> list[HardwareType]:
        hardware: list[HardwareType] = []
        for raw_value in self.hardware_modes:
            hardware.append(HardwareType(raw_value))
        return hardware

    def supported_modes(self) -> list[QuantMode]:
        modes = [QuantMode.DISCRETE, QuantMode.GROUPED, QuantMode.PER_LAYER]
        if self.dynamic_quant:
            modes.append(QuantMode.DYNAMIC)
        if self.learned_quant:
            modes.append(QuantMode.LEARNED)
        return modes

    def clone(self, **changes: object) -> "FrameworkConfig":
        return replace(self, **changes)
