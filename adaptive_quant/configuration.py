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
    checkpoint_dir: str = "outputs/checkpoints"
    report_dir: str = "outputs/reports"
    run_name: str = "adaptive_universal_policy"
    cache_prompt_features: bool = True
    log_every_n_episodes: int = 1
    write_training_history: bool = True
    write_research_report: bool = True
    resume_from_checkpoint: str | None = None
    backend: str = "simulator"
    llama_cpp_binary: str | None = None
    llama_cpp_model: str | None = None
    llama_cpp_threads: int = 8
    llama_cpp_context: int = 2048
    torch_device: str = "cuda"
    torch_gpu_profile: str = "auto"
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
    online_learning: bool = False
    online_requests: int = 256
    online_exploration_rate: float = 0.12
    online_canary_ratio: float = 0.50
    online_replay_capacity: int = 2048
    online_min_replay_size: int = 64
    online_update_interval: int = 32
    online_batch_size: int = 128
    online_reward_guard: float = 0.75
    online_max_latency_ratio: float = 1.20
    online_max_memory_ratio: float = 1.15
    online_max_perplexity_delta: float = 1.25
    online_drift_window: int = 48
    online_drift_reward_delta: float = 1.50
    online_safe_mode_cooldown: int = 16
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

    def online_telemetry_path(self) -> str:
        return f"{self.log_dir}/{self.run_name}_online_telemetry.jsonl"

    def online_replay_path(self) -> str:
        return f"{self.log_dir}/{self.run_name}_online_replay.jsonl"

    def training_history_path(self) -> str:
        return f"{self.benchmark_dir}/{self.run_name}_training_history.json"

    def final_checkpoint_path(self) -> str:
        return f"{self.checkpoint_dir}/{self.run_name}_final.pt"

    def report_path(self) -> str:
        return f"{self.report_dir}/{self.run_name}_report.md"
