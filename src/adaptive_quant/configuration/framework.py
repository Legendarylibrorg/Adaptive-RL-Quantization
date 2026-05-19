from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import validation as v
from .reward import RewardWeights
from adaptive_quant.types import HardwareType, QuantMode


@dataclass
class FrameworkConfig:
    """Experiment contract: env sampling, rewards, backends, artifact dirs, and trainer knobs in one place.

    ``training_backend="python"`` keeps the core **simulator + stdlib trainer** path import-light.
    ``training_backend="pytorch"`` selects the CUDA-capable trainer; ``backend="llama_cpp"`` swaps the
    measurement backend for a configured **llama.cpp** CLI. Use ``from_file`` / ``easy_config.load_config``
    for reproducible JSON/TOML; ``clone()`` / ``replace`` preserve validated path semantics.
    """

    training_backend: str = "python"
    multi_hardware: bool = True
    dynamic_quant: bool = True
    learned_quant: bool = True
    moe_enabled: bool = False
    quant_mode: str = QuantMode.HYBRID.value
    detect_host_hardware: bool = True
    hardware_modes: tuple[str, ...] = ("gpu", "cpu", "low_resource")
    discrete_bit_widths: tuple[int, ...] = (2, 4, 8)
    num_groups: int = 4
    num_layers: int = 8
    moe_num_experts: int = 16
    moe_top_k: int = 2
    moe_variant_names: tuple[str, ...] = ("safe", "balanced", "aggressive")
    moe_fixed_variant: str | None = None
    moe_gpu_resident_experts: int = 8
    moe_swap_penalty: float = 0.015
    moe_cache_miss_penalty: float = 0.120
    moe_variant_churn_penalty: float = 0.050
    moe_max_aggressive_experts: int = 1
    moe_max_swap_cost_ms: float = 7.5
    training_episodes: int = 3_000
    evaluation_episodes: int = 400
    benchmark_training_episodes: int | None = None
    benchmark_evaluation_episodes: int | None = None
    recommendation_eval_episodes: int = 96
    recommendation_candidate_limit: int = 12
    continuous_training: bool = False
    eval_interval: int = 1_000
    checkpoint_interval: int = 5_000
    max_training_episodes: int = 50_000
    learning_rate: float = 0.035
    value_learning_rate: float = 0.020
    continuous_stddev: float = 0.18
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
    jsonl_buffered: bool = False
    jsonl_flush_every: int = 1
    write_training_history: bool = True
    write_research_report: bool = True
    resume_from_checkpoint: str | None = None
    backend: str = "simulator"
    training_host_label: str | None = None
    prompt_split_enabled: bool = False
    prompt_split_seed: int = 2027
    prompt_train_fraction: float = 0.8
    env_sampling_mode: str = "random"
    env_forced_prompt_id: str | None = None
    env_forced_hardware: str | None = None
    rl_train_policy_mode: str = "stochastic"
    stability_probe_sampling: str = "random"
    llama_cpp_binary: str | None = None
    llama_cpp_model: str | None = None
    llama_cpp_threads: int = 8
    llama_cpp_context: int = 2048
    llama_cpp_timeout_s: float = 30.0
    llama_cpp_max_prompt_chars: int = 4096
    llama_cpp_generate_tokens: int = 64
    llama_cpp_cache_enabled: bool = False
    llama_cpp_cache_max_entries: int = 256
    external_quality_path: str | None = None
    external_quality_metric: str = "perplexity"
    sim_calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    torch_device: str = "cuda"
    torch_gpu_profile: str = "auto"
    torch_dtype: str = "bfloat16"
    torch_compile: bool = True
    torch_amp: bool = True
    torch_tf32: bool = True
    torch_deterministic: bool = False
    torch_hidden_dim: int = 768
    torch_mlp_depth: int = 4
    torch_learning_rate: float = 3e-4
    torch_weight_decay: float = 1e-4
    torch_batch_episodes: int = 256
    torch_minibatch_size: int = 128
    torch_update_epochs: int = 6
    torch_ppo_clip: float = 0.2
    torch_policy_algorithm: str = "ppo"
    torch_awr_beta: float = 1.0
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
    replay_buffer_capacity: int = 50_000
    replay_buffer_on_gpu: bool = True
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
    router_enabled: bool = False
    router_routes: tuple[str, ...] = ()
    router_feature_backend: str = "hash"
    router_hf_embedding_model: str | None = None
    router_hf_embedding_revision: str | None = None
    router_hf_local_files_only: bool = False
    router_hf_allowed_models: tuple[str, ...] = ()
    router_learning_rate: float = 0.050
    router_value_learning_rate: float = 0.025
    router_exploration: float = 0.10
    router_max_perplexity_ratio: float = 1.05
    router_max_perplexity_delta: float = 0.50
    router_regression_penalty: float = 5_000.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    reward_perplexity_reference: float | None = None
    seed: int = 13

    def __post_init__(self) -> None:
        v.validate_run_name(self.run_name)
        v.validate_discrete_bit_widths(self.discrete_bit_widths)
        v.validate_artifact_dir("outputs_dir", self.outputs_dir)
        v.validate_artifact_dir("log_dir", self.log_dir)
        v.validate_artifact_dir("benchmark_dir", self.benchmark_dir)
        v.validate_artifact_dir("analysis_dir", self.analysis_dir)
        v.validate_artifact_dir("checkpoint_dir", self.checkpoint_dir)
        v.validate_artifact_dir("report_dir", self.report_dir)
        v.validate_optional_filesystem_path("resume_from_checkpoint", self.resume_from_checkpoint)
        v.validate_optional_filesystem_path("llama_cpp_binary", self.llama_cpp_binary)
        v.validate_optional_filesystem_path("llama_cpp_model", self.llama_cpp_model)
        v.validate_optional_filesystem_path("external_quality_path", self.external_quality_path)
        v.validate_backend(self.backend)
        v.validate_torch_policy_algorithm(self.torch_policy_algorithm)
        v.validate_env_sampling_mode(self.env_sampling_mode)
        v.validate_rl_train_policy_mode(self.rl_train_policy_mode)
        v.validate_stability_probe_sampling(self.stability_probe_sampling)
        v.validate_optional_hf_revision("router_hf_embedding_revision", self.router_hf_embedding_revision)
        v.validate_hf_allowed_models(self.router_hf_allowed_models)
        v.validate_positive_int("recommendation_eval_episodes", self.recommendation_eval_episodes)
        v.validate_positive_int("recommendation_candidate_limit", self.recommendation_candidate_limit)
        v.validate_positive_int("llama_cpp_generate_tokens", self.llama_cpp_generate_tokens)
        v.validate_positive_int("jsonl_flush_every", self.jsonl_flush_every)
        v.validate_positive_int("llama_cpp_cache_max_entries", self.llama_cpp_cache_max_entries)
        v.validate_bounded_positive_int("training_episodes", self.training_episodes)
        v.validate_bounded_positive_int("evaluation_episodes", self.evaluation_episodes)
        v.validate_bounded_positive_int("max_training_episodes", self.max_training_episodes)
        if self.benchmark_training_episodes is not None:
            v.validate_bounded_positive_int("benchmark_training_episodes", self.benchmark_training_episodes)
        if self.benchmark_evaluation_episodes is not None:
            v.validate_bounded_positive_int("benchmark_evaluation_episodes", self.benchmark_evaluation_episodes)
        v.validate_bounded_positive_int("online_requests", self.online_requests)
        v.validate_bounded_nonneg_int("replay_buffer_capacity", self.replay_buffer_capacity)
        v.validate_bounded_positive_int("online_replay_capacity", self.online_replay_capacity)
        v.validate_router_routes(self.router_routes)

    def rl_train_deterministic(self) -> bool:
        return self.rl_train_policy_mode.strip().lower() == "deterministic"

    @classmethod
    def reproducible_research(
        cls,
        *,
        seed: int = 13,
        run_name: str = "reproducible_research",
        training_backend: str = "python",
        **kwargs: Any,
    ) -> FrameworkConfig:
        base: dict[str, Any] = {
            "seed": seed,
            "prompt_split_seed": seed,
            "env_sampling_mode": "sequential",
            "rl_train_policy_mode": "deterministic",
            "stability_probe_sampling": "deterministic",
            "torch_deterministic": True,
            "torch_compile": False,
            "run_name": run_name,
            "training_backend": training_backend,
        }
        base.update(kwargs)
        return cls(**base)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        base: FrameworkConfig | None = None,
        strict: bool = False,
    ) -> FrameworkConfig:
        from adaptive_quant.easy_config import config_from_dict

        return config_from_dict(data, base=base, strict=strict)

    @classmethod
    def from_file(cls, path: str | Path, *, strict: bool = True) -> FrameworkConfig:
        from adaptive_quant.easy_config import load_config

        return load_config(path, strict=strict)

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

    def moe_variant_count(self) -> int:
        return len(self.moe_variant_names)

    def default_moe_variant_index(self) -> int:
        if not self.moe_variant_names:
            return 0
        if "balanced" in self.moe_variant_names:
            return self.moe_variant_names.index("balanced")
        return min(1, len(self.moe_variant_names) - 1)

    def moe_variant_index(self, name: str) -> int | None:
        if name in self.moe_variant_names:
            return self.moe_variant_names.index(name)
        return None

    def aggressive_moe_variant_index(self) -> int | None:
        if "aggressive" in self.moe_variant_names:
            return self.moe_variant_names.index("aggressive")
        return None

    def moe_state_dim(self) -> int:
        if not self.moe_enabled:
            return 0
        return 4 + self.moe_top_k * (5 + self.moe_variant_count())

    def state_vector_dim(self) -> int:
        return len(self.ordered_hardware()) + 5 + 2 + self.num_layers + 3 + self.moe_state_dim()

    def clone(self, **changes: object) -> FrameworkConfig:
        reward_weights = changes.pop("reward_weights", replace(self.reward_weights))
        return replace(self, reward_weights=reward_weights, **changes)

    def online_telemetry_path(self) -> str:
        return f"{self.log_dir}/{self.run_name}_online_telemetry.jsonl"

    def online_replay_path(self) -> str:
        return f"{self.log_dir}/{self.run_name}_online_replay.jsonl"

    def summary_path(self) -> str:
        return f"{self.benchmark_dir}/{self.run_name}_summary.json"

    def online_summary_path(self) -> str:
        return f"{self.benchmark_dir}/{self.run_name}_online_summary.json"

    def training_history_path(self) -> str:
        return f"{self.benchmark_dir}/{self.run_name}_training_history.json"

    def recommendation_path(self) -> str:
        return f"{self.benchmark_dir}/{self.run_name}_recommendation.json"

    def final_checkpoint_path(self) -> str:
        return f"{self.checkpoint_dir}/{self.run_name}_final.pt"

    def report_path(self) -> str:
        return f"{self.report_dir}/{self.run_name}_report.md"
