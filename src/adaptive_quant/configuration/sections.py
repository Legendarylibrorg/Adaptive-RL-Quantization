"""Grouped configuration sections composed by :class:`~adaptive_quant.configuration.FrameworkConfig`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArtifactPaths:
    outputs_dir: str = "outputs"
    log_dir: str = "outputs/logs"
    benchmark_dir: str = "outputs/benchmarks"
    analysis_dir: str = "outputs/analysis"
    checkpoint_dir: str = "outputs/checkpoints"
    report_dir: str = "outputs/reports"
    gguf_export_dir: str = "outputs/gguf"
    run_name: str = "adaptive_universal_policy"


@dataclass
class RustSettings:
    """Optional Rust CLI accelerators (Python remains orchestrator)."""

    simulator_enabled: bool = False
    cli_binary: str | None = None


@dataclass
class MoESettings:
    num_experts: int = 16
    top_k: int = 2
    variant_names: tuple[str, ...] = ("safe", "balanced", "aggressive")
    fixed_variant: str | None = None
    gpu_resident_experts: int = 8
    swap_penalty: float = 0.015
    cache_miss_penalty: float = 0.120
    variant_churn_penalty: float = 0.050
    max_aggressive_experts: int = 1
    max_swap_cost_ms: float = 7.5


@dataclass
class LlamaCppSettings:
    binary: str | None = None
    model: str | None = None
    threads: int = 8
    context: int = 2048
    timeout_s: float = 30.0
    max_prompt_chars: int = 4096
    generate_tokens: int = 64
    cache_enabled: bool = False
    cache_max_entries: int = 256
    gguf_export_enabled: bool = False
    gguf_export_source: str | None = None
    gguf_export_quant_type: str = "Q4_K_M"
    gguf_quantize_binary: str | None = None
    gguf_export_allow_requantize: bool = False


@dataclass
class TorchSettings:
    device: str = "cuda"
    require_cuda: bool = False
    gpu_profile: str = "auto"
    dtype: str = "bfloat16"
    compile: bool = True
    amp: bool = True
    tf32: bool = True
    deterministic: bool = False
    hidden_dim: int = 768
    mlp_depth: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_episodes: int = 256
    minibatch_size: int = 128
    update_epochs: int = 6
    ppo_clip: float = 0.2
    policy_algorithm: str = "ppo"
    awr_beta: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    fused_optimizer: bool = True
    preflight: bool = True
    preflight_batch_size: int = 4096
    preflight_warmup_steps: int = 10
    preflight_steps: int = 40
    preflight_min_free_memory_gb: float = 8.0


@dataclass
class OnlineSettings:
    learning: bool = False
    requests: int = 256
    exploration_rate: float = 0.12
    canary_ratio: float = 0.50
    replay_buffer_capacity: int = 50_000
    replay_buffer_on_gpu: bool = True
    replay_capacity: int = 2048
    min_replay_size: int = 64
    update_interval: int = 32
    batch_size: int = 128
    reward_guard: float = 0.75
    max_latency_ratio: float = 1.20
    max_memory_ratio: float = 1.15
    max_perplexity_delta: float = 1.25
    drift_window: int = 48
    drift_reward_delta: float = 1.50
    safe_mode_cooldown: int = 16
    max_replay_entries_per_prompt_hash: int = 32


@dataclass
class RouterSettings:
    enabled: bool = False
    routes: tuple[str, ...] = ()
    feature_backend: str = "hash"
    hf_embedding_model: str | None = None
    hf_embedding_revision: str | None = None
    hf_local_files_only: bool = False
    hf_allowed_models: tuple[str, ...] = ()
    route_hf_allowed_repos: tuple[str, ...] = ()
    learning_rate: float = 0.050
    value_learning_rate: float = 0.025
    exploration: float = 0.10
    max_perplexity_ratio: float = 1.05
    max_perplexity_delta: float = 0.50
    regression_penalty: float = 5_000.0


@dataclass
class TrainingSettings:
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
    cache_prompt_features: bool = True
    log_every_n_episodes: int = 1
    jsonl_buffered: bool = False
    jsonl_flush_every: int = 1
    write_training_history: bool = True
    write_research_report: bool = True
    resume_from_checkpoint: str | None = None
    prompt_split_enabled: bool = False
    prompt_split_seed: int = 2027
    prompt_train_fraction: float = 0.8
    env_sampling_mode: str = "random"
    env_forced_prompt_id: str | None = None
    env_forced_hardware: str | None = None
    rl_train_policy_mode: str = "stochastic"
    stability_probe_sampling: str = "random"
    jsonl_integrity_chain: bool = False
    replay_manifest_enabled: bool = False
    replay_verify_after_run: bool = True
    prompt_library_path: str | None = None


# Flat JSON/TOML keys → (section attribute on FrameworkConfig, field on section dataclass).
FLAT_FIELD_MAP: dict[str, tuple[str, str]] = {}

_ARTIFACT_FLAT = {
    "outputs_dir": ("artifacts", "outputs_dir"),
    "log_dir": ("artifacts", "log_dir"),
    "benchmark_dir": ("artifacts", "benchmark_dir"),
    "analysis_dir": ("artifacts", "analysis_dir"),
    "checkpoint_dir": ("artifacts", "checkpoint_dir"),
    "report_dir": ("artifacts", "report_dir"),
    "gguf_export_dir": ("artifacts", "gguf_export_dir"),
    "run_name": ("artifacts", "run_name"),
}
FLAT_FIELD_MAP.update(_ARTIFACT_FLAT)

for _field in MoESettings.__dataclass_fields__:
    FLAT_FIELD_MAP[f"moe_{_field}"] = ("moe", _field)

for _field in LlamaCppSettings.__dataclass_fields__:
    FLAT_FIELD_MAP[f"llama_cpp_{_field}"] = ("llama_cpp", _field)

for _field in RustSettings.__dataclass_fields__:
    FLAT_FIELD_MAP[f"rust_{_field}"] = ("rust", _field)

for _field in TorchSettings.__dataclass_fields__:
    FLAT_FIELD_MAP[f"torch_{_field}"] = ("torch", _field)

_ONLINE_FLAT: dict[str, tuple[str, str]] = {
    "online_learning": ("online", "learning"),
    "replay_buffer_capacity": ("online", "replay_buffer_capacity"),
    "replay_buffer_on_gpu": ("online", "replay_buffer_on_gpu"),
}
for _field in OnlineSettings.__dataclass_fields__:
    if _field in ("learning", "replay_buffer_capacity", "replay_buffer_on_gpu"):
        continue
    FLAT_FIELD_MAP[f"online_{_field}"] = ("online", _field)
FLAT_FIELD_MAP.update(_ONLINE_FLAT)

for _field in RouterSettings.__dataclass_fields__:
    if _field == "route_hf_allowed_repos":
        FLAT_FIELD_MAP["route_hf_allowed_repos"] = ("router", _field)
    else:
        FLAT_FIELD_MAP[f"router_{_field}"] = ("router", _field)

for _field in TrainingSettings.__dataclass_fields__:
    FLAT_FIELD_MAP[_field] = ("training", _field)

NESTED_SECTION_KEYS = frozenset(
    {
        "artifacts",
        "moe",
        "llama_cpp",
        "rust",
        "torch",
        "online",
        "router",
        "training",
        "reward_weights",
    }
)

SECTION_TYPES: dict[str, type] = {
    "artifacts": ArtifactPaths,
    "moe": MoESettings,
    "llama_cpp": LlamaCppSettings,
    "rust": RustSettings,
    "torch": TorchSettings,
    "online": OnlineSettings,
    "router": RouterSettings,
    "training": TrainingSettings,
}
