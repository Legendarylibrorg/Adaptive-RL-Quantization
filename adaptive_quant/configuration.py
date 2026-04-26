from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from adaptive_quant.types import HardwareType, QuantMode


@dataclass
class RewardWeights:
    alpha_latency: float = 0.020
    beta_throughput: float = 0.060
    gamma_perplexity: float = 0.850
    delta_memory: float = 0.002
    epsilon_instability: float = 1.000
    # Hinge on perplexity vs reward_perplexity_reference (0 disables the extra term).
    zeta_perplexity_over_ref: float = 0.0


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
    # When False (default), only split checkpoints (.pt + .checkpoint.json) load; legacy single-file
    # pickle checkpoints require explicit opt-in (trusted source only).
    allow_legacy_checkpoint_load: bool = False
    backend: str = "simulator"
    training_host_label: str | None = None
    prompt_split_enabled: bool = False
    prompt_split_seed: int = 2027
    prompt_train_fraction: float = 0.8
    # Research env: prompt/hardware schedule. random (default) | sequential | forced
    env_sampling_mode: str = "random"
    # Used when env_sampling_mode='forced' and reset() does not pass explicit prompt/hardware.
    env_forced_prompt_id: str | None = None
    env_forced_hardware: str | None = None
    # Policy during training data collection: stochastic (sample π) | deterministic (argmax π)
    rl_train_policy_mode: str = "stochastic"
    # Extra probes for stability_penalty: random | deterministic (sorted prompt ids)
    stability_probe_sampling: str = "random"
    llama_cpp_binary: str | None = None
    llama_cpp_model: str | None = None
    llama_cpp_threads: int = 8
    llama_cpp_context: int = 2048
    llama_cpp_timeout_s: float = 30.0
    llama_cpp_max_prompt_chars: int = 4096
    sim_calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    torch_device: str = "cuda"
    torch_gpu_profile: str = "auto"
    torch_dtype: str = "bfloat16"
    torch_compile: bool = True
    torch_amp: bool = True
    torch_tf32: bool = True
    # Strict reproducibility: CUDNN deterministic, CUBLAS workspace, seeds, deterministic algorithms (slower).
    torch_deterministic: bool = False
    torch_hidden_dim: int = 768
    torch_mlp_depth: int = 4
    torch_learning_rate: float = 3e-4
    torch_weight_decay: float = 1e-4
    torch_batch_episodes: int = 256
    torch_minibatch_size: int = 128
    torch_update_epochs: int = 6
    torch_ppo_clip: float = 0.2
    # Policy gradient variant for training_backend=pytorch: ppo | vpg | awr
    torch_policy_algorithm: str = "ppo"
    # Temperature for awr weights exp(advantage / beta); ignored for ppo/vpg.
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
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    # Simulator / offline RL: soft constraint to push tok/s while discouraging quality regression.
    reward_perplexity_reference: float | None = None
    seed: int = 13

    def __post_init__(self) -> None:
        _validate_run_name(self.run_name)
        _validate_artifact_dir("outputs_dir", self.outputs_dir)
        _validate_artifact_dir("log_dir", self.log_dir)
        _validate_artifact_dir("benchmark_dir", self.benchmark_dir)
        _validate_artifact_dir("analysis_dir", self.analysis_dir)
        _validate_artifact_dir("checkpoint_dir", self.checkpoint_dir)
        _validate_artifact_dir("report_dir", self.report_dir)
        _validate_optional_filesystem_path("resume_from_checkpoint", self.resume_from_checkpoint)
        _validate_optional_filesystem_path("llama_cpp_binary", self.llama_cpp_binary)
        _validate_optional_filesystem_path("llama_cpp_model", self.llama_cpp_model)
        _validate_torch_policy_algorithm(self.torch_policy_algorithm)
        _validate_env_sampling_mode(self.env_sampling_mode)
        _validate_rl_train_policy_mode(self.rl_train_policy_mode)
        _validate_stability_probe_sampling(self.stability_probe_sampling)
        _validate_positive_int("recommendation_eval_episodes", self.recommendation_eval_episodes)
        _validate_positive_int("recommendation_candidate_limit", self.recommendation_candidate_limit)

    def rl_train_deterministic(self) -> bool:
        """True when train rollouts use greedy (argmax) actions for reproducible bandit-style experiments."""
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
        """
        One-shot preset for research runs: sequential (prompt, hardware) schedule,
        greedy policy during train collection, deterministic stability probes, aligned
        prompt_split_seed; with training_backend=\"pytorch\" also enables torch_deterministic
        and disables torch.compile. Pass any FrameworkConfig field via kwargs to override.
        """
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
        """Merge a plain dict (JSON/TOML-friendly) into a config; see ``adaptive_quant.easy_config``."""
        from adaptive_quant.easy_config import config_from_dict

        return config_from_dict(data, base=base, strict=strict)

    @classmethod
    def from_file(cls, path: str | Path, *, strict: bool = True) -> FrameworkConfig:
        """Load ``.json`` or ``.toml`` with optional ``preset`` key; strict by default."""
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

    def clone(self, **changes: object) -> "FrameworkConfig":
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


_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_TORCH_POLICY_ALGORITHMS = frozenset({"ppo", "vpg", "awr"})
_ENV_SAMPLING_MODES = frozenset({"random", "sequential", "forced"})
_RL_TRAIN_POLICY_MODES = frozenset({"stochastic", "deterministic"})
_STABILITY_PROBE_SAMPLING = frozenset({"random", "deterministic"})


def _validate_env_sampling_mode(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("env_sampling_mode must be a string")
    key = name.strip().lower()
    if key not in _ENV_SAMPLING_MODES:
        raise ValueError(
            f"env_sampling_mode must be one of {sorted(_ENV_SAMPLING_MODES)}, got {name!r}"
        )


def _validate_rl_train_policy_mode(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("rl_train_policy_mode must be a string")
    key = name.strip().lower()
    if key not in _RL_TRAIN_POLICY_MODES:
        raise ValueError(
            f"rl_train_policy_mode must be one of {sorted(_RL_TRAIN_POLICY_MODES)}, got {name!r}"
        )


def _validate_stability_probe_sampling(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("stability_probe_sampling must be a string")
    key = name.strip().lower()
    if key not in _STABILITY_PROBE_SAMPLING:
        raise ValueError(
            f"stability_probe_sampling must be one of {sorted(_STABILITY_PROBE_SAMPLING)}, got {name!r}"
        )


def _validate_torch_policy_algorithm(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("torch_policy_algorithm must be a string")
    key = name.strip().lower()
    if key not in _TORCH_POLICY_ALGORITHMS:
        raise ValueError(
            f"torch_policy_algorithm must be one of {sorted(_TORCH_POLICY_ALGORITHMS)}, got {name!r}"
        )


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _validate_run_name(run_name: str) -> None:
    """
    Prevent path traversal / odd filesystem behavior.

    - No path separators or '..'
    - ASCII-ish slug for stable filenames
    """
    if not isinstance(run_name, str):
        raise TypeError("run_name must be a string")
    if "/" in run_name or "\\" in run_name or "\x00" in run_name or ".." in run_name:
        raise ValueError(f"Invalid run_name {run_name!r}: must not contain path separators, NUL, or '..'")
    if not _RUN_NAME_RE.match(run_name):
        raise ValueError(
            f"Invalid run_name {run_name!r}: expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/"
        )


def _path_has_parent_reference(path: str) -> bool:
    """True if any path component is '..', after normalizing to a Path."""
    return ".." in Path(path).parts


def _validate_artifact_dir(field_name: str, path: str) -> None:
    """
    Reject traversal / control characters in directory prefixes used with run_name-based filenames.

    Absolute paths are allowed (e.g. /data/outputs); '..' components are not.
    """
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string")
    stripped = path.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"{field_name} contains invalid control characters")
    if _path_has_parent_reference(path):
        raise ValueError(f"{field_name} must not contain '..' ({path!r})")


def _validate_optional_filesystem_path(field_name: str, path: str | None) -> None:
    """Same rules as artifact dirs when the field is set (None is allowed)."""
    if path is None:
        return
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string or None")
    stripped = path.strip()
    if not stripped:
        raise ValueError(f"{field_name} if set must be non-empty")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"{field_name} contains invalid control characters")
    if _path_has_parent_reference(path):
        raise ValueError(f"{field_name} must not contain '..' ({path!r})")
