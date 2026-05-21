from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from adaptive_quant.configuration.flat_access import (
    apply_flat_kwargs,
    config_to_flat_dict,
    get_flat_field,
    set_flat_field,
)
from adaptive_quant.configuration.sections import (
    FLAT_FIELD_MAP,
    ArtifactPaths,
    LlamaCppSettings,
    MoESettings,
    OnlineSettings,
    RouterSettings,
    TorchSettings,
    TrainingSettings,
)
from adaptive_quant.types import HardwareType, QuantMode

from . import validation as v


@dataclass
class RewardWeights:
    alpha_latency: float = 0.020
    beta_throughput: float = 0.060
    gamma_perplexity: float = 0.850
    delta_memory: float = 0.002
    epsilon_instability: float = 1.000
    eta_token_latency: float = 0.0
    zeta_perplexity_over_ref: float = 0.0


@dataclass(init=False)
class FrameworkConfig:
    """Experiment contract: env sampling, rewards, backends, artifact dirs, and trainer knobs.

    Settings are grouped in nested sections (``moe``, ``torch``, ``llama_cpp``, …). Flat names such
    as ``training_episodes`` and ``llama_cpp_threads`` remain valid via attribute delegation.
    """

    artifacts: ArtifactPaths
    moe: MoESettings
    llama_cpp: LlamaCppSettings
    torch: TorchSettings
    online: OnlineSettings
    router: RouterSettings
    training: TrainingSettings
    reward_weights: RewardWeights
    training_backend: str
    multi_hardware: bool
    dynamic_quant: bool
    learned_quant: bool
    moe_enabled: bool
    quant_mode: str
    detect_host_hardware: bool
    hardware_modes: tuple[str, ...]
    discrete_bit_widths: tuple[int, ...]
    num_groups: int
    num_layers: int
    backend: str
    training_host_label: str | None
    external_quality_path: str | None
    external_quality_metric: str
    sim_calibration: dict[str, dict[str, float]]
    reward_perplexity_reference: float | None
    seed: int

    def __init__(self, /, **kwargs: Any) -> None:
        object.__setattr__(self, "artifacts", ArtifactPaths())
        object.__setattr__(self, "moe", MoESettings())
        object.__setattr__(self, "llama_cpp", LlamaCppSettings())
        object.__setattr__(self, "torch", TorchSettings())
        object.__setattr__(self, "online", OnlineSettings())
        object.__setattr__(self, "router", RouterSettings())
        object.__setattr__(self, "training", TrainingSettings())
        object.__setattr__(self, "reward_weights", RewardWeights())
        object.__setattr__(self, "training_backend", "python")
        object.__setattr__(self, "multi_hardware", True)
        object.__setattr__(self, "dynamic_quant", True)
        object.__setattr__(self, "learned_quant", True)
        object.__setattr__(self, "moe_enabled", False)
        object.__setattr__(self, "quant_mode", QuantMode.HYBRID.value)
        object.__setattr__(self, "detect_host_hardware", True)
        object.__setattr__(self, "hardware_modes", ("gpu", "cpu", "low_resource"))
        object.__setattr__(self, "discrete_bit_widths", (2, 4, 8))
        object.__setattr__(self, "num_groups", 4)
        object.__setattr__(self, "num_layers", 8)
        object.__setattr__(self, "backend", "simulator")
        object.__setattr__(self, "training_host_label", None)
        object.__setattr__(self, "external_quality_path", None)
        object.__setattr__(self, "external_quality_metric", "perplexity")
        object.__setattr__(self, "sim_calibration", {})
        object.__setattr__(self, "reward_perplexity_reference", None)
        object.__setattr__(self, "seed", 13)
        if kwargs:
            apply_flat_kwargs(self, kwargs)
        self.__post_init__()

    def __getattr__(self, name: str) -> Any:
        if name in FLAT_FIELD_MAP:
            return get_flat_field(self, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in FLAT_FIELD_MAP or name in {
            "training_backend",
            "multi_hardware",
            "dynamic_quant",
            "learned_quant",
            "moe_enabled",
            "quant_mode",
            "detect_host_hardware",
            "hardware_modes",
            "discrete_bit_widths",
            "num_groups",
            "num_layers",
            "backend",
            "training_host_label",
            "external_quality_path",
            "external_quality_metric",
            "sim_calibration",
            "reward_perplexity_reference",
            "seed",
            "reward_weights",
            "artifacts",
            "moe",
            "llama_cpp",
            "torch",
            "online",
            "router",
            "training",
        }:
            if name in FLAT_FIELD_MAP:
                set_flat_field(self, name, value)
            else:
                object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

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
        v.validate_optional_hf_revision(
            "router_hf_embedding_revision", self.router_hf_embedding_revision
        )
        v.validate_hf_allowed_models(self.router_hf_allowed_models)
        v.validate_route_hf_allowed_repos(self.route_hf_allowed_repos)
        v.validate_router_hf_settings(
            router_feature_backend=self.router_feature_backend,
            router_hf_embedding_model=self.router_hf_embedding_model,
            router_hf_embedding_revision=self.router_hf_embedding_revision,
            router_hf_allowed_models=self.router_hf_allowed_models,
        )
        v.validate_bounded_positive_int(
            "recommendation_eval_episodes",
            self.recommendation_eval_episodes,
            ceiling=v.MAX_RECOMMENDATION_EVAL_EPISODES,
        )
        v.validate_bounded_positive_int(
            "recommendation_candidate_limit",
            self.recommendation_candidate_limit,
            ceiling=v.MAX_RECOMMENDATION_CANDIDATE_LIMIT,
        )
        v.validate_bounded_positive_int(
            "llama_cpp_generate_tokens",
            self.llama_cpp_generate_tokens,
            ceiling=v.MAX_LLAMA_CPP_GENERATE_TOKENS,
        )
        v.validate_bounded_positive_int(
            "jsonl_flush_every", self.jsonl_flush_every, ceiling=v.MAX_JSONL_FLUSH_EVERY
        )
        v.validate_bounded_positive_int(
            "llama_cpp_cache_max_entries",
            self.llama_cpp_cache_max_entries,
            ceiling=v.MAX_LLAMA_CPP_CACHE_ENTRIES,
        )
        v.validate_bounded_positive_int("training_episodes", self.training_episodes)
        v.validate_bounded_positive_int("evaluation_episodes", self.evaluation_episodes)
        v.validate_bounded_positive_int("max_training_episodes", self.max_training_episodes)
        if self.benchmark_training_episodes is not None:
            v.validate_bounded_positive_int(
                "benchmark_training_episodes", self.benchmark_training_episodes
            )
        if self.benchmark_evaluation_episodes is not None:
            v.validate_bounded_positive_int(
                "benchmark_evaluation_episodes", self.benchmark_evaluation_episodes
            )
        v.validate_bounded_positive_int("online_requests", self.online_requests)
        v.validate_bounded_nonneg_int("replay_buffer_capacity", self.replay_buffer_capacity)
        v.validate_bounded_positive_int("online_replay_capacity", self.online_replay_capacity)
        v.validate_router_routes(self.router_routes)
        v.validate_moe_topology(
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            gpu_resident=self.moe_gpu_resident_experts,
            max_aggressive=self.moe_max_aggressive_experts,
        )
        v.validate_structural_limits(
            num_groups=self.num_groups,
            num_layers=self.num_layers,
            eval_interval=self.eval_interval,
            checkpoint_interval=self.checkpoint_interval,
            stability_probe_count=self.stability_probe_count,
            log_every_n_episodes=self.log_every_n_episodes,
            llama_cpp_threads=self.llama_cpp_threads,
            llama_cpp_context=self.llama_cpp_context,
            llama_cpp_max_prompt_chars=self.llama_cpp_max_prompt_chars,
            torch_hidden_dim=self.torch_hidden_dim,
            torch_mlp_depth=self.torch_mlp_depth,
            torch_batch_episodes=self.torch_batch_episodes,
            torch_minibatch_size=self.torch_minibatch_size,
            torch_update_epochs=self.torch_update_epochs,
            torch_preflight_batch_size=self.torch_preflight_batch_size,
            torch_preflight_warmup_steps=self.torch_preflight_warmup_steps,
            torch_preflight_steps=self.torch_preflight_steps,
            online_min_replay_size=self.online_min_replay_size,
            online_update_interval=self.online_update_interval,
            online_batch_size=self.online_batch_size,
            online_drift_window=self.online_drift_window,
            online_safe_mode_cooldown=self.online_safe_mode_cooldown,
        )

    def rl_train_deterministic(self) -> bool:
        mode = cast(str, self.rl_train_policy_mode)
        return mode.strip().lower() == "deterministic"

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
        return len(cast(tuple[str, ...], self.moe_variant_names))

    def default_moe_variant_index(self) -> int:
        names = cast(tuple[str, ...], self.moe_variant_names)
        if not names:
            return 0
        if "balanced" in names:
            return names.index("balanced")
        return min(1, len(names) - 1)

    def moe_variant_index(self, name: str) -> int | None:
        names = cast(tuple[str, ...], self.moe_variant_names)
        if name in names:
            return names.index(name)
        return None

    def aggressive_moe_variant_index(self) -> int | None:
        names = cast(tuple[str, ...], self.moe_variant_names)
        if "aggressive" in names:
            return names.index("aggressive")
        return None

    def moe_state_dim(self) -> int:
        if not self.moe_enabled:
            return 0
        top_k = cast(int, self.moe_top_k)
        return 4 + top_k * (5 + self.moe_variant_count())

    def state_vector_dim(self) -> int:
        return len(self.ordered_hardware()) + 5 + 2 + self.num_layers + 3 + self.moe_state_dim()

    def clone(self, **changes: Any) -> FrameworkConfig:
        flat = config_to_flat_dict(self)
        flat.update(changes)
        return FrameworkConfig(**flat)

    def to_flat_dict(self) -> dict[str, Any]:
        return config_to_flat_dict(self)

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
