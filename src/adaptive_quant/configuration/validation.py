from __future__ import annotations

import os
import re
from pathlib import Path

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HF_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

# Absolute ceiling for episode / buffer counters loaded from JSON/TOML (DoS guard).
MAX_EPISODE_COUNT = 1_000_000

# Architecture / workload ceilings (generous vs presets; block hostile JSON/TOML blow-ups).
MAX_NUM_LAYERS = 512
MAX_NUM_GROUPS = 256
MAX_MOE_EXPERTS = 4_096
MAX_MOE_TOP_K = 64
MAX_TORCH_HIDDEN_DIM = 8_192
MAX_TORCH_MLP_DEPTH = 32
MAX_TORCH_BATCH_EPISODES = 65_536
MAX_TORCH_MINIBATCH_SIZE = 16_384
MAX_TORCH_UPDATE_EPOCHS = 128
MAX_TORCH_PREFLIGHT_BATCH_SIZE = 65_536
MAX_TORCH_PREFLIGHT_STEPS = 10_000
MAX_LLAMA_CPP_THREADS = 256
MAX_LLAMA_CPP_CONTEXT = 262_144
MAX_LLAMA_CPP_MAX_PROMPT_CHARS = 1_048_576
MAX_LOG_EVERY_N_EPISODES = 100_000
MAX_ONLINE_BATCH_SIZE = 16_384
MAX_STABILITY_PROBE_COUNT = 1_024

_LLAMA_CPP_BINARY_PREFIXES_ENV = "ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES"

_BACKENDS = frozenset({"simulator", "llama_cpp"})
_TORCH_POLICY_ALGORITHMS = frozenset({"ppo", "vpg", "awr"})
_ENV_SAMPLING_MODES = frozenset({"random", "sequential", "forced"})
_RL_TRAIN_POLICY_MODES = frozenset({"stochastic", "deterministic"})
_STABILITY_PROBE_SAMPLING = frozenset({"random", "deterministic"})


def validate_backend(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("backend must be a string")
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {name!r}")


def validate_env_sampling_mode(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("env_sampling_mode must be a string")
    key = name.strip().lower()
    if key not in _ENV_SAMPLING_MODES:
        raise ValueError(
            f"env_sampling_mode must be one of {sorted(_ENV_SAMPLING_MODES)}, got {name!r}"
        )


def validate_rl_train_policy_mode(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("rl_train_policy_mode must be a string")
    key = name.strip().lower()
    if key not in _RL_TRAIN_POLICY_MODES:
        raise ValueError(
            f"rl_train_policy_mode must be one of {sorted(_RL_TRAIN_POLICY_MODES)}, got {name!r}"
        )


def validate_stability_probe_sampling(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("stability_probe_sampling must be a string")
    key = name.strip().lower()
    if key not in _STABILITY_PROBE_SAMPLING:
        raise ValueError(
            f"stability_probe_sampling must be one of {sorted(_STABILITY_PROBE_SAMPLING)}, got {name!r}"
        )


def validate_torch_policy_algorithm(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("torch_policy_algorithm must be a string")
    key = name.strip().lower()
    if key not in _TORCH_POLICY_ALGORITHMS:
        raise ValueError(
            f"torch_policy_algorithm must be one of {sorted(_TORCH_POLICY_ALGORITHMS)}, got {name!r}"
        )


def validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def validate_bounded_positive_int(
    name: str,
    value: int,
    *,
    ceiling: int = MAX_EPISODE_COUNT,
) -> None:
    validate_positive_int(name, value)
    if value > ceiling:
        raise ValueError(f"{name} must be <= {ceiling}, got {value!r}")


def validate_bounded_nonneg_int(
    name: str,
    value: int,
    *,
    ceiling: int = MAX_EPISODE_COUNT,
) -> None:
    """Like :func:`validate_bounded_positive_int` but allows zero (e.g. disabled replay buffer)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    if value > ceiling:
        raise ValueError(f"{name} must be <= {ceiling}, got {value!r}")


def _validate_path_string(field_name: str, path: str) -> None:
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string")
    if not path.strip():
        raise ValueError(f"{field_name} must be non-empty")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"{field_name} contains invalid control characters")
    if path_has_parent_reference(path):
        raise ValueError(f"{field_name} must not contain '..' ({path!r})")


def validate_discrete_bit_widths(values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError("discrete_bit_widths must be a tuple of ints")
    if not values:
        raise ValueError("discrete_bit_widths must be non-empty")
    seen: set[int] = set()
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError("discrete_bit_widths must contain ints")
        if item <= 0:
            raise ValueError(f"discrete_bit_widths must be > 0, got {item!r}")
        if item in seen:
            raise ValueError(f"discrete_bit_widths must not contain duplicates (got {item!r} twice)")
        seen.add(item)


def validate_run_name(run_name: str) -> None:
    if not isinstance(run_name, str):
        raise TypeError("run_name must be a string")
    if "/" in run_name or "\\" in run_name or "\x00" in run_name or ".." in run_name:
        raise ValueError(f"Invalid run_name {run_name!r}: must not contain path separators, NUL, or '..'")
    if not _RUN_NAME_RE.match(run_name):
        raise ValueError(
            f"Invalid run_name {run_name!r}: expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/"
        )


def path_has_parent_reference(path: str) -> bool:
    return ".." in Path(path).parts


def validate_artifact_dir(field_name: str, path: str) -> None:
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string")
    _validate_path_string(field_name, path)


def validate_optional_filesystem_path(field_name: str, path: str | None) -> None:
    if path is None:
        return
    _validate_path_string(field_name, path)


def validate_runtime_filesystem_path(field_name: str, path: str) -> None:
    """Validate a required on-disk path (llama.cpp binary/model, route GGUF, etc.)."""
    _validate_path_string(field_name, path)


def validate_llama_cpp_binary_allowlist(resolved_binary: str) -> None:
    """When ``ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES`` is set, require the binary under those roots.

    Prefixes are separated by ``os.pathsep`` (``:`` on Unix, ``;`` on Windows). Each entry is
    resolved with :func:`os.path.realpath` before comparison. When the env var is unset, this is
    a no-op so local dev workflows stay unchanged.
    """
    raw = os.environ.get(_LLAMA_CPP_BINARY_PREFIXES_ENV, "").strip()
    if not raw:
        return
    prefixes: list[str] = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            prefixes.append(os.path.realpath(entry))
        except OSError as exc:
            raise ValueError(
                f"{_LLAMA_CPP_BINARY_PREFIXES_ENV} contains invalid prefix {entry!r}: {exc}"
            ) from exc
    if not prefixes:
        return
    if not any(
        resolved_binary == prefix or resolved_binary.startswith(prefix + os.sep) for prefix in prefixes
    ):
        raise ValueError(
            f"llama_cpp_binary resolves to {resolved_binary!r}, which is outside "
            f"{_LLAMA_CPP_BINARY_PREFIXES_ENV}={raw!r}"
        )


def validate_moe_topology(*, num_experts: int, top_k: int, gpu_resident: int, max_aggressive: int) -> None:
    validate_bounded_positive_int("moe_num_experts", num_experts, ceiling=MAX_MOE_EXPERTS)
    validate_bounded_positive_int("moe_top_k", top_k, ceiling=MAX_MOE_TOP_K)
    if top_k > num_experts:
        raise ValueError(f"moe_top_k ({top_k}) must be <= moe_num_experts ({num_experts})")
    validate_bounded_positive_int("moe_gpu_resident_experts", gpu_resident, ceiling=MAX_MOE_EXPERTS)
    if gpu_resident > num_experts:
        raise ValueError(
            f"moe_gpu_resident_experts ({gpu_resident}) must be <= moe_num_experts ({num_experts})"
        )
    validate_bounded_positive_int("moe_max_aggressive_experts", max_aggressive, ceiling=MAX_MOE_EXPERTS)
    if max_aggressive > num_experts:
        raise ValueError(
            f"moe_max_aggressive_experts ({max_aggressive}) must be <= moe_num_experts ({num_experts})"
        )


def validate_structural_limits(
    *,
    num_groups: int,
    num_layers: int,
    eval_interval: int,
    checkpoint_interval: int,
    stability_probe_count: int,
    log_every_n_episodes: int,
    llama_cpp_threads: int,
    llama_cpp_context: int,
    llama_cpp_max_prompt_chars: int,
    torch_hidden_dim: int,
    torch_mlp_depth: int,
    torch_batch_episodes: int,
    torch_minibatch_size: int,
    torch_update_epochs: int,
    torch_preflight_batch_size: int,
    torch_preflight_warmup_steps: int,
    torch_preflight_steps: int,
    online_min_replay_size: int,
    online_update_interval: int,
    online_batch_size: int,
    online_drift_window: int,
    online_safe_mode_cooldown: int,
) -> None:
    """Reject pathological architecture / workload integers from untrusted config files."""
    validate_bounded_positive_int("num_groups", num_groups, ceiling=MAX_NUM_GROUPS)
    validate_bounded_positive_int("num_layers", num_layers, ceiling=MAX_NUM_LAYERS)
    validate_bounded_nonneg_int("eval_interval", eval_interval, ceiling=MAX_EPISODE_COUNT)
    validate_bounded_nonneg_int("checkpoint_interval", checkpoint_interval, ceiling=MAX_EPISODE_COUNT)
    validate_bounded_positive_int(
        "stability_probe_count", stability_probe_count, ceiling=MAX_STABILITY_PROBE_COUNT
    )
    validate_bounded_positive_int("log_every_n_episodes", log_every_n_episodes, ceiling=MAX_LOG_EVERY_N_EPISODES)
    validate_bounded_positive_int("llama_cpp_threads", llama_cpp_threads, ceiling=MAX_LLAMA_CPP_THREADS)
    validate_bounded_positive_int("llama_cpp_context", llama_cpp_context, ceiling=MAX_LLAMA_CPP_CONTEXT)
    validate_bounded_positive_int(
        "llama_cpp_max_prompt_chars", llama_cpp_max_prompt_chars, ceiling=MAX_LLAMA_CPP_MAX_PROMPT_CHARS
    )
    validate_bounded_positive_int("torch_hidden_dim", torch_hidden_dim, ceiling=MAX_TORCH_HIDDEN_DIM)
    validate_bounded_positive_int("torch_mlp_depth", torch_mlp_depth, ceiling=MAX_TORCH_MLP_DEPTH)
    validate_bounded_positive_int("torch_batch_episodes", torch_batch_episodes, ceiling=MAX_TORCH_BATCH_EPISODES)
    validate_bounded_positive_int("torch_minibatch_size", torch_minibatch_size, ceiling=MAX_TORCH_MINIBATCH_SIZE)
    validate_bounded_positive_int("torch_update_epochs", torch_update_epochs, ceiling=MAX_TORCH_UPDATE_EPOCHS)
    validate_bounded_positive_int(
        "torch_preflight_batch_size", torch_preflight_batch_size, ceiling=MAX_TORCH_PREFLIGHT_BATCH_SIZE
    )
    validate_bounded_positive_int(
        "torch_preflight_warmup_steps", torch_preflight_warmup_steps, ceiling=MAX_TORCH_PREFLIGHT_STEPS
    )
    validate_bounded_positive_int("torch_preflight_steps", torch_preflight_steps, ceiling=MAX_TORCH_PREFLIGHT_STEPS)
    validate_bounded_positive_int("online_min_replay_size", online_min_replay_size, ceiling=MAX_EPISODE_COUNT)
    validate_bounded_positive_int("online_update_interval", online_update_interval, ceiling=MAX_EPISODE_COUNT)
    validate_bounded_positive_int("online_batch_size", online_batch_size, ceiling=MAX_ONLINE_BATCH_SIZE)
    validate_bounded_positive_int("online_drift_window", online_drift_window, ceiling=MAX_EPISODE_COUNT)
    validate_bounded_positive_int("online_safe_mode_cooldown", online_safe_mode_cooldown, ceiling=MAX_EPISODE_COUNT)


def validate_router_routes(routes: tuple[str, ...]) -> None:
    """Parse every configured router route so hostile paths fail at config load time."""
    if not routes:
        return
    from adaptive_quant.routing import parse_route

    for index, route in enumerate(routes):
        if not isinstance(route, str) or not route.strip():
            raise ValueError(f"router_routes[{index}] must be a non-empty string")
        try:
            parse_route(route)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"router_routes[{index}] is invalid ({route!r}): {exc}") from exc


def validate_optional_hf_revision(field_name: str, revision: str | None) -> None:
    if revision is None:
        return
    if not isinstance(revision, str):
        raise TypeError(f"{field_name} must be a string or None")
    if not revision.strip():
        raise ValueError(f"{field_name} if set must be non-empty")
    if "\x00" in revision or "\n" in revision or "\r" in revision:
        raise ValueError(f"{field_name} contains invalid control characters")
    if path_has_parent_reference(revision) or revision.startswith("-"):
        raise ValueError(f"{field_name} must not contain '..' or start with '-' ({revision!r})")
    if not _HF_REVISION_RE.match(revision):
        raise ValueError(f"{field_name} contains unsupported characters ({revision!r})")


def validate_hf_allowed_models(models: tuple[str, ...]) -> None:
    if not isinstance(models, tuple):
        raise TypeError("router_hf_allowed_models must be a tuple of strings")
    for model in models:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("router_hf_allowed_models entries must be non-empty strings")
        if "\x00" in model or "\n" in model or "\r" in model or path_has_parent_reference(model):
            raise ValueError(f"Invalid router_hf_allowed_models entry: {model!r}")
