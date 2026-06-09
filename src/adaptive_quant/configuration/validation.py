from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_NAME_RE = _SAFE_ID_RE
_HF_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
# Hub repo ids: ``org/name`` (preferred) or a single legacy segment (e.g. ``gpt2``).
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_HF_LEGACY_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_HF_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")

_HF_ALLOWED_REPOS_ENV = "ADAPTIVE_RL_HF_ALLOWED_REPOS"
_HF_ALLOW_UNLISTED_ENV = "ADAPTIVE_RL_HF_ALLOW_UNLISTED"

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
MAX_RECOMMENDATION_EVAL_EPISODES = MAX_EPISODE_COUNT
MAX_RECOMMENDATION_CANDIDATE_LIMIT = 10_000
MAX_LLAMA_CPP_GENERATE_TOKENS = MAX_LLAMA_CPP_CONTEXT
MAX_JSONL_FLUSH_EVERY = MAX_LOG_EVERY_N_EPISODES
MAX_LLAMA_CPP_CACHE_ENTRIES = 65_536
# Router / online prompt text (UTF-8 code units) — limits tokenizer RAM blow-ups.
MAX_ROUTER_TASK_TEXT_CHARS = 262_144
MAX_ONLINE_PROMPT_TEXT_CHARS = MAX_ROUTER_TASK_TEXT_CHARS
MAX_ONLINE_REPLAY_ENTRIES_PER_PROMPT_HASH = 64
# CLI analysis scripts (log path + output dir).
MAX_CLI_PATH_CHARS = 4096

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
    if key in _BACKENDS:
        return
    from adaptive_quant.backends.registry import _EXTRA_BUILDERS

    if key in _EXTRA_BUILDERS:
        return
    allowed = sorted(_BACKENDS | set(_EXTRA_BUILDERS))
    raise ValueError(
        f"backend must be one of {allowed} (built-in or register_backend), got {name!r}"
    )


def _validate_choice(field_name: str, value: str, allowed: frozenset[str]) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    key = value.strip().lower()
    if key not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")


def validate_env_sampling_mode(name: str) -> None:
    _validate_choice("env_sampling_mode", name, _ENV_SAMPLING_MODES)


def validate_rl_train_policy_mode(name: str) -> None:
    _validate_choice("rl_train_policy_mode", name, _RL_TRAIN_POLICY_MODES)


def validate_stability_probe_sampling(name: str) -> None:
    _validate_choice("stability_probe_sampling", name, _STABILITY_PROBE_SAMPLING)


def validate_torch_policy_algorithm(name: str) -> None:
    _validate_choice("torch_policy_algorithm", name, _TORCH_POLICY_ALGORITHMS)


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
            raise ValueError(
                f"discrete_bit_widths must not contain duplicates (got {item!r} twice)"
            )
        seen.add(item)


def validate_run_name(run_name: str) -> None:
    if not isinstance(run_name, str):
        raise TypeError("run_name must be a string")
    if "/" in run_name or "\\" in run_name or "\x00" in run_name or ".." in run_name:
        raise ValueError(
            f"Invalid run_name {run_name!r}: must not contain path separators, NUL, or '..'"
        )
    if not _RUN_NAME_RE.match(run_name):
        raise ValueError(
            f"Invalid run_name {run_name!r}: expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/"
        )


def validate_safe_identifier(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"Invalid {field_name} {value!r}: expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/."
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


def validate_cli_path_argument(label: str, path: str) -> None:
    """Validate filesystem paths passed on the CLI (analysis tools, etc.)."""
    _validate_path_string(label, path)
    if len(path) > MAX_CLI_PATH_CHARS:
        raise ValueError(f"{label} exceeds {MAX_CLI_PATH_CHARS} characters")


def sanitize_user_text(text: str) -> str:
    """Normalize and strip invisible Unicode before feature extraction or subprocess use."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf" and ch != "\x00")
    return stripped.strip()


def validate_router_task_text(text: str) -> str:
    return _validate_bounded_user_text("task_text", text, max_chars=MAX_ROUTER_TASK_TEXT_CHARS)


def _validate_bounded_user_text(field_name: str, text: str, *, max_chars: int) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")
    if "\x00" in text:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    sanitized = sanitize_user_text(text)
    if len(sanitized) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters ({len(sanitized)} given)")
    return sanitized


def validate_online_prompt_text(text: str) -> str:
    return _validate_bounded_user_text("prompt_text", text, max_chars=MAX_ONLINE_PROMPT_TEXT_CHARS)


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
        resolved_binary == prefix or resolved_binary.startswith(prefix + os.sep)
        for prefix in prefixes
    ):
        raise ValueError(
            f"llama_cpp_binary resolves to {resolved_binary!r}, which is outside "
            f"{_LLAMA_CPP_BINARY_PREFIXES_ENV}={raw!r}"
        )


def validate_moe_topology(
    *, num_experts: int, top_k: int, gpu_resident: int, max_aggressive: int
) -> None:
    validate_bounded_positive_int("moe_num_experts", num_experts, ceiling=MAX_MOE_EXPERTS)
    validate_bounded_positive_int("moe_top_k", top_k, ceiling=MAX_MOE_TOP_K)
    if top_k > num_experts:
        raise ValueError(f"moe_top_k ({top_k}) must be <= moe_num_experts ({num_experts})")
    validate_bounded_positive_int("moe_gpu_resident_experts", gpu_resident, ceiling=MAX_MOE_EXPERTS)
    if gpu_resident > num_experts:
        raise ValueError(
            f"moe_gpu_resident_experts ({gpu_resident}) must be <= moe_num_experts ({num_experts})"
        )
    validate_bounded_positive_int(
        "moe_max_aggressive_experts", max_aggressive, ceiling=MAX_MOE_EXPERTS
    )
    if max_aggressive > num_experts:
        raise ValueError(
            f"moe_max_aggressive_experts ({max_aggressive}) must be <= moe_num_experts ({num_experts})"
        )


_STRUCTURAL_POSITIVE_LIMITS: tuple[tuple[str, int], ...] = (
    ("num_groups", MAX_NUM_GROUPS),
    ("num_layers", MAX_NUM_LAYERS),
    ("stability_probe_count", MAX_STABILITY_PROBE_COUNT),
    ("log_every_n_episodes", MAX_LOG_EVERY_N_EPISODES),
    ("llama_cpp_threads", MAX_LLAMA_CPP_THREADS),
    ("llama_cpp_context", MAX_LLAMA_CPP_CONTEXT),
    ("llama_cpp_max_prompt_chars", MAX_LLAMA_CPP_MAX_PROMPT_CHARS),
    ("torch_hidden_dim", MAX_TORCH_HIDDEN_DIM),
    ("torch_mlp_depth", MAX_TORCH_MLP_DEPTH),
    ("torch_batch_episodes", MAX_TORCH_BATCH_EPISODES),
    ("torch_minibatch_size", MAX_TORCH_MINIBATCH_SIZE),
    ("torch_update_epochs", MAX_TORCH_UPDATE_EPOCHS),
    ("torch_preflight_batch_size", MAX_TORCH_PREFLIGHT_BATCH_SIZE),
    ("torch_preflight_warmup_steps", MAX_TORCH_PREFLIGHT_STEPS),
    ("torch_preflight_steps", MAX_TORCH_PREFLIGHT_STEPS),
    ("online_min_replay_size", MAX_EPISODE_COUNT),
    ("online_update_interval", MAX_EPISODE_COUNT),
    ("online_batch_size", MAX_ONLINE_BATCH_SIZE),
    ("online_drift_window", MAX_EPISODE_COUNT),
    ("online_safe_mode_cooldown", MAX_EPISODE_COUNT),
)

_STRUCTURAL_NONNEG_LIMITS: tuple[tuple[str, int], ...] = (
    ("eval_interval", MAX_EPISODE_COUNT),
    ("checkpoint_interval", MAX_EPISODE_COUNT),
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
    values = locals()
    for name, ceiling in _STRUCTURAL_POSITIVE_LIMITS:
        validate_bounded_positive_int(name, values[name], ceiling=ceiling)
    for name, ceiling in _STRUCTURAL_NONNEG_LIMITS:
        validate_bounded_nonneg_int(name, values[name], ceiling=ceiling)


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


def validate_hf_model_id(
    field_name: str,
    model_id: str,
    *,
    require_hub_namespace: bool = False,
) -> None:
    """Validate a Hugging Face Hub model or repo identifier (no shell metacharacters)."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = model_id.strip()
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"{field_name} contains invalid control characters")
    if path_has_parent_reference(normalized):
        raise ValueError(f"{field_name} must not contain '..' ({normalized!r})")
    if normalized.startswith("-"):
        raise ValueError(f"{field_name} must not start with '-' ({normalized!r})")
    if "/" in normalized:
        if not _HF_REPO_ID_RE.match(normalized):
            raise ValueError(
                f"{field_name} {normalized!r} is invalid: expected '<org>/<name>' with "
                "alphanumeric, '.', '_', or '-' only."
            )
        return
    if require_hub_namespace:
        raise ValueError(
            f"{field_name} {normalized!r} must use '<org>/<name>' format for Hugging Face Hub."
        )
    if not _HF_LEGACY_MODEL_ID_RE.match(normalized):
        raise ValueError(
            f"{field_name} {normalized!r} is invalid: expected alphanumeric, '.', '_', or '-' only."
        )


def validate_hf_filename(field_name: str, filename: str) -> None:
    if not isinstance(filename, str) or not _HF_FILENAME_RE.match(filename):
        raise ValueError(
            f"{field_name} {filename!r} is invalid: expected alphanumeric, '.', '_', '-', or '/'."
        )
    if filename.startswith("-"):
        raise ValueError(f"{field_name} must not start with '-' ({filename!r})")
    if ".." in Path(filename).parts:
        raise ValueError(f"{field_name} must not contain '..' ({filename!r})")


def validate_hf_revision(field_name: str, revision: str) -> None:
    validate_optional_hf_revision(field_name, revision)


def hf_allow_unlisted_from_env() -> bool:
    raw = os.environ.get(_HF_ALLOW_UNLISTED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def hf_allowed_repos_from_env() -> frozenset[str]:
    raw = os.environ.get(_HF_ALLOWED_REPOS_ENV, "").strip()
    if not raw:
        return frozenset()
    repos: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        validate_hf_model_id(_HF_ALLOWED_REPOS_ENV, entry, require_hub_namespace=True)
        repos.add(entry)
    return frozenset(repos)


def assert_hf_repo_allowed(
    repo_id: str,
    *,
    field_name: str = "repo_id",
    config_allowlist: tuple[str, ...] = (),
) -> None:
    """Require ``repo_id`` on an allowlist unless ``ADAPTIVE_RL_HF_ALLOW_UNLISTED=1``."""
    if hf_allow_unlisted_from_env():
        return
    env_repos = hf_allowed_repos_from_env()
    combined = env_repos | frozenset(config_allowlist)
    if not combined:
        raise ValueError(
            f"Hugging Face repo downloads are denied by default. Set {_HF_ALLOWED_REPOS_ENV} "
            f"(comma-separated org/name ids) and/or route_hf_allowed_repos in config, or set "
            f"{_HF_ALLOW_UNLISTED_ENV}=1 only in trusted local environments."
        )
    normalized = repo_id.strip()
    if normalized not in combined:
        raise ValueError(
            f"{field_name} {normalized!r} is not in the Hugging Face repo allowlist "
            f"({sorted(combined)!r}). Set router_hf_allowed_models / route_hf_allowed_repos "
            f"in config or {_HF_ALLOWED_REPOS_ENV} (comma-separated org/name ids)."
        )


def validate_router_hf_settings(
    *,
    router_feature_backend: str,
    router_hf_embedding_model: str | None,
    router_hf_embedding_revision: str | None,
    router_hf_allowed_models: tuple[str, ...],
) -> None:
    """Harden HF embedding router: allowlist + pinned revision + vetted model id format."""
    backend = router_feature_backend.strip().lower()
    if backend != "hf":
        return
    if not router_hf_embedding_model or not str(router_hf_embedding_model).strip():
        raise ValueError(
            "router_feature_backend='hf' requires router_hf_embedding_model to be set."
        )
    if not router_hf_embedding_revision or not str(router_hf_embedding_revision).strip():
        raise ValueError(
            "router_feature_backend='hf' requires router_hf_embedding_revision "
            "(pin a commit hash or tag; do not leave revision unset)."
        )
    if not router_hf_allowed_models:
        raise ValueError(
            "router_feature_backend='hf' requires a non-empty router_hf_allowed_models "
            "allowlist; list every embedding model id you permit."
        )
    model_id = str(router_hf_embedding_model).strip()
    validate_hf_model_id("router_hf_embedding_model", model_id, require_hub_namespace=True)
    allowed = {entry.strip() for entry in router_hf_allowed_models}
    if model_id not in allowed:
        raise ValueError(
            f"router_hf_embedding_model {model_id!r} is not in router_hf_allowed_models."
        )


def validate_route_hf_allowed_repos(repos: tuple[str, ...]) -> None:
    if not isinstance(repos, tuple):
        raise TypeError("route_hf_allowed_repos must be a tuple of strings")
    for repo in repos:
        validate_hf_model_id("route_hf_allowed_repos", repo, require_hub_namespace=True)


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
        validate_hf_model_id("router_hf_allowed_models", model.strip(), require_hub_namespace=True)


def validate_gguf_export_settings(
    *,
    gguf_export_enabled: bool,
    gguf_export_source: str | None,
    gguf_export_quant_type: str,
    gguf_quantize_binary: str | None,
    llama_cpp_binary: str | None,
    llama_cpp_model: str | None,
) -> None:
    if not gguf_export_enabled:
        return
    from adaptive_quant.model_routes import QUANT_BITS

    if not isinstance(gguf_export_quant_type, str) or not gguf_export_quant_type.strip():
        raise ValueError(
            "llama_cpp_gguf_export_quant_type must be a non-empty string when export is enabled"
        )
    quant_key = gguf_export_quant_type.strip().upper()
    if quant_key not in QUANT_BITS:
        allowed = ", ".join(sorted(QUANT_BITS))
        raise ValueError(
            f"llama_cpp_gguf_export_quant_type must be one of [{allowed}], got {gguf_export_quant_type!r}"
        )
    source = gguf_export_source or llama_cpp_model
    if not source:
        raise ValueError(
            "llama_cpp_gguf_export_enabled requires llama_cpp_gguf_export_source or llama_cpp_model"
        )
    validate_optional_filesystem_path("llama_cpp_gguf_export_source", source)
    validate_optional_filesystem_path("llama_cpp_gguf_quantize_binary", gguf_quantize_binary)
    if gguf_quantize_binary is None and not llama_cpp_binary:
        raise ValueError(
            "llama_cpp_gguf_export_enabled requires llama_cpp_gguf_quantize_binary or llama_cpp_binary"
        )
