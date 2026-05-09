from __future__ import annotations

import re
from pathlib import Path

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HF_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

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
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


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
    stripped = path.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"{field_name} contains invalid control characters")
    if path_has_parent_reference(path):
        raise ValueError(f"{field_name} must not contain '..' ({path!r})")


def validate_optional_filesystem_path(field_name: str, path: str | None) -> None:
    if path is None:
        return
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string or None")
    stripped = path.strip()
    if not stripped:
        raise ValueError(f"{field_name} if set must be non-empty")
    if "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError(f"{field_name} contains invalid control characters")
    if path_has_parent_reference(path):
        raise ValueError(f"{field_name} must not contain '..' ({path!r})")


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
