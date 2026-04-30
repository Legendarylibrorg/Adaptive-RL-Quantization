from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig, RewardWeights
from adaptive_quant.logging_utils import enforce_local_read_limit

_FRAMEWORK_FIELD_NAMES = {f.name for f in fields(FrameworkConfig)}
_REWARD_FIELD_NAMES = {f.name for f in fields(RewardWeights)}
_TUPLE_STRING_FIELDS = frozenset({"hardware_modes", "moe_variant_names", "router_hf_allowed_models"})
_TUPLE_INT_FIELDS = frozenset({"discrete_bit_widths"})
_TUPLE_FLOAT_FIELDS = frozenset({"scale_bounds", "clip_bounds", "precision_bounds"})


def named_preset(name: str) -> FrameworkConfig:
    """Builtin short-hands for layered config files (see ``preset`` key in JSON/TOML)."""
    key = name.strip().lower().replace("-", "_")
    if key in ("", "default", "baseline"):
        return FrameworkConfig()
    if key in ("repro", "reproducible", "reproducible_research"):
        return FrameworkConfig.reproducible_research()
    if key in ("gpu", "pytorch", "torch"):
        return FrameworkConfig(training_backend="pytorch", torch_gpu_profile="auto")
    if key in ("minimal", "fast"):
        return FrameworkConfig(
            training_episodes=256,
            evaluation_episodes=64,
            stability_probe_count=1,
            replay_buffer_capacity=0,
            torch_preflight=False,
            run_name="minimal_run",
        )
    raise ValueError(
        f"Unknown config preset {name!r}. "
        f"Use: default, reproducible, pytorch, minimal (or pass no preset)."
    )


def config_from_dict(
    data: Mapping[str, Any],
    *,
    base: FrameworkConfig | None = None,
    strict: bool = False,
) -> FrameworkConfig:
    """
    Build a FrameworkConfig from a plain mapping (e.g. JSON/TOML).

    Lists are coerced to tuples where needed. Nested ``reward_weights`` is merged
    onto the base. Unknown top-level keys are ignored (warning) unless ``strict=True``.
    """
    base_obj = base or FrameworkConfig()
    d = dict(data)
    rw_raw = d.pop("reward_weights", None)

    if strict:
        bad_fw = set(d) - _FRAMEWORK_FIELD_NAMES
        if bad_fw:
            raise ValueError(f"Unknown FrameworkConfig keys: {sorted(bad_fw)}")
        if rw_raw is not None and isinstance(rw_raw, dict):
            bad_rw = set(rw_raw) - _REWARD_FIELD_NAMES
            if bad_rw:
                raise ValueError(f"Unknown RewardWeights keys: {sorted(bad_rw)}")
    else:
        bad_fw = set(d) - _FRAMEWORK_FIELD_NAMES
        if bad_fw:
            warnings.warn(
                f"Ignoring unknown FrameworkConfig keys: {sorted(bad_fw)}",
                UserWarning,
                stacklevel=2,
            )
        if rw_raw is not None and isinstance(rw_raw, dict):
            bad_rw = set(rw_raw) - _REWARD_FIELD_NAMES
            if bad_rw:
                warnings.warn(
                    f"Ignoring unknown RewardWeights keys: {sorted(bad_rw)}",
                    UserWarning,
                    stacklevel=2,
                )

    coerced: dict[str, Any] = {}
    for k, v in d.items():
        if k not in _FRAMEWORK_FIELD_NAMES:
            continue
        if k in _TUPLE_STRING_FIELDS and isinstance(v, list):
            v = tuple(str(x) for x in v)
        elif k in _TUPLE_INT_FIELDS and isinstance(v, list):
            v = tuple(int(x) for x in v)
        elif k in _TUPLE_FLOAT_FIELDS and isinstance(v, list):
            v = tuple(float(x) for x in v)
        coerced[k] = v

    new_rw = base_obj.reward_weights
    if rw_raw is not None:
        if not isinstance(rw_raw, Mapping):
            raise TypeError("reward_weights must be a mapping")
        rw_dict = {k: v for k, v in dict(rw_raw).items() if k in _REWARD_FIELD_NAMES}
        new_rw = replace(base_obj.reward_weights, **rw_dict)

    return replace(base_obj, reward_weights=new_rw, **coerced)


def quick_config(**kwargs: Any) -> FrameworkConfig:
    """Shorthand: ``quick_config(run_name=\"x\", training_episodes=100)`` with defaults for omitted fields."""
    return config_from_dict(kwargs)


def load_config(path: str | Path, *, strict: bool = True) -> FrameworkConfig:
    """
    Load ``.json`` or ``.toml`` into FrameworkConfig.

    Optional top-level string key ``preset`` selects a base profile
    (``default``, ``reproducible``, ``pytorch``, ``minimal``); remaining keys override it.
    File-backed config loading is strict by default so typos fail fast.
    """
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Config file not found: {raw_path}")
    data = _parse_config_file(raw_path)
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be an object/dict, got {type(data).__name__}")
    payload = dict(data)
    preset_name = payload.pop("preset", None)
    base = named_preset(preset_name) if preset_name is not None else FrameworkConfig()
    return config_from_dict(payload, base=base, strict=strict)


def _parse_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    enforce_local_read_limit(path, label="Config file")
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".toml", ".tml"):
        import tomllib

        return tomllib.loads(text)
    raise ValueError(f"Unsupported config extension {suffix!r} (use .json or .toml)")
