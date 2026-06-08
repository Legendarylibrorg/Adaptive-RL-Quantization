from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig, RewardWeights
from adaptive_quant.configuration.flat_access import all_flat_config_keys, apply_flat_kwargs
from adaptive_quant.configuration.sections import NESTED_SECTION_KEYS, SECTION_TYPES
from adaptive_quant.logging_utils import (
    enforce_local_read_limit,
    enforce_safe_parsed_json,
    safe_json_loads,
)

_FRAMEWORK_FIELD_NAMES = all_flat_config_keys() | NESTED_SECTION_KEYS
_REWARD_FIELD_NAMES = {f.name for f in fields(RewardWeights)}
_TUPLE_STRING_FIELDS = frozenset(
    {
        "hardware_modes",
        "moe_variant_names",
        "router_hf_allowed_models",
        "route_hf_allowed_repos",
        "router_routes",
    }
)
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
        return FrameworkConfig(
            training_backend="pytorch",
            torch_gpu_profile="auto",
            torch_require_cuda=True,
        )
    if key in ("post_train", "posttrain", "llm_post_train"):
        from adaptive_quant.presets.post_train import CONFIG_POST_TRAIN

        return CONFIG_POST_TRAIN.clone()
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
        f"Use: default, reproducible, pytorch, post_train, minimal (or pass no preset)."
    )


def config_from_dict(
    data: Mapping[str, Any],
    *,
    base: FrameworkConfig | None = None,
    strict: bool = False,
) -> FrameworkConfig:
    """
    Build a FrameworkConfig from a plain mapping (e.g. JSON/TOML).

    Lists are coerced to tuples where needed. Nested ``reward_weights`` or section keys
    (``moe``, ``torch``, ``llama_cpp``, …) are merged onto the base. Unknown top-level keys
    are ignored (warning) unless ``strict=True``.
    """
    base_obj = base.clone() if base is not None else FrameworkConfig()
    d = dict(data)
    rw_raw = d.pop("reward_weights", None)
    nested_raw = {
        k: d.pop(k) for k in list(d) if k in NESTED_SECTION_KEYS and k != "reward_weights"
    }

    if strict:
        bad_fw = set(d) - _FRAMEWORK_FIELD_NAMES
        if bad_fw:
            raise ValueError(f"Unknown FrameworkConfig keys: {sorted(bad_fw)}")
        if rw_raw is not None and isinstance(rw_raw, dict):
            bad_rw = set(rw_raw) - _REWARD_FIELD_NAMES
            if bad_rw:
                raise ValueError(f"Unknown RewardWeights keys: {sorted(bad_rw)}")
        _validate_nested_sections_strict(nested_raw)
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

    if rw_raw is not None:
        if not isinstance(rw_raw, Mapping):
            raise TypeError("reward_weights must be a mapping")
        rw_dict = {k: v for k, v in dict(rw_raw).items() if k in _REWARD_FIELD_NAMES}
        coerced["reward_weights"] = replace(base_obj.reward_weights, **rw_dict)

    apply_flat_kwargs(base_obj, coerced)
    for section_key, section_value in nested_raw.items():
        apply_flat_kwargs(base_obj, {section_key: section_value})
    base_obj.__post_init__()
    return base_obj


def _validate_nested_sections_strict(nested_raw: Mapping[str, Any]) -> None:
    for section_key, section_value in nested_raw.items():
        section_type = SECTION_TYPES[section_key]
        if isinstance(section_value, section_type):
            continue
        if not isinstance(section_value, Mapping):
            raise TypeError(f"{section_key} must be a mapping or {section_type.__name__}")
        allowed = {f.name for f in fields(section_type)}
        bad = set(section_value) - allowed
        if bad:
            raise ValueError(f"Unknown {section_type.__name__} keys: {sorted(bad)}")


def quick_config(**kwargs: Any) -> FrameworkConfig:
    """Shorthand: ``quick_config(run_name=\"x\", training_episodes=100)`` with defaults for omitted fields."""
    return config_from_dict(kwargs)


def load_config(path: str | Path, *, strict: bool = True) -> FrameworkConfig:
    """
    Load ``.json`` or ``.toml`` into FrameworkConfig.

    Optional top-level string key ``preset`` selects a base profile
    (``default``, ``reproducible``, ``pytorch``, ``post_train``, ``minimal``); remaining keys override it.
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
    cfg_label = f"Config file {path}"
    if suffix == ".json":
        data = safe_json_loads(text, label=cfg_label)
        if not isinstance(data, dict):
            raise TypeError(f"Config root must be an object/dict, got {type(data).__name__}")
        return data
    if suffix in (".toml", ".tml"):
        from tomllib import loads as toml_loads

        data = toml_loads(text)
        enforce_safe_parsed_json(data, label=cfg_label)
        if not isinstance(data, dict):
            raise TypeError(f"Config root must be an object/dict, got {type(data).__name__}")
        return data
    raise ValueError(f"Unsupported config extension {suffix!r} (use .json or .toml)")
