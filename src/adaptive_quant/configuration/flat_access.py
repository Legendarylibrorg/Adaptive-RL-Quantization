"""Flat field access and serialization for nested :class:`FrameworkConfig` sections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from typing import Any

from adaptive_quant.configuration.sections import (
    FLAT_FIELD_MAP,
    NESTED_SECTION_KEYS,
    SECTION_TYPES,
    ArtifactPaths,
    LlamaCppSettings,
    MoESettings,
    OnlineSettings,
    RouterSettings,
    TorchSettings,
    TrainingSettings,
)

_REWARD_WEIGHTS_KEY = "reward_weights"

_TOP_LEVEL_FIELD_NAMES = frozenset(
    {
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
    }
)

_ALL_FLAT_CONFIG_KEYS = _TOP_LEVEL_FIELD_NAMES | set(FLAT_FIELD_MAP) | {_REWARD_WEIGHTS_KEY}


def all_flat_config_keys() -> frozenset[str]:
    return _ALL_FLAT_CONFIG_KEYS


def get_flat_field(config: object, flat_name: str) -> Any:
    if flat_name in _TOP_LEVEL_FIELD_NAMES:
        return getattr(config, flat_name)
    if flat_name == _REWARD_WEIGHTS_KEY:
        return getattr(config, _REWARD_WEIGHTS_KEY)
    section_name, field_name = FLAT_FIELD_MAP[flat_name]
    return getattr(getattr(config, section_name), field_name)


def set_flat_field(config: object, flat_name: str, value: Any) -> None:
    if flat_name in _TOP_LEVEL_FIELD_NAMES:
        object.__setattr__(config, flat_name, value)
        return
    if flat_name == _REWARD_WEIGHTS_KEY:
        object.__setattr__(config, flat_name, value)
        return
    section_name, field_name = FLAT_FIELD_MAP[flat_name]
    section = getattr(config, section_name)
    setattr(section, field_name, value)


def apply_flat_kwargs(config: object, kwargs: Mapping[str, Any]) -> None:
    for key, value in kwargs.items():
        if key in NESTED_SECTION_KEYS:
            _merge_section(config, key, value)
        elif key in FLAT_FIELD_MAP or key in _TOP_LEVEL_FIELD_NAMES or key == _REWARD_WEIGHTS_KEY:
            set_flat_field(config, key, value)
        else:
            raise TypeError(f"Unexpected FrameworkConfig keyword argument: {key!r}")


def _merge_section(config: object, section_name: str, value: Any) -> None:
    if section_name == _REWARD_WEIGHTS_KEY:
        from adaptive_quant.configuration.framework import RewardWeights

        current = getattr(config, _REWARD_WEIGHTS_KEY)
        if isinstance(value, RewardWeights):
            object.__setattr__(config, _REWARD_WEIGHTS_KEY, value)
            return
        if not isinstance(value, Mapping):
            raise TypeError("reward_weights must be a mapping or RewardWeights")
        object.__setattr__(
            config,
            _REWARD_WEIGHTS_KEY,
            replace(current, **{k: v for k, v in dict(value).items()}),
        )
        return
    section_type = SECTION_TYPES[section_name]
    current = getattr(config, section_name)
    if isinstance(value, section_type):
        object.__setattr__(config, section_name, value)
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{section_name} must be a mapping or {section_type.__name__}")
    allowed = {f.name for f in fields(section_type)}
    updates = {k: v for k, v in dict(value).items() if k in allowed}
    object.__setattr__(config, section_name, replace(current, **updates))


def config_to_flat_dict(config: object) -> dict[str, Any]:
    """Flatten nested sections for JSON summaries (backward-compatible flat ``config`` keys)."""
    out: dict[str, Any] = {}
    for name in sorted(_TOP_LEVEL_FIELD_NAMES):
        out[name] = getattr(config, name)
    for flat_name in sorted(FLAT_FIELD_MAP):
        out[flat_name] = get_flat_field(config, flat_name)
    out[_REWARD_WEIGHTS_KEY] = asdict(getattr(config, _REWARD_WEIGHTS_KEY))
    return out


__all__ = [
    "ArtifactPaths",
    "LlamaCppSettings",
    "MoESettings",
    "OnlineSettings",
    "RouterSettings",
    "TorchSettings",
    "TrainingSettings",
    "all_flat_config_keys",
    "apply_flat_kwargs",
    "config_to_flat_dict",
    "get_flat_field",
    "set_flat_field",
]
