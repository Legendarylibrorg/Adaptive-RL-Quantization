"""Startup CLI override collection, validation, and privileged-key gating."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import fields
from typing import Any

from adaptive_quant.configuration import RewardWeights
from adaptive_quant.configuration.flat_access import all_flat_config_keys
from adaptive_quant.configuration.sections import NESTED_SECTION_KEYS, SECTION_TYPES
from adaptive_quant.logging_utils import safe_json_loads, to_jsonable

_ALLOW_PRIVILEGED_OVERRIDES_ENV = "ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES"
_MAX_OVERRIDE_KEY_CHARS = 128
_MAX_OVERRIDE_STRING_CHARS = 4096
_OVERRIDE_KEY_RE = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?$")

EXPLICIT_OVERRIDE_ARGS = {
    "training_episodes": "training_episodes",
    "evaluation_episodes": "evaluation_episodes",
    "benchmark_training_episodes": "benchmark_training_episodes",
    "benchmark_evaluation_episodes": "benchmark_evaluation_episodes",
    "run_name": "run_name",
    "seed": "seed",
}

_PRIVILEGED_FLAT_KEYS = frozenset(
    {
        "backend",
        "training_backend",
        "resume_from_checkpoint",
        "external_quality_path",
        "sim_calibration",
        "route_hf_allowed_repos",
        "router_enabled",
        "router_routes",
        "router_feature_backend",
        "router_hf_embedding_model",
        "router_hf_embedding_revision",
        "router_hf_local_files_only",
        "router_hf_allowed_models",
        "moe_enabled",
        "online_learning",
    }
)

_PRIVILEGED_SECTIONS = frozenset({"llama_cpp", "router", "moe"})

_PRIVILEGED_PREFIXES = ("llama_cpp_", "router_")


def allow_privileged_overrides_from_env() -> bool:
    raw = os.environ.get(_ALLOW_PRIVILEGED_OVERRIDES_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalize_override_key(raw_key: str) -> str:
    if not isinstance(raw_key, str):
        raise TypeError("Override key must be a string")
    if "\x00" in raw_key:
        raise ValueError("Override key contains an invalid NUL byte")
    key = unicodedata.normalize("NFKC", raw_key.strip())
    if not key:
        raise ValueError("Override key must be non-empty")
    if len(key) > _MAX_OVERRIDE_KEY_CHARS:
        raise ValueError(f"Override key exceeds {_MAX_OVERRIDE_KEY_CHARS} characters")
    if not key.isascii():
        raise ValueError(f"Override key must be ASCII: {key!r}")
    if not _OVERRIDE_KEY_RE.match(key):
        raise ValueError(
            f"Invalid override key {key!r}; expected flat or dotted names like "
            "training_episodes or reward_weights.beta_throughput"
        )
    return key


def override_key_is_privileged(key: str) -> bool:
    if key in _PRIVILEGED_FLAT_KEYS:
        return True
    if any(key.startswith(prefix) for prefix in _PRIVILEGED_PREFIXES):
        return True
    if key in _PRIVILEGED_SECTIONS:
        return True
    if "." in key:
        section, field = key.split(".", 1)
        if section in _PRIVILEGED_SECTIONS:
            return True
        if section == "training" and field == "resume_from_checkpoint":
            return True
    return False


def parse_override_value(value_text: str) -> Any:
    text = value_text.strip()
    if not text:
        return ""
    if text[0] in "{[\"0123456789-ntf":
        try:
            return safe_json_loads(text, label="CLI --set value")
        except json.JSONDecodeError:
            pass
    if len(text) > _MAX_OVERRIDE_STRING_CHARS:
        raise ValueError(
            f"CLI --set string value exceeds {_MAX_OVERRIDE_STRING_CHARS} characters"
        )
    return text


def parse_config_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE, for example training_episodes=500")
    raw_key, value_text = raw.split("=", 1)
    try:
        key = normalize_override_key(raw_key)
        value = parse_override_value(value_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return key, value


def merge_override(overrides: dict[str, Any], key: str, value: Any) -> None:
    flat_keys = all_flat_config_keys()
    if key in flat_keys or key in NESTED_SECTION_KEYS:
        overrides[key] = value
        return
    if "." not in key:
        raise ValueError(f"Unknown FrameworkConfig key: {key!r}")

    section_name, field_name = key.split(".", 1)
    if not section_name or not field_name or "." in field_name:
        raise ValueError(f"Invalid dotted override key: {key!r}")
    if section_name == "reward_weights":
        allowed = {field.name for field in fields(RewardWeights)}
    elif section_name in SECTION_TYPES:
        allowed = {field.name for field in fields(SECTION_TYPES[section_name])}
    else:
        raise ValueError(f"Unknown FrameworkConfig section: {section_name!r}")
    if field_name not in allowed:
        raise ValueError(f"Unknown {section_name} config key: {field_name!r}")
    section_overrides = overrides.setdefault(section_name, {})
    if not isinstance(section_overrides, dict):
        raise ValueError(f"Cannot mix whole-section and dotted overrides for {section_name!r}")
    section_overrides[field_name] = value


def collect_startup_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for arg_name, config_key in EXPLICIT_OVERRIDE_ARGS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_key] = value
    try:
        for key, value in getattr(args, "config_overrides", None) or ():
            merge_override(overrides, normalize_override_key(key), value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return overrides


def apply_startup_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return config
    from adaptive_quant.easy_config import config_from_dict

    try:
        return config_from_dict(overrides, base=config, strict=True)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def privileged_override_keys(overrides: dict[str, Any]) -> list[str]:
    flagged: list[str] = []
    for key, value in overrides.items():
        if override_key_is_privileged(key):
            flagged.append(key)
            continue
        if isinstance(value, dict):
            for field in value:
                dotted = f"{key}.{field}"
                if override_key_is_privileged(dotted):
                    flagged.append(dotted)
    return sorted(set(flagged))


def enforce_privileged_override_policy(overrides: dict[str, Any]) -> None:
    blocked = privileged_override_keys(overrides)
    if not blocked:
        return
    if allow_privileged_overrides_from_env():
        return
    keys = ", ".join(blocked)
    raise SystemExit(
        "Refusing privileged CLI startup overrides without "
        f"{_ALLOW_PRIVILEGED_OVERRIDES_ENV}=1: "
        f"{keys}. Use a reviewed --config file for backend, llama.cpp, router, "
        "checkpoint, or HF allowlist changes."
    )


def cli_overrides_audit_snapshot(overrides: dict[str, Any]) -> dict[str, Any]:
    snapshot = to_jsonable(overrides)
    assert isinstance(snapshot, dict)
    return snapshot


__all__ = [
    "EXPLICIT_OVERRIDE_ARGS",
    "allow_privileged_overrides_from_env",
    "apply_startup_overrides",
    "cli_overrides_audit_snapshot",
    "collect_startup_overrides",
    "enforce_privileged_override_policy",
    "merge_override",
    "normalize_override_key",
    "override_key_is_privileged",
    "parse_config_override",
    "privileged_override_keys",
]
