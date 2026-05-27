"""Shared CLI helpers: ``--config`` / ``-c`` loading and research pipeline startup."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from typing import Any

from adaptive_quant.configuration import FrameworkConfig, RewardWeights
from adaptive_quant.configuration.flat_access import all_flat_config_keys
from adaptive_quant.configuration.sections import NESTED_SECTION_KEYS, SECTION_TYPES


_EXPLICIT_OVERRIDE_ARGS = {
    "training_episodes": "training_episodes",
    "evaluation_episodes": "evaluation_episodes",
    "benchmark_training_episodes": "benchmark_training_episodes",
    "benchmark_evaluation_episodes": "benchmark_evaluation_episodes",
    "run_name": "run_name",
    "seed": "seed",
}


def add_config_file_argument(
    parser: argparse.ArgumentParser,
    *,
    help_suffix: str = "",
) -> None:
    extra = f" {help_suffix}" if help_suffix else ""
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Load settings from a .json or .toml file (optional top-level 'preset' key).{extra}",
    )


def _parse_config_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE, for example training_episodes=500")
    key, value_text = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Override key must be non-empty")
    if "\x00" in key:
        raise argparse.ArgumentTypeError("Override key contains an invalid NUL byte")
    try:
        value: Any = json.loads(value_text)
    except json.JSONDecodeError:
        value = value_text
    return key, value


def add_config_override_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("startup config overrides")
    group.add_argument(
        "--training-episodes",
        type=int,
        default=None,
        help="Override training_episodes for this run.",
    )
    group.add_argument(
        "--evaluation-episodes",
        type=int,
        default=None,
        help="Override evaluation_episodes for this run.",
    )
    group.add_argument(
        "--benchmark-training-episodes",
        type=int,
        default=None,
        help="Override benchmark_training_episodes for this run.",
    )
    group.add_argument(
        "--benchmark-evaluation-episodes",
        type=int,
        default=None,
        help="Override benchmark_evaluation_episodes for this run.",
    )
    group.add_argument("--run-name", default=None, help="Override run_name for output artifacts.")
    group.add_argument("--seed", type=int, default=None, help="Override the top-level random seed.")
    group.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        type=_parse_config_override,
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override any FrameworkConfig key for this run. VALUE is parsed as JSON when possible; "
            "use flat keys like torch_batch_episodes=64 or dotted section keys like "
            "reward_weights.beta_throughput=0.08. May be passed more than once."
        ),
    )


def _merge_override(overrides: dict[str, Any], key: str, value: Any) -> None:
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


def apply_config_overrides(config: FrameworkConfig, args: argparse.Namespace) -> FrameworkConfig:
    overrides: dict[str, Any] = {}
    for arg_name, config_key in _EXPLICIT_OVERRIDE_ARGS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_key] = value
    try:
        for key, value in getattr(args, "config_overrides", None) or ():
            _merge_override(overrides, key, value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not overrides:
        return config

    from adaptive_quant.easy_config import config_from_dict

    try:
        return config_from_dict(overrides, base=config, strict=True)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def load_config_or_fallback(path: str | None, fallback: FrameworkConfig) -> FrameworkConfig:
    if path is None:
        return fallback
    from adaptive_quant.configuration.validation import validate_cli_path_argument

    validate_cli_path_argument("config", path)
    try:
        return FrameworkConfig.from_file(path)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


def run_research_pipeline_cli(
    *,
    fallback: FrameworkConfig,
    description: str,
    config_help_suffix: str = "",
) -> None:
    from adaptive_quant.research_pipeline import run_pipeline_entrypoint

    parser = argparse.ArgumentParser(description=description)
    add_config_file_argument(parser, help_suffix=config_help_suffix)
    add_config_override_arguments(parser)
    args = parser.parse_args()
    config = apply_config_overrides(load_config_or_fallback(args.config, fallback), args)
    run_pipeline_entrypoint(config)
