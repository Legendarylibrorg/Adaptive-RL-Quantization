"""Shared CLI helpers: ``--config`` / ``-c`` loading and research pipeline startup."""

from __future__ import annotations

import argparse
from typing import Any

from adaptive_quant.cli.startup_overrides import (
    apply_startup_overrides,
    cli_overrides_audit_snapshot,
    collect_startup_overrides,
    enforce_privileged_override_policy,
    parse_config_override,
)
from adaptive_quant.configuration import FrameworkConfig


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
        type=parse_config_override,
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a FrameworkConfig field for this run. VALUE is parsed as bounded JSON when "
            "possible; use flat keys like torch_batch_episodes=64 or dotted section keys like "
            "reward_weights.beta_throughput=0.08. Backend, llama.cpp, router, checkpoint, and HF "
            "allowlist keys require ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES=1. May be repeated."
        ),
    )


def resolve_startup_config(
    config: FrameworkConfig,
    args: argparse.Namespace,
) -> tuple[FrameworkConfig, dict[str, Any] | None]:
    overrides = collect_startup_overrides(args)
    enforce_privileged_override_policy(overrides)
    resolved = apply_startup_overrides(config, overrides)
    audit = cli_overrides_audit_snapshot(overrides) if overrides else None
    return resolved, audit


def apply_config_overrides(config: FrameworkConfig, args: argparse.Namespace) -> FrameworkConfig:
    resolved, _audit = resolve_startup_config(config, args)
    return resolved


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
    config, cli_overrides = resolve_startup_config(
        load_config_or_fallback(args.config, fallback), args
    )
    run_pipeline_entrypoint(config, cli_startup_overrides=cli_overrides)
