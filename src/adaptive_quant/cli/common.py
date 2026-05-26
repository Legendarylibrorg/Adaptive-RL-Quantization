"""Shared CLI helpers: ``--config`` / ``-c`` loading and research pipeline startup."""

from __future__ import annotations

import argparse

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
    args = parser.parse_args()
    run_pipeline_entrypoint(load_config_or_fallback(args.config, fallback))
