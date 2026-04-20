"""Shared CLI helpers: ``--config`` / ``-c`` loading and one-call **full pipeline** startup.

``run_research_pipeline_cli`` wires argparse to ``run_pipeline_entrypoint`` (train → eval → benchmarks → analysis).
"""

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
    """Return FrameworkConfig from file if path is set, else the module preset."""
    if path is None:
        return fallback
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
    """Parse argv for ``--config`` / ``-c``, merge into ``fallback``, then run the full research pipeline."""
    from adaptive_quant.research_pipeline import run_pipeline_entrypoint

    parser = argparse.ArgumentParser(description=description)
    add_config_file_argument(parser, help_suffix=config_help_suffix)
    args = parser.parse_args()
    run_pipeline_entrypoint(load_config_or_fallback(args.config, fallback))
