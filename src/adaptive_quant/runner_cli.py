"""Backward-compatible re-export; prefer ``adaptive_quant.cli.common``."""

from adaptive_quant.cli.common import (
    add_config_file_argument,
    load_config_or_fallback,
    run_research_pipeline_cli,
)

__all__ = [
    "add_config_file_argument",
    "load_config_or_fallback",
    "run_research_pipeline_cli",
]
