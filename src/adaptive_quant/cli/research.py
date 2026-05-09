"""CLI: offline research pipeline (stdlib trainer; simulator or llama.cpp per config)."""

from __future__ import annotations

from adaptive_quant.cli.common import run_research_pipeline_cli
from adaptive_quant.presets.baseline import CONFIG


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG,
        description=(
            "Adaptive quantization RL: train -> evaluate -> benchmarks -> analysis "
            "(Python trainer; simulator or llama.cpp per config). Run from repo root."
        ),
        config_help_suffix="Overrides `config.py` when set. See config.example.json.",
    )


if __name__ == "__main__":
    main()
