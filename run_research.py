"""Entrypoint: **offline research pipeline** with the stdlib trainer (simulator by default).

Set ``backend="llama_cpp"`` plus binary/model paths in ``config.py`` or ``--config`` to score real runs.
For CUDA policy training use ``run_pytorch.py`` instead.
"""

from __future__ import annotations

from adaptive_quant.runner_cli import run_research_pipeline_cli
from config import CONFIG


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG,
        description=(
            "Adaptive quantization RL: train → evaluate → benchmarks → analysis "
            "(Python trainer; simulator or llama.cpp per config). Run from repo root."
        ),
        config_help_suffix="Overrides `config.py` when set. See config.example.json.",
    )


if __name__ == "__main__":
    main()
