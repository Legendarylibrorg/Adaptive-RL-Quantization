from __future__ import annotations

from adaptive_quant.runner_cli import run_research_pipeline_cli
from config import CONFIG


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG,
        description="Offline research pipeline (simulator / python trainer by default). Linux: run from repo root.",
        config_help_suffix="Overrides `config.py` when set. See config.example.json.",
    )


if __name__ == "__main__":
    main()
