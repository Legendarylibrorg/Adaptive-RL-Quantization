from __future__ import annotations

import argparse

from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from adaptive_quant.runner_cli import add_config_file_argument, load_config_or_fallback
from config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline research pipeline (simulator / python trainer by default). Linux: run from repo root."
    )
    add_config_file_argument(parser, help_suffix='Overrides `config.py` when set. See config.example.json.')
    args = parser.parse_args()
    run_pipeline_entrypoint(load_config_or_fallback(args.config, CONFIG))


if __name__ == "__main__":
    main()
