from __future__ import annotations

import argparse

from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from adaptive_quant.runner_cli import add_config_file_argument, load_config_or_fallback
from config_moe import CONFIG_MOE


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE research pipeline (see config_moe.py or --config).")
    add_config_file_argument(parser)
    args = parser.parse_args()
    run_pipeline_entrypoint(load_config_or_fallback(args.config, CONFIG_MOE))


if __name__ == "__main__":
    main()
