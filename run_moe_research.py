from __future__ import annotations

from adaptive_quant.runner_cli import run_research_pipeline_cli
from config_moe import CONFIG_MOE


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG_MOE,
        description="MoE research pipeline (see config_moe.py or --config).",
    )


if __name__ == "__main__":
    main()
