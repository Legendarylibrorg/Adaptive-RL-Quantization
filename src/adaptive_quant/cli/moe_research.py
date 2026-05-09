"""CLI: MoE research pipeline."""

from __future__ import annotations

from adaptive_quant.cli.common import run_research_pipeline_cli
from adaptive_quant.presets.moe import CONFIG_MOE


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG_MOE,
        description="MoE adaptive quantization: train -> evaluate -> MoE benchmarks -> analysis (see config_moe.py or --config).",
    )


if __name__ == "__main__":
    main()
