"""Entrypoint: **MoE** research pipeline—packed expert variants, swap/cache/churn penalties in the reward.

Dense quantization work uses ``run_research.py``; this preset turns on ``moe_enabled`` and MoE benchmarks.
"""

from __future__ import annotations

from adaptive_quant.runner_cli import run_research_pipeline_cli
from config_moe import CONFIG_MOE


def main() -> None:
    run_research_pipeline_cli(
        fallback=CONFIG_MOE,
        description="MoE adaptive quantization: train -> evaluate -> MoE benchmarks -> analysis (see config_moe.py or --config).",
    )


if __name__ == "__main__":
    main()
