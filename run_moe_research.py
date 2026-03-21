from __future__ import annotations

from config_moe import CONFIG_MOE
from adaptive_quant.research_pipeline import ResearchPipeline


def main() -> None:
    summary = ResearchPipeline(CONFIG_MOE).run()
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{CONFIG_MOE.benchmark_dir}/{CONFIG_MOE.run_name}_summary.json")


if __name__ == "__main__":
    main()
