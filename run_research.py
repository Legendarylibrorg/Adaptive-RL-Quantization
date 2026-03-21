from __future__ import annotations

from config import CONFIG
from adaptive_quant.research_pipeline import ResearchPipeline


def main() -> None:
    summary = ResearchPipeline(CONFIG).run()
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{CONFIG.benchmark_dir}/{CONFIG.run_name}_summary.json")


if __name__ == "__main__":
    main()
