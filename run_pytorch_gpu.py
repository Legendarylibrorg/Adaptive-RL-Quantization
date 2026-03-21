from __future__ import annotations

from config_gpu import CONFIG_GPU
from adaptive_quant.research_pipeline import ResearchPipeline


def main() -> None:
    summary = ResearchPipeline(CONFIG_GPU).run()
    print("GPU profile:", summary["gpu_profile"])
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{CONFIG_GPU.benchmark_dir}/{CONFIG_GPU.run_name}_summary.json")


if __name__ == "__main__":
    main()
