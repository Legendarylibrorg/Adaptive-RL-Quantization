from __future__ import annotations

from config_4090 import CONFIG_4090
from adaptive_quant.research_pipeline import ResearchPipeline


def main() -> None:
    summary = ResearchPipeline(CONFIG_4090, requested_profile=CONFIG_4090.torch_gpu_profile).run()
    print("GPU profile:", summary["gpu_profile"])
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{CONFIG_4090.benchmark_dir}/{CONFIG_4090.run_name}_summary.json")


if __name__ == "__main__":
    main()
