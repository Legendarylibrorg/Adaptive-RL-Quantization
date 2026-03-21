from __future__ import annotations

from config_4090_universal import CONFIG_4090_UNIVERSAL
from adaptive_quant.research_pipeline import ResearchPipeline


def main() -> None:
    summary = ResearchPipeline(CONFIG_4090_UNIVERSAL, requested_profile=CONFIG_4090_UNIVERSAL.torch_gpu_profile).run()
    print("Training host:", CONFIG_4090_UNIVERSAL.training_host_label)
    print("Target hardware modes:", ", ".join(CONFIG_4090_UNIVERSAL.hardware_modes))
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{CONFIG_4090_UNIVERSAL.benchmark_dir}/{CONFIG_4090_UNIVERSAL.run_name}_summary.json")


if __name__ == "__main__":
    main()
