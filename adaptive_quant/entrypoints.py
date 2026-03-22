from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.research_pipeline import ResearchPipeline


def run_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    requested_profile: str | None = None,
    show_gpu_profile: bool = False,
    show_training_host: bool = False,
    show_target_hardware: bool = False,
) -> dict[str, object]:
    summary = ResearchPipeline(config, requested_profile=requested_profile).run()
    if show_training_host and config.training_host_label:
        print("Training host:", config.training_host_label)
    if show_target_hardware:
        print("Target hardware modes:", ", ".join(config.hardware_modes))
    if show_gpu_profile:
        print("GPU profile:", summary["gpu_profile"])
    print("Training summary:", summary["train"])
    print("Evaluation summary:", summary["evaluation"])
    print("Benchmark summary written to:", f"{config.benchmark_dir}/{config.run_name}_summary.json")
    return summary
