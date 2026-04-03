from __future__ import annotations

from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from config_4090_universal import CONFIG_4090_UNIVERSAL


def main() -> None:
    run_pipeline_entrypoint(
        CONFIG_4090_UNIVERSAL,
        requested_profile=CONFIG_4090_UNIVERSAL.torch_gpu_profile,
        show_training_host=True,
        show_target_hardware=True,
    )


if __name__ == "__main__":
    main()
