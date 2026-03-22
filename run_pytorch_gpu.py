from __future__ import annotations

from adaptive_quant.entrypoints import run_pipeline_entrypoint
from config_gpu import CONFIG_GPU


def main() -> None:
    run_pipeline_entrypoint(
        CONFIG_GPU,
        show_gpu_profile=True,
    )


if __name__ == "__main__":
    main()
