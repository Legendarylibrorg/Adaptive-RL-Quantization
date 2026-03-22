from __future__ import annotations

from adaptive_quant.entrypoints import run_pipeline_entrypoint
from config_4090 import CONFIG_4090


def main() -> None:
    run_pipeline_entrypoint(
        CONFIG_4090,
        requested_profile=CONFIG_4090.torch_gpu_profile,
        show_gpu_profile=True,
    )


if __name__ == "__main__":
    main()
