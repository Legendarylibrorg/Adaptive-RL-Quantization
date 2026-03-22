from __future__ import annotations

from adaptive_quant.entrypoints import run_pipeline_entrypoint
from config_moe import CONFIG_MOE


def main() -> None:
    run_pipeline_entrypoint(CONFIG_MOE)


if __name__ == "__main__":
    main()
