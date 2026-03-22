from __future__ import annotations

from adaptive_quant.entrypoints import run_pipeline_entrypoint
from config import CONFIG


def main() -> None:
    run_pipeline_entrypoint(CONFIG)


if __name__ == "__main__":
    main()
