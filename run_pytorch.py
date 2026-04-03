from __future__ import annotations

import argparse
from typing import Iterable

from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from adaptive_quant.configuration import FrameworkConfig
from config_4090 import CONFIG_4090
from config_gpu import CONFIG_GPU


def _preset_map() -> dict[str, tuple[FrameworkConfig, str | None]]:
    return {
        "gpu": (CONFIG_GPU, None),
        "4090": (CONFIG_4090, CONFIG_4090.torch_gpu_profile),
    }


def main(argv: Iterable[str] | None = None) -> None:
    from adaptive_quant.runner_cli import add_config_file_argument, load_config_or_fallback

    presets = _preset_map()
    parser = argparse.ArgumentParser(
        description="CUDA research pipeline (PyTorch backend). Linux + NVIDIA recommended.",
        epilog="If --config is set, it replaces --preset entirely.",
    )
    add_config_file_argument(parser)
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()),
        default="gpu",
        help="Used only when --config is omitted: gpu=auto VRAM profile; 4090=fixed RTX 4090 preset.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.config is not None:
        config = load_config_or_fallback(args.config, CONFIG_GPU)
        requested = None
        if config.training_backend != "pytorch":
            raise SystemExit(
                "run_pytorch.py with --config requires training_backend='pytorch' in that file "
                f"(got {config.training_backend!r})."
            )
    else:
        config, requested = presets[args.preset]
    run_pipeline_entrypoint(config, requested_profile=requested, show_gpu_profile=True)


if __name__ == "__main__":
    main()
