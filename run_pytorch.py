from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from config_4090 import CONFIG_4090
from config_4090_universal import CONFIG_4090_UNIVERSAL
from config_gpu import CONFIG_GPU


@dataclass(frozen=True)
class _PytorchPreset:
    config: FrameworkConfig
    requested_profile: str | None
    show_gpu_profile: bool = True
    show_training_host: bool = False
    show_target_hardware: bool = False


def _preset_map() -> dict[str, _PytorchPreset]:
    return {
        "gpu": _PytorchPreset(CONFIG_GPU, None),
        "4090": _PytorchPreset(CONFIG_4090, CONFIG_4090.torch_gpu_profile),
        "4090-universal": _PytorchPreset(
            CONFIG_4090_UNIVERSAL,
            CONFIG_4090_UNIVERSAL.torch_gpu_profile,
            show_gpu_profile=False,
            show_training_host=True,
            show_target_hardware=True,
        ),
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
        help=(
            "Used only when --config is omitted: gpu=auto VRAM profile; 4090=fixed RTX 4090 preset; "
            "4090-universal=multi-hardware policy trained on a 4090-class host (see config_4090_universal.py)."
        ),
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
        run_pipeline_entrypoint(config, requested_profile=requested, show_gpu_profile=True)
    else:
        preset = presets[args.preset]
        run_pipeline_entrypoint(
            preset.config,
            requested_profile=preset.requested_profile,
            show_gpu_profile=preset.show_gpu_profile,
            show_training_host=preset.show_training_host,
            show_target_hardware=preset.show_target_hardware,
        )


if __name__ == "__main__":
    main()
