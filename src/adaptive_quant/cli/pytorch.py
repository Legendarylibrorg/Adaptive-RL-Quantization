"""CLI: research pipeline with ``training_backend=pytorch``."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass

from adaptive_quant.cli.common import (
    add_config_file_argument,
    add_config_override_arguments,
    load_config_or_fallback,
    resolve_startup_config,
)
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.presets.gpu import CONFIG_GPU
from adaptive_quant.presets.rtx3090 import CONFIG_3090
from adaptive_quant.presets.rtx4090 import CONFIG_4090
from adaptive_quant.presets.rtx4090_universal import CONFIG_4090_UNIVERSAL
from adaptive_quant.research_pipeline import run_pipeline_entrypoint


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
        "3090": _PytorchPreset(CONFIG_3090, CONFIG_3090.torch_gpu_profile),
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
    presets = _preset_map()
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive quantization RL on CUDA: full pipeline with PyTorch PPO/VPG/AWR-style trainer. "
            "Linux + NVIDIA recommended."
        ),
        epilog="If --config is set, it replaces --preset entirely.",
    )
    add_config_file_argument(parser)
    add_config_override_arguments(parser)
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()),
        default="gpu",
        help=(
            "Used when --config is omitted: gpu=auto-detected VRAM profile; 3090/4090=fixed host presets; "
            "4090-universal=multi-hardware policy on a 4090-class host (see config.CONFIG_4090_UNIVERSAL)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.config is not None:
        config, cli_overrides = resolve_startup_config(
            load_config_or_fallback(args.config, CONFIG_GPU),
            args,
        )
        requested = None
        if config.training_backend != "pytorch":
            raise SystemExit(
                "run_pytorch.py with --config requires training_backend='pytorch' in that file "
                f"(got {config.training_backend!r})."
            )
        run_pipeline_entrypoint(
            config,
            requested_profile=requested,
            show_gpu_profile=True,
            cli_startup_overrides=cli_overrides,
        )
    else:
        preset = presets[args.preset]
        config, cli_overrides = resolve_startup_config(preset.config, args)
        run_pipeline_entrypoint(
            config,
            requested_profile=preset.requested_profile,
            show_gpu_profile=preset.show_gpu_profile,
            show_training_host=preset.show_training_host,
            show_target_hardware=preset.show_target_hardware,
            cli_startup_overrides=cli_overrides,
        )


if __name__ == "__main__":
    main()
