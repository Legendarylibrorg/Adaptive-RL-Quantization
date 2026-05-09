from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from adaptive_quant.backend import (
    SimulatorBackend,
    require_llama_cpp_paths,
    run_llama_cpp_measurement,
)
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import write_json
from adaptive_quant.math_utils import ratio_mean
from adaptive_quant.cli.common import add_config_file_argument, load_config_or_fallback
from adaptive_quant.types import HardwareType, QuantizationDecision, QuantMode


def _build_calibration_config(base_cfg: FrameworkConfig, seed: int) -> FrameworkConfig:
    return base_cfg.clone(backend="llama_cpp", prompt_split_enabled=False, seed=seed)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit simulator calibration multipliers from llama.cpp measurements.")
    add_config_file_argument(parser, help_suffix="Otherwise uses config.py as base.")
    parser.add_argument("--run-name", default="llama_cpp_calibration", help="Output artifact run name prefix.")
    parser.add_argument("--prompts", default="6", help="Number of random prompts to sample for calibration.")
    parser.add_argument("--seed", default="1234", help="RNG seed for prompt sampling.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from adaptive_quant.presets.baseline import CONFIG as BASE

    # Calibration must run with a real llama.cpp binary + model.
    base_cfg = load_config_or_fallback(args.config, BASE)
    config = _build_calibration_config(base_cfg, int(args.seed))
    try:
        llama_cpp_binary, llama_cpp_model = require_llama_cpp_paths(config)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    prompt_count = max(1, int(args.prompts))

    env = AdaptiveQuantizationEnv(config, log_path=f"{config.log_dir}/{args.run_name}_calibration.jsonl")
    sim_backend = SimulatorBackend(config.clone(backend="simulator"))

    # Use a fixed decision so we calibrate the raw backend response, not policy behavior.
    fixed_decision = QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=config.safe_default_bits)

    by_hw: dict[str, dict[str, Any]] = {}
    for hw in (HardwareType.GPU, HardwareType.CPU, HardwareType.LOW_RESOURCE):
        observed_latency: list[float] = []
        sim_latency: list[float] = []
        observed_throughput: list[float] = []
        sim_throughput: list[float] = []
        observed_memory: list[float] = []
        sim_memory: list[float] = []

        for episode_i in range(prompt_count):
            state = env.reset(forced_hardware=hw, phase="eval", episode_index=episode_i)
            parsed = run_llama_cpp_measurement(
                config,
                llama_cpp_binary=llama_cpp_binary,
                llama_cpp_model=llama_cpp_model,
                prompt_text=state.prompt.text,
                ngl=state.hardware_profile.ngl,
            )
            sim_metrics = sim_backend.evaluate(state, fixed_decision)
            # Convert "ms per token" to "total latency" consistent with env metric definition.
            if "latency_ms_per_token" in parsed:
                observed_latency.append(float(parsed["latency_ms_per_token"]) * max(1, state.input_features.prompt_length))
                sim_latency.append(float(sim_metrics["latency_ms"]))
            if "throughput_tps" in parsed:
                observed_throughput.append(float(parsed["throughput_tps"]))
                sim_throughput.append(float(sim_metrics["throughput_tps"]))
            if "memory_mb" in parsed:
                observed_memory.append(float(parsed["memory_mb"]))
                sim_memory.append(float(sim_metrics["memory_mb"]))

        by_hw[hw.value] = {
            "samples": prompt_count,
            "observed": {
                "latency_ms": observed_latency,
                "throughput_tps": observed_throughput,
                "memory_mb": observed_memory,
            },
            "simulated": {
                "latency_ms": sim_latency,
                "throughput_tps": sim_throughput,
                "memory_mb": sim_memory,
            },
            "fit": {
                "latency_multiplier": ratio_mean(observed_latency, sim_latency) if observed_latency else 1.0,
                "throughput_multiplier": ratio_mean(observed_throughput, sim_throughput) if observed_throughput else 1.0,
                "memory_multiplier": ratio_mean(observed_memory, sim_memory) if observed_memory else 1.0,
            },
        }

    calibration = {hw: stats["fit"] for hw, stats in by_hw.items()}
    output = {
        "run_name": args.run_name,
        "llama_cpp_binary": config.llama_cpp_binary,
        "llama_cpp_model": str(Path(config.llama_cpp_model).name),
        "llama_cpp_threads": config.llama_cpp_threads,
        "llama_cpp_context": config.llama_cpp_context,
        "prompt_samples_per_hardware": prompt_count,
        "calibration": calibration,
        "details": by_hw,
        "how_to_apply": {
            "config_field": "sim_calibration",
            "example": calibration,
        },
    }
    out_path = f"{config.benchmark_dir}/{args.run_name}_calibration.json"
    write_json(out_path, output)
    from adaptive_quant.run_footer import print_calibration_footer

    print_calibration_footer(run_name=args.run_name, out_path=out_path, calibration=calibration)


if __name__ == "__main__":
    main()
