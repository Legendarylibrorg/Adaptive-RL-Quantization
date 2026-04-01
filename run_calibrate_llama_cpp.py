from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable

from adaptive_quant.backend import SimulatorBackend, _extract_numeric, require_llama_cpp_paths
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import write_json
from adaptive_quant.types import HardwareType, QuantMode, QuantizationDecision


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_ratio(numer: float, denom: float) -> float | None:
    if not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return None
    return numer / denom


def _run_llama_cpp_once(
    config: FrameworkConfig,
    *,
    prompt_text: str,
    ngl: int,
) -> tuple[float | None, float | None, float | None]:
    """
    Returns: (latency_ms_per_token, throughput_tps, memory_mb) where memory_mb is best-effort.
    """
    command = [
        config.llama_cpp_binary or "",
        "-m",
        config.llama_cpp_model or "",
        "-p",
        prompt_text,
        "-ngl",
        str(ngl),
        "-t",
        str(config.llama_cpp_threads),
        "-c",
        str(config.llama_cpp_context),
        "-n",
        "64",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(getattr(config, "llama_cpp_timeout_s", 30.0)),
    )
    stdout = (completed.stdout or "").lower()
    stderr = (completed.stderr or "").lower()
    combined = stdout + "\n" + stderr

    throughput_tps = _extract_numeric(combined, "tok/s", default=0.0)
    latency_ms_per_token = _extract_numeric(combined, "ms per token", default=0.0)

    # Best-effort memory extraction (llama.cpp outputs vary).
    # We look for nearby "mb" markers and take the most recent number.
    memory_mb = _extract_numeric(combined, "mb", default=0.0)

    return (
        latency_ms_per_token if latency_ms_per_token > 0 else None,
        throughput_tps if throughput_tps > 0 else None,
        memory_mb if memory_mb > 0 else None,
    )


def _fit_multiplier(observed: list[float], simulated: list[float]) -> float:
    ratios: list[float] = []
    for o, s in zip(observed, simulated, strict=True):
        r = _safe_ratio(o, s)
        if r is not None and 0.01 < r < 100.0:
            ratios.append(r)
    if not ratios:
        return 1.0
    return _mean(ratios)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit simulator calibration multipliers from llama.cpp measurements.")
    parser.add_argument("--run-name", default="llama_cpp_calibration", help="Output artifact run name prefix.")
    parser.add_argument("--prompts", default="6", help="Number of random prompts to sample for calibration.")
    parser.add_argument("--seed", default="1234", help="RNG seed for prompt sampling.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from config import CONFIG as BASE

    # Calibration must run with a real llama.cpp binary + model.
    config = BASE.clone(backend="llama_cpp", prompt_split_enabled=False)
    try:
        require_llama_cpp_paths(config)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(int(args.seed))
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

        for _ in range(prompt_count):
            state = env.reset(forced_hardware=hw, phase="eval")
            prompt_text = state.prompt.text[: config.llama_cpp_max_prompt_chars]
            lat_ms_tok, tps, mem_mb = _run_llama_cpp_once(config, prompt_text=prompt_text, ngl=state.hardware_profile.ngl)

            sim_metrics = sim_backend.evaluate(state, fixed_decision)
            # Convert "ms per token" to "total latency" consistent with env metric definition.
            if lat_ms_tok is not None:
                observed_latency.append(lat_ms_tok * max(1, state.input_features.prompt_length))
                sim_latency.append(float(sim_metrics["latency_ms"]))
            if tps is not None:
                observed_throughput.append(tps)
                sim_throughput.append(float(sim_metrics["throughput_tps"]))
            if mem_mb is not None:
                observed_memory.append(mem_mb)
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
                "latency_multiplier": _fit_multiplier(observed_latency, sim_latency) if observed_latency else 1.0,
                "throughput_multiplier": _fit_multiplier(observed_throughput, sim_throughput) if observed_throughput else 1.0,
                "memory_multiplier": _fit_multiplier(observed_memory, sim_memory) if observed_memory else 1.0,
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
    print("Wrote calibration:", out_path)


if __name__ == "__main__":
    main()

