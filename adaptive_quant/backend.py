from __future__ import annotations

import os
import re
import subprocess
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import load_jsonl, read_json
from adaptive_quant.math_utils import clamp, mean, variance
from adaptive_quant.moe import ExpertBank
from adaptive_quant.types import (
    BackendMetricDict,
    EpisodeState,
    HardwareType,
    QuantizationDecision,
    QuantMode,
)


class Backend(Protocol):
    """Evaluation backend interface (simulator or llama.cpp)."""

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict: ...


def build_backend(config: FrameworkConfig) -> Backend:
    """Build the measurement backend configured for an experiment."""
    backend = config.backend.strip().lower()
    if backend == "simulator":
        return SimulatorBackend(config)
    if backend == "llama_cpp":
        return LlamaCppBackend(config)
    raise ValueError(f"Unsupported backend {config.backend!r}; expected 'simulator' or 'llama_cpp'.")


def per_token_latency_fields(state: EpisodeState, latency_ms: float) -> dict[str, float]:
    """Normalize wall-clock latency by prompt length for logging and optional reward (see reward_weights.eta_token_latency)."""
    tokens = float(max(1, state.input_features.prompt_length))
    return {
        "tokens_processed": tokens,
        "latency_ms_per_token": float(latency_ms) / tokens,
    }


class SimulatorBackend:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.expert_bank = ExpertBank(config) if config.moe_enabled else None
        self.external_quality = ExternalQualityScores.from_config(config)

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict:
        hardware = state.hardware_profile
        avg_bits = mean(decision.effective_layer_bits)
        bit_variance = variance(decision.effective_layer_bits)
        complexity = state.input_features.complexity_score
        sensitivity = mean(state.sensitivity.layer_stats)
        prompt_length = max(8, state.input_features.prompt_length)
        compression = max(0.0, (8.0 - avg_bits) / 6.0)

        mode_bonus = {
            QuantMode.DISCRETE: 0.10,
            QuantMode.GROUPED: 0.16,
            QuantMode.PER_LAYER: 0.18,
            QuantMode.DYNAMIC: 0.28,
            QuantMode.LEARNED: 0.34,
        }[decision.mode]

        latency_ms = (
            8.5
            * prompt_length
            * hardware.latency_bias
            / max(0.35, hardware.compute_factor + (8.0 - avg_bits) * 0.12 + mode_bonus)
        )
        latency_ms *= 1.0 + complexity * 0.55 + max(0.0, bit_variance - hardware.kernel_uniformity_preference) * 0.18

        throughput_tps = (
            140.0
            * hardware.throughput_bias
            * (1.0 + (8.0 - avg_bits) * 0.10 + mode_bonus * 0.40)
            / (1.0 + complexity * 0.80 + hardware.latency_bias * 0.08)
        )
        if hardware.hardware_type == HardwareType.GPU:
            throughput_tps *= 1.0 - min(0.12, bit_variance * 0.03)
        else:
            throughput_tps *= 1.0 + min(0.10, max(0.0, hardware.preferred_bits - avg_bits) * 0.02)

        memory_mb = 4_800.0 * (avg_bits / 16.0) * (1.0 + complexity * 0.15)
        if decision.mode in {QuantMode.PER_LAYER, QuantMode.LEARNED}:
            memory_mb *= 1.02

        perplexity = (
            5.6
            + complexity * 3.4
            + max(0.0, 5.5 - avg_bits) * (0.60 + complexity * 0.90 + sensitivity * 0.35)
            + abs(1.0 - decision.scale_factor) * 0.65
            + max(0.0, 1.05 - decision.clipping_range) * 1.20
            - mode_bonus * 0.70
        )

        hardware_alignment = abs(avg_bits - hardware.preferred_bits)
        latency_ms *= 1.0 + hardware_alignment * 0.04
        throughput_tps *= 1.0 - hardware_alignment * 0.02
        perplexity += hardware_alignment * 0.15

        if hardware.hardware_type in {HardwareType.CPU, HardwareType.LOW_RESOURCE} and avg_bits > hardware.preferred_bits:
            excess_bits = avg_bits - hardware.preferred_bits
            latency_ms *= 1.0 + excess_bits * (0.16 if hardware.hardware_type == HardwareType.CPU else 0.24)
            throughput_tps *= max(0.55, 1.0 - excess_bits * (0.07 if hardware.hardware_type == HardwareType.CPU else 0.12))
            memory_mb *= 1.0 + excess_bits * (0.10 if hardware.hardware_type == HardwareType.CPU else 0.18)
        elif hardware.hardware_type == HardwareType.GPU and avg_bits < hardware.preferred_bits:
            deficit_bits = hardware.preferred_bits - avg_bits
            perplexity += deficit_bits * 0.45
            throughput_tps *= max(0.78, 1.0 - deficit_bits * 0.03)

        if decision.mode == QuantMode.DYNAMIC:
            latency_ms *= 0.92
            throughput_tps *= 1.06
            perplexity -= 0.25 + complexity * 0.20
        elif decision.mode == QuantMode.LEARNED:
            latency_ms *= 0.82 - compression * 0.06
            throughput_tps *= 1.12 + compression * 0.08
            memory_mb *= 0.78 - compression * 0.04
            perplexity -= 0.38 + sensitivity * 0.22
        elif decision.mode == QuantMode.GROUPED and hardware.hardware_type != HardwareType.GPU:
            latency_ms *= 0.95
            throughput_tps *= 1.03

        overflow_ratio = max(0.0, memory_mb - hardware.memory_budget_mb) / hardware.memory_budget_mb
        if overflow_ratio > 0.0:
            latency_ms *= 1.0 + overflow_ratio * 2.50
            throughput_tps *= 1.0 / (1.0 + overflow_ratio * 1.8)
            perplexity += overflow_ratio * 1.50

        swap_cost_ms = 0.0
        cache_miss_count = 0.0
        variant_churn = float(decision.metadata.get("moe_variant_churn", 0.0))
        if self.expert_bank is not None and state.moe_context is not None and decision.moe_variant_indices:
            latency_ms, throughput_tps, perplexity, memory_mb, swap_cost_ms, cache_miss_count = self._apply_moe_adjustments(
                state,
                decision,
                latency_ms,
                throughput_tps,
                perplexity,
                memory_mb,
            )

        metrics: BackendMetricDict = {
            "latency_ms": clamp(latency_ms, 5.0, 20_000.0),
            "throughput_tps": clamp(throughput_tps, 1.0, 10_000.0),
            "perplexity": clamp(perplexity, 3.0, 100.0),
            "memory_mb": clamp(memory_mb, 200.0, 128_000.0),
            "swap_cost_ms": swap_cost_ms,
            "cache_miss_count": cache_miss_count,
            "variant_churn": variant_churn,
        }
        calibration = getattr(self.config, "sim_calibration", None)
        if isinstance(calibration, dict):
            hw_key = state.hardware_profile.hardware_type.value
            hw_cal = calibration.get(hw_key, {}) if isinstance(calibration.get(hw_key, {}), dict) else {}
            latency_mul = float(hw_cal.get("latency_multiplier", 1.0))
            throughput_mul = float(hw_cal.get("throughput_multiplier", 1.0))
            memory_mul = float(hw_cal.get("memory_multiplier", 1.0))
            if latency_mul > 0:
                metrics["latency_ms"] = clamp(metrics["latency_ms"] * latency_mul, 1.0, 60_000.0)
            if throughput_mul > 0:
                metrics["throughput_tps"] = clamp(metrics["throughput_tps"] * throughput_mul, 0.1, 100_000.0)
            if memory_mul > 0:
                metrics["memory_mb"] = clamp(metrics["memory_mb"] * memory_mul, 50.0, 512_000.0)
        metrics.update(per_token_latency_fields(state, metrics["latency_ms"]))
        metrics.update(
            {
                "latency_source": "simulator",
                "throughput_source": "simulator",
                "memory_source": "simulator",
                "perplexity_source": "simulator",
            }
        )
        apply_external_quality(metrics, state, self.external_quality)
        return metrics

    def _apply_moe_adjustments(
        self,
        state: EpisodeState,
        decision: QuantizationDecision,
        latency_ms: float,
        throughput_tps: float,
        perplexity: float,
        memory_mb: float,
    ) -> tuple[float, float, float, float, float, float]:
        assert self.expert_bank is not None
        total_swap_cost = 0.0
        cache_misses = 0.0
        throughput_multiplier = 1.0
        memory_multiplier = 1.0
        sensitivity_penalty = 0.0
        latency_multiplier = 1.0

        for expert, variant_index in zip(state.moe_context.experts, decision.moe_variant_indices):
            variant = self.expert_bank.variant_by_index(variant_index)
            routing_weight = 0.60 + expert.router_probability
            latency_multiplier *= 1.0 + (variant.latency_multiplier - 1.0) * routing_weight * 0.50
            throughput_multiplier *= 1.0 + (variant.throughput_multiplier - 1.0) * routing_weight * 0.55
            memory_multiplier *= 1.0 + (variant.memory_multiplier - 1.0) * routing_weight * 0.40
            sensitivity_penalty += variant.perplexity_penalty * expert.sensitivity * routing_weight
            if expert.resident_on_device < 0.5:
                cache_misses += 1.0
                total_swap_cost += variant.swap_cost_ms * (1.0 + expert.router_probability) * (1.10 - 0.35 * expert.hotness)

        latency_ms = latency_ms * latency_multiplier + total_swap_cost
        throughput_tps *= throughput_multiplier
        memory_mb *= memory_multiplier
        perplexity += sensitivity_penalty
        return latency_ms, throughput_tps, perplexity, memory_mb, total_swap_cost, cache_misses


class LlamaCppBackend:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self._simulator = SimulatorBackend(config)
        self.external_quality = ExternalQualityScores.from_config(config)

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict:
        llama_cpp_binary, llama_cpp_model = require_llama_cpp_paths(
            self.config,
            model_override=decision.metadata.get("llama_cpp_model_path"),
        )
        parsed = run_llama_cpp_measurement(
            self.config,
            llama_cpp_binary=llama_cpp_binary,
            llama_cpp_model=llama_cpp_model,
            prompt_text=state.prompt.text,
            ngl=state.hardware_profile.ngl,
        )
        metrics = self._simulator.evaluate(state, decision)

        if parsed.get("throughput_tps", 0.0) > 0.0:
            metrics["throughput_tps"] = float(parsed["throughput_tps"])
        if parsed.get("latency_ms_per_token", 0.0) > 0.0:
            metrics["latency_ms"] = float(parsed["latency_ms_per_token"]) * max(1, state.input_features.prompt_length)
        if parsed.get("memory_mb", 0.0) > 0.0:
            metrics["memory_mb"] = float(parsed["memory_mb"])
        metrics.update(per_token_latency_fields(state, metrics["latency_ms"]))
        metrics.update(
            {
                "latency_source": "llama_cpp",
                "throughput_source": "llama_cpp",
                "memory_source": "llama_cpp" if parsed.get("memory_mb", 0.0) > 0.0 else "simulator",
                "perplexity_source": "simulator",
            }
        )
        apply_external_quality(metrics, state, self.external_quality)
        return metrics


class ExternalQualityScores:
    """Prompt-level external quality scores loaded from JSON/JSONL.

    Expected shape examples:
      {"very_complex": {"perplexity": 8.9}}
      [{"prompt_id": "very_complex", "perplexity": 8.9}]

    The default metric is "perplexity", which is compatible with the existing lower-is-better reward.
    """

    def __init__(self, scores_by_prompt_id: Mapping[str, float], *, metric: str, path: str) -> None:
        self.scores_by_prompt_id = dict(scores_by_prompt_id)
        self.metric = metric
        self.path = path

    @classmethod
    def from_config(cls, config: FrameworkConfig) -> "ExternalQualityScores | None":
        path = getattr(config, "external_quality_path", None)
        if not path:
            return None
        metric = str(getattr(config, "external_quality_metric", "perplexity") or "perplexity")
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"External quality file not found: {source}")
        if source.suffix.lower() == ".jsonl":
            rows = load_jsonl(str(source))
            scores = _quality_scores_from_rows(rows, metric=metric)
        else:
            payload = read_json(source, label="External quality file")
            scores = _quality_scores_from_payload(payload, metric=metric)
        if not scores:
            raise ValueError(f"External quality file has no finite `{metric}` scores keyed by prompt_id: {source}")
        return cls(scores, metric=metric, path=str(source))

    def score_for_prompt(self, prompt_id: str) -> float | None:
        return self.scores_by_prompt_id.get(prompt_id)


def apply_external_quality(
    metrics: BackendMetricDict,
    state: EpisodeState,
    external_quality: ExternalQualityScores | None,
) -> None:
    if external_quality is None:
        return
    score = external_quality.score_for_prompt(state.prompt.prompt_id)
    if score is None:
        metrics["perplexity_source"] = f"simulator_missing_external_{external_quality.metric}"
        return
    metrics["perplexity"] = float(score)
    metrics["perplexity_source"] = f"external:{external_quality.metric}"


def _quality_scores_from_payload(payload: object, *, metric: str) -> dict[str, float]:
    if isinstance(payload, Mapping):
        rows: list[Mapping[str, Any]] = []
        for prompt_id, value in payload.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("prompt_id", prompt_id)
                rows.append(row)
            else:
                rows.append({"prompt_id": prompt_id, metric: value})
        return _quality_scores_from_rows(rows, metric=metric)
    if isinstance(payload, list):
        return _quality_scores_from_rows(payload, metric=metric)
    raise TypeError("External quality JSON must be an object or list of rows")


def _quality_scores_from_rows(rows: object, *, metric: str) -> dict[str, float]:
    if not isinstance(rows, list):
        raise TypeError("External quality rows must be a list")
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        prompt_id = row.get("prompt_id")
        raw_value = row.get(metric)
        if not isinstance(prompt_id, str) or not prompt_id:
            continue
        if isinstance(raw_value, bool) or raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores[prompt_id] = value
    return scores


def run_llama_cpp_measurement(
    config: FrameworkConfig,
    *,
    llama_cpp_binary: str,
    llama_cpp_model: str,
    prompt_text: str,
    ngl: int,
) -> dict[str, float]:
    command = _llama_cpp_command(
        config,
        llama_cpp_binary=llama_cpp_binary,
        llama_cpp_model=llama_cpp_model,
        prompt_text=prompt_text,
        ngl=ngl,
    )
    timeout_s = float(config.llama_cpp_timeout_s)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"llama.cpp command timed out after {timeout_s:.1f}s.") from exc

    raw_output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    parsed = parse_llama_cpp_metrics(raw_output.lower())
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama.cpp command failed with exit code {completed.returncode}: "
            f"{_output_excerpt(raw_output)}"
        )
    if not any(key in parsed for key in ("throughput_tps", "latency_ms_per_token")):
        raise RuntimeError(
            "llama.cpp output did not include parseable throughput or latency timings: "
            f"{_output_excerpt(raw_output)}"
        )
    return parsed


def _llama_cpp_command(
    config: FrameworkConfig,
    *,
    llama_cpp_binary: str,
    llama_cpp_model: str,
    prompt_text: str,
    ngl: int,
) -> list[str]:
    max_chars = int(config.llama_cpp_max_prompt_chars)
    if max_chars > 0 and len(prompt_text) > max_chars:
        prompt_text = prompt_text[:max_chars]
    return [
        llama_cpp_binary,
        "-m",
        llama_cpp_model,
        "-p",
        prompt_text,
        "-ngl",
        str(ngl),
        "-t",
        str(config.llama_cpp_threads),
        "-c",
        str(config.llama_cpp_context),
        "-n",
        str(config.llama_cpp_generate_tokens),
    ]


def _output_excerpt(text: str, *, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "(no output)"
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def require_llama_cpp_paths(
    config: FrameworkConfig,
    *,
    model_override: object | None = None,
) -> tuple[str, str]:
    binary = getattr(config, "llama_cpp_binary", None)
    model = model_override or getattr(config, "llama_cpp_model", None)
    if not binary or not model:
        raise FileNotFoundError("llama.cpp backend requires both a binary path and a model path.")
    for label, path in (("llama_cpp_binary", binary), ("llama_cpp_model", model)):
        if not isinstance(path, str):
            raise TypeError(f"{label} must be a string path, got {type(path).__name__}")
        if "\n" in path or "\r" in path or "\x00" in path:
            raise ValueError(f"{label} contains invalid control characters; refuse ambiguous paths.")
    binary = os.path.realpath(binary)
    model = os.path.realpath(model)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Missing llama.cpp binary: {binary}")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Missing model file: {model}")
    return str(binary), str(model)


_NUMBER_RE = r"-?\d+(?:\.\d+)?"


def _extract_numeric(text: str, marker: str, default: float) -> float:
    """
    Extract the last float that appears immediately before a marker.

    Example markers: "tok/s", "ms per token", "mb".
    """
    if not text or not marker:
        return default
    escaped = re.escape(marker)
    # Support both "12.3 tok/s" and "tok/s 12.3" styles.
    pattern = re.compile(
        rf"(?:(?P<num_before>{_NUMBER_RE})\s*{escaped}|{escaped}\s*(?P<num_after>{_NUMBER_RE}))"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return default
    try:
        last = matches[-1]
        value = last.group("num_before") or last.group("num_after")
        return float(value) if value is not None else default
    except ValueError:
        return default


def _extract_memory_mb(text: str, default: float = 0.0) -> float:
    """
    Best-effort extraction for memory usage lines in llama.cpp output.

    We intentionally avoid the old "find any number before 'mb'" behavior because many logs
    contain unrelated "mb" tokens that would poison the metrics (e.g. batch sizes, model info).
    """
    if not text:
        return default

    # Common llama.cpp / system patterns tend to include a memory label near the unit.
    # Examples we try to match:
    #   "mem: 1234.5 mb"
    #   "memory 1234 mb"
    #   "rss 512 mb"
    #   "kv cache: 2048 mb"
    #   "cuda memory: 4096 MiB"
    label = r"(?:mem(?:ory)?|rss|resident|kv(?:\s+cache)?|cuda(?:\s+memory)?|gpu(?:\s+memory)?)"
    unit = r"(?:mib|mb)"

    patterns = [
        # label ... number unit
        re.compile(rf"{label}[^0-9]{{0,64}}(?P<num>{_NUMBER_RE})\s*(?P<unit>{unit})\b"),
        # number unit ... label
        re.compile(rf"(?P<num>{_NUMBER_RE})\s*(?P<unit>{unit})\b[^a-z]{{0,64}}{label}\b"),
    ]

    last_value = default
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw = match.group("num")
            raw_unit = (match.group("unit") or "").lower()
            try:
                value = float(raw)
            except ValueError:
                continue
            if value <= 0.0:
                continue
            # MiB and MB are close enough for this coarse telemetry; keep the number as "MB-like".
            if raw_unit not in ("mb", "mib"):
                continue
            last_value = value
    return last_value


def parse_llama_cpp_metrics(text: str) -> dict[str, float]:
    """
    Best-effort parser for common llama.cpp CLI output.
    Returns keys when found:
      - latency_ms_per_token
      - throughput_tps
      - memory_mb (best-effort; may be absent)
    """
    throughput_tps = _extract_numeric(text, "tok/s", default=0.0)
    latency_ms_per_token = _extract_numeric(text, "ms per token", default=0.0)
    memory_mb = _extract_memory_mb(text, default=0.0)
    result: dict[str, float] = {}
    if throughput_tps > 0.0:
        result["throughput_tps"] = throughput_tps
    if latency_ms_per_token > 0.0:
        result["latency_ms_per_token"] = latency_ms_per_token
    if memory_mb > 0.0:
        result["memory_mb"] = memory_mb
    return result
