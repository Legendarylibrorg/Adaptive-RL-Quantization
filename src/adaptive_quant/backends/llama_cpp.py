from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections import OrderedDict

from adaptive_quant.backends.protocol import per_token_latency_fields
from adaptive_quant.backends.quality import ExternalQualityScores, apply_external_quality
from adaptive_quant.backends.simulator import SimulatorBackend
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import (
    validate_llama_cpp_binary_allowlist,
    validate_runtime_filesystem_path,
)
from adaptive_quant.types import BackendMetricDict, EpisodeState, QuantizationDecision

_NUMBER_RE = r"-?\d+(?:\.\d+)?"


def _extract_numeric(text: str, marker: str, default: float) -> float:
    if not text or not marker:
        return default
    escaped = re.escape(marker)
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
    if not text:
        return default

    label = r"(?:mem(?:ory)?|rss|resident|kv(?:\s+cache)?|cuda(?:\s+memory)?|gpu(?:\s+memory)?)"
    unit = r"(?:mib|mb)"

    patterns = [
        re.compile(rf"{label}[^0-9]{{0,64}}(?P<num>{_NUMBER_RE})\s*(?P<unit>{unit})\b"),
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
            if raw_unit not in ("mb", "mib"):
                continue
            last_value = value
    return last_value


def parse_llama_cpp_metrics(text: str) -> dict[str, float]:
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


def _output_excerpt(text: str, *, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "(no output)"
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _llama_cpp_command(
    config: FrameworkConfig,
    *,
    llama_cpp_binary: str,
    llama_cpp_model: str,
    prompt_text: str,
    ngl: int,
) -> list[str]:
    prompt_text = (
        (prompt_text or "").replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
    )
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


def require_llama_cpp_paths(
    config: FrameworkConfig,
    *,
    model_override: object | None = None,
) -> tuple[str, str]:
    binary = getattr(config, "llama_cpp_binary", None)
    model = model_override or getattr(config, "llama_cpp_model", None)
    if not binary or not model:
        raise FileNotFoundError("llama.cpp backend requires both a binary path and a model path.")
    validate_runtime_filesystem_path("llama_cpp_binary", str(binary))
    validate_runtime_filesystem_path("llama_cpp_model", str(model))
    binary = os.path.realpath(binary)
    model = os.path.realpath(model)
    validate_llama_cpp_binary_allowlist(binary)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Missing llama.cpp binary: {binary}")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Missing model file: {model}")
    return str(binary), str(model)


class LlamaCppBackend:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self._simulator = SimulatorBackend(config)
        self.external_quality = ExternalQualityScores.from_config(config)
        self._cache: OrderedDict[str, dict[str, float]] | None = None
        if bool(getattr(config, "llama_cpp_cache_enabled", False)):
            self._cache = OrderedDict()

    def evaluate(self, state: EpisodeState, decision: QuantizationDecision) -> BackendMetricDict:
        llama_cpp_binary, llama_cpp_model = require_llama_cpp_paths(
            self.config,
            model_override=decision.metadata.get("llama_cpp_model_path"),
        )
        parsed = self._run_or_cache_measurement(
            llama_cpp_binary=llama_cpp_binary,
            llama_cpp_model=llama_cpp_model,
            prompt_text=state.prompt.text,
            ngl=state.hardware_profile.ngl,
        )
        metrics = self._simulator.evaluate(state, decision)

        if parsed.get("throughput_tps", 0.0) > 0.0:
            metrics["throughput_tps"] = float(parsed["throughput_tps"])
        if parsed.get("latency_ms_per_token", 0.0) > 0.0:
            metrics["latency_ms"] = float(parsed["latency_ms_per_token"]) * max(
                1, state.input_features.prompt_length
            )
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

    def _run_or_cache_measurement(
        self,
        *,
        llama_cpp_binary: str,
        llama_cpp_model: str,
        prompt_text: str,
        ngl: int,
    ) -> dict[str, float]:
        cache = self._cache
        if cache is None:
            return run_llama_cpp_measurement(
                self.config,
                llama_cpp_binary=llama_cpp_binary,
                llama_cpp_model=llama_cpp_model,
                prompt_text=prompt_text,
                ngl=ngl,
            )

        max_chars = int(getattr(self.config, "llama_cpp_max_prompt_chars", 0))
        text = (
            (prompt_text or "").replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
        )
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        key = "|".join(
            [
                str(llama_cpp_binary),
                str(llama_cpp_model),
                str(int(ngl)),
                str(int(self.config.llama_cpp_threads)),
                str(int(self.config.llama_cpp_context)),
                str(int(self.config.llama_cpp_generate_tokens)),
                digest,
            ]
        )
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return dict(cached)

        parsed = run_llama_cpp_measurement(
            self.config,
            llama_cpp_binary=llama_cpp_binary,
            llama_cpp_model=llama_cpp_model,
            prompt_text=text,
            ngl=ngl,
        )
        cache[key] = dict(parsed)
        cache.move_to_end(key)
        max_entries = int(getattr(self.config, "llama_cpp_cache_max_entries", 256))
        while len(cache) > max_entries:
            cache.popitem(last=False)
        return parsed
