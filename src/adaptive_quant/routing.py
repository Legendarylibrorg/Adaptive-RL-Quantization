"""Learned task routing for picking the most compute-efficient "route".

A *route* is typically "which model to use" + "which quantization size/config to use".
This module provides a small contextual bandit (policy-gradient + value baseline) that
can learn from observed rewards (e.g. negative latency, cost, quality penalties).

Design constraints:
- Default path is **stdlib-only** (hash features).
- Optional Hugging Face embeddings via ``transformers`` + ``torch`` when configured;
  model weights are loaded with ``use_safetensors=True`` only (no Hub ``*.bin`` tensors).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import (
    validate_hf_model_id,
    validate_router_task_text,
    validate_runtime_filesystem_path,
)
from adaptive_quant.math_utils import argmax, dot, sample_categorical, softmax


def _router_hf_pretrained_kwargs(config: FrameworkConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(tokenizer_kwargs, model_kwargs)`` for Hugging Face Hub loads.

    Model weights must be ``safetensors``; ``trust_remote_code`` stays disabled.
    """
    shared: dict[str, Any] = {
        "revision": config.router_hf_embedding_revision,
        "local_files_only": bool(config.router_hf_local_files_only),
        "trust_remote_code": False,
    }
    model_kw = {**shared, "use_safetensors": True}
    return shared, model_kw


@dataclass(frozen=True)
class RouteCandidate:
    backend: str
    model_id: str
    quant_bits: int | None = None

    @property
    def key(self) -> str:
        prefix = f"{self.backend}:{self.model_id}"
        if self.quant_bits is None:
            return prefix
        return f"{prefix}@q{self.quant_bits}"

    def llama_cpp_model_path(self) -> str | None:
        """Filesystem GGUF path for ``llama_cpp`` routes (validated at parse time)."""
        if self.backend != "llama_cpp":
            return None
        return self.model_id


def parse_route(route: str) -> RouteCandidate:
    """Parse ``<model>@q<bits>`` into a ``RouteCandidate`` (or treat as bare model id)."""
    if not isinstance(route, str):
        raise TypeError("route must be a string")
    raw = route.strip()
    if not raw:
        raise ValueError("route must be non-empty")

    # Backend prefix: "hf:<model_id>" or "llama_cpp:<path>".
    backend = "hf"
    model_part = raw
    if ":" in raw:
        maybe_backend, rest = raw.split(":", 1)
        maybe_backend = maybe_backend.strip().lower()
        if maybe_backend in {"hf", "llama_cpp"}:
            backend = maybe_backend
            model_part = rest.strip()

    if "@q" not in model_part:
        candidate = RouteCandidate(backend=backend, model_id=model_part, quant_bits=None)
        if backend == "llama_cpp":
            validate_runtime_filesystem_path("llama_cpp_route_model", model_part)
        elif backend == "hf":
            validate_hf_model_id("router_route_model_id", model_part, require_hub_namespace=True)
        return candidate
    model_id, suffix = model_part.rsplit("@q", 1)
    model_id = model_id.strip()
    suffix = suffix.strip()
    if not model_id:
        raise ValueError(f"Invalid route {route!r}: missing model id")
    try:
        bits = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid route {route!r}: expected '@q<int>' suffix") from exc
    if bits <= 0:
        raise ValueError(f"Invalid route {route!r}: quant bits must be > 0")
    candidate = RouteCandidate(backend=backend, model_id=model_id, quant_bits=bits)
    if backend == "llama_cpp":
        validate_runtime_filesystem_path("llama_cpp_route_model", model_id)
    elif backend == "hf":
        validate_hf_model_id("router_route_model_id", model_id, require_hub_namespace=True)
    return candidate


@dataclass
class RouterTrace:
    feature_vector: list[float]
    selected_index: int
    probabilities: list[float]
    value_prediction: float


class _CategoricalHead:
    def __init__(self, input_dim: int, output_dim: int, rng: random.Random) -> None:
        scale = 0.08
        self.weights = [
            [rng.uniform(-scale, scale) for _ in range(input_dim)] for _ in range(output_dim)
        ]
        self.bias = [0.0] * output_dim

    def logits(self, feature_vector: list[float]) -> list[float]:
        return [
            dot(row, feature_vector) + b for row, b in zip(self.weights, self.bias, strict=True)
        ]

    def sample(
        self,
        feature_vector: list[float],
        rng: random.Random,
        *,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> tuple[int, list[float]]:
        """Sample an arm. With probability ``epsilon``, pick **uniformly** among arms (not ε-greedy vs current π)."""
        probabilities = softmax(self.logits(feature_vector))
        if deterministic:
            return argmax(probabilities), probabilities
        if epsilon > 0.0 and rng.random() < float(epsilon):
            return rng.randrange(len(probabilities)), probabilities
        return sample_categorical(probabilities, rng), probabilities

    def update(
        self,
        feature_vector: list[float],
        selected_index: int,
        probabilities: list[float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        for row_index, row in enumerate(self.weights):
            coefficient = (
                (1.0 if row_index == selected_index else 0.0) - probabilities[row_index]
            ) * advantage
            for column_index, value in enumerate(feature_vector):
                row[column_index] += learning_rate * coefficient * value
            self.bias[row_index] += learning_rate * coefficient


class _ValueHead:
    def __init__(self, input_dim: int, rng: random.Random) -> None:
        # Start at 0 so early advantages reflect observed rewards (more stable than random init).
        del rng
        self.weights = [0.0 for _ in range(input_dim)]
        self.bias = 0.0

    def predict(self, feature_vector: list[float]) -> float:
        return dot(self.weights, feature_vector) + self.bias

    def update(self, feature_vector: list[float], target: float, learning_rate: float) -> None:
        prediction = self.predict(feature_vector)
        error = target - prediction
        for index, value in enumerate(feature_vector):
            self.weights[index] += learning_rate * error * value
        self.bias += learning_rate * error


def _stable_l2_normalize(vector: list[float]) -> list[float]:
    norm2 = sum(v * v for v in vector)
    if norm2 <= 0.0:
        return vector
    inv = 1.0 / math.sqrt(norm2)
    return [v * inv for v in vector]


def _hash_features(text: str, *, dim: int) -> list[float]:
    """Stdlib-only feature extractor using a hashing trick into a dense vector."""
    text = (text or "").strip()
    vec = [0.0] * dim
    if not text:
        return vec
    # Simple tokenization: whitespace, lowercase. Good enough for routing signals.
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little", signed=False) % dim
        sign = -1.0 if (digest[4] & 1) else 1.0
        vec[idx] += sign
    return _stable_l2_normalize(vec)


def _finite(value: float, *, label: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return v


class EfficientTaskRouter:
    """Learned router that selects among configured ``router_routes`` and updates from rewards."""

    def __init__(self, config: FrameworkConfig, *, seed_offset: int = 909) -> None:
        self.config = config
        self.rng = random.Random(config.seed + seed_offset)
        self.routes = [parse_route(r) for r in config.router_routes]
        if not self.routes:
            raise ValueError("Router enabled but config.router_routes is empty.")
        if len({r.key for r in self.routes}) != len(self.routes):
            raise ValueError("Router routes must be unique.")

        # We pick a fixed feature width for the stdlib backend; HF backend uses model hidden size.
        self.hash_dim = 128
        self._hf: dict[str, object] | None = None
        backend = self.config.router_feature_backend.strip().lower()
        if backend == "hf":
            if not self.config.router_hf_embedding_model:
                raise ValueError(
                    "router_feature_backend='hf' requires router_hf_embedding_model to be set."
                )
            # Lazy init on first call so importing this module remains light.
            self._feature_dim: int | None = None
        elif backend == "hash":
            self._feature_dim = self.hash_dim
        else:
            raise ValueError(
                f"Unsupported router_feature_backend: {self.config.router_feature_backend!r}"
            )

        self.policy_head = _CategoricalHead(self.feature_dim, len(self.routes), self.rng)
        self.value_head = _ValueHead(self.feature_dim, self.rng)

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            # Discover hidden size by producing one embedding. Cached after first call.
            vec = self._hf_features("shape probe")
            self._feature_dim = len(vec)
        return int(self._feature_dim)

    def _ensure_hf_loaded(self) -> dict[str, object]:
        if self._hf is not None:
            return self._hf
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "router_feature_backend='hf' requires 'transformers' and 'torch'. "
                "Install them (e.g. pip install -e \".[torch,router]\") or use router_feature_backend='hash'."
            ) from exc

        model_id = (self.config.router_hf_embedding_model or "").strip()
        if not model_id:
            raise ValueError(
                "router_hf_embedding_model must be set for router_feature_backend='hf'."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_kw, model_kw = _router_hf_pretrained_kwargs(self.config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kw)
        model = AutoModel.from_pretrained(model_id, **model_kw)
        model.eval()
        model.to(device)
        self._hf = {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device}
        return self._hf

    def _hf_features(self, text: str) -> list[float]:
        hf = self._ensure_hf_loaded()
        torch = hf["torch"]
        tokenizer = hf["tokenizer"]
        model = hf["model"]
        device = hf["device"]

        with torch.no_grad():
            batch = tokenizer(text or "", return_tensors="pt", truncation=True, max_length=256)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            hidden = out.last_hidden_state
            mask = batch.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask_f).sum(dim=1)
                denom = mask_f.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            vec = pooled[0].detach().float().cpu().tolist()
        return _stable_l2_normalize([float(x) for x in vec])

    def featurize(self, *, task_text: str) -> list[float]:
        task_text = validate_router_task_text(task_text)
        backend = self.config.router_feature_backend.strip().lower()
        if backend == "hf":
            return self._hf_features(task_text)
        return _hash_features(task_text, dim=self.hash_dim)

    def route(
        self, *, task_text: str, deterministic: bool = False
    ) -> tuple[RouteCandidate, RouterTrace]:
        task_text = validate_router_task_text(task_text)
        feature_vector = self.featurize(task_text=task_text)
        value_prediction = self.value_head.predict(feature_vector)
        selected_index, probabilities = self.policy_head.sample(
            feature_vector,
            self.rng,
            deterministic=deterministic,
            epsilon=float(self.config.router_exploration),
        )
        trace = RouterTrace(
            feature_vector=feature_vector,
            selected_index=selected_index,
            probabilities=probabilities,
            value_prediction=value_prediction,
        )
        return self.routes[selected_index], trace

    def update(self, trace: RouterTrace, *, reward: float) -> None:
        advantage = float(reward) - float(trace.value_prediction)
        self.policy_head.update(
            trace.feature_vector,
            trace.selected_index,
            trace.probabilities,
            advantage,
            float(self.config.router_learning_rate),
        )
        self.value_head.update(
            trace.feature_vector,
            float(reward),
            float(self.config.router_value_learning_rate),
        )

    def reward_from_metrics(
        self,
        *,
        memory_mb: float,
        perplexity: float,
        baseline_perplexity: float | None,
        latency_ms: float | None = None,
    ) -> float:
        """Compute a router reward that prioritizes VRAM without quality regression.

        - If a baseline perplexity is provided and the route regresses beyond configured limits,
          return a large negative penalty.
        - Otherwise reward is dominated by *lower memory* (VRAM), with optional small latency term.
        """
        mem = _finite(memory_mb, label="memory_mb")
        ppl = _finite(perplexity, label="perplexity")
        if mem < 0.0:
            mem = 0.0
        if baseline_perplexity is not None:
            base = _finite(baseline_perplexity, label="baseline_perplexity")
            if base <= 0.0:
                base = 1e-6
            ratio_limit = float(self.config.router_max_perplexity_ratio)
            delta_limit = float(self.config.router_max_perplexity_delta)
            regresses = (ppl > base * ratio_limit) or ((ppl - base) > delta_limit)
            if regresses:
                return -float(self.config.router_regression_penalty) - mem

        # Main objective: minimize memory. Secondary: (optional) minimize latency a bit.
        reward = -mem
        if latency_ms is not None:
            lat = _finite(latency_ms, label="latency_ms")
            reward -= 0.001 * lat
        return reward


__all__ = [
    "EfficientTaskRouter",
    "RouteCandidate",
    "RouterTrace",
    "parse_route",
]
