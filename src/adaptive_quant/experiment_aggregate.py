"""Shared numeric flattening and aggregation for multiseed / hyperparameter sweep runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from adaptive_quant.paper_bundle import aggregate_values


@dataclass(frozen=True)
class AggregateStat:
    mean: float
    std: float
    n: int
    stderr: float = 0.0
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    effect_size_vs_zero: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "stderr": self.stderr,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "effect_size_vs_zero": self.effect_size_vs_zero,
        }


def is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_numeric(
    obj: object,
    *,
    prefix: str = "",
    max_items: int = 10_000,
) -> dict[str, float]:
    out: dict[str, float] = {}

    def walk(node: object, path: str) -> None:
        if len(out) >= max_items:
            return
        if is_numeric(node):
            out[path] = float(node)  # type: ignore[arg-type]
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    continue
                walk(v, f"{path}.{k}" if path else k)
            return
        if isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(obj, prefix)
    return out


def default_key_filter(key: str) -> bool:
    key_lower = key.lower()
    if key_lower.startswith("config."):
        return False
    if key_lower.endswith("_delta"):
        return True
    if "gap" in key_lower:
        return True
    for needle in (
        "mean_reward",
        "mean_latency_ms",
        "mean_throughput_tps",
        "mean_memory_mb",
        "mean_perplexity",
        "mean_stability_penalty",
        "generalization_gap_improvement",
        "single_policy_gap",
        "multi_policy_gap",
    ):
        if needle in key_lower:
            return True
    return False


def aggregate_numeric_maps(maps: list[dict[str, float]]) -> dict[str, AggregateStat]:
    keys: set[str] = set()
    for m in maps:
        keys.update(m.keys())

    aggregated: dict[str, AggregateStat] = {}
    for key in sorted(keys):
        values = [m[key] for m in maps if key in m and math.isfinite(m[key])]
        if not values:
            continue
        stats = aggregate_values(values)
        aggregated[key] = AggregateStat(
            mean=float(stats["mean"]),
            std=float(stats["std"]),
            n=int(stats["n"]),
            stderr=float(stats["stderr"]),
            ci95_low=float(stats["ci95_low"]),
            ci95_high=float(stats["ci95_high"]),
            effect_size_vs_zero=float(stats["effect_size_vs_zero"]),
        )
    return aggregated


def extract_metric(summary: dict[str, Any], objective: str) -> float | None:
    flat = flatten_numeric(summary)
    if objective in flat and math.isfinite(flat[objective]):
        return flat[objective]
    suffix = f".{objective}"
    for key, value in flat.items():
        if key.endswith(suffix) or key.split(".")[-1] == objective:
            if math.isfinite(value):
                return value
    return None


__all__ = [
    "AggregateStat",
    "aggregate_numeric_maps",
    "default_key_filter",
    "extract_metric",
    "flatten_numeric",
    "is_numeric",
]
