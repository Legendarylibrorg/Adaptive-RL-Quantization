from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Sequence


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stable_sigmoid(value: float) -> float:
    """Numerically stable logistic (sigmoid) for a scalar."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(lhs * rhs for lhs, rhs in zip(left, right, strict=False))


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    shifted = [math.exp(logit - max_logit) for logit in logits]
    total = sum(shifted)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in shifted]


def sample_categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if threshold <= running:
            return index
    return max(0, len(probabilities) - 1)


def argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def deterministic_float(key: str, lower: float = 0.0, upper: float = 1.0) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return lower + (upper - lower) * bucket


def stable_hash_int(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def gaussian_sample(mean_value: float, stddev: float, rng: random.Random) -> float:
    if stddev <= 0.0:
        return mean_value
    return rng.gauss(mean_value, stddev)


def safe_ratio(numer: float, denom: float) -> float | None:
    if not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return None
    return numer / denom


def ratio_mean(observed: list[float], simulated: list[float], *, clamp: tuple[float, float] = (0.01, 100.0)) -> float:
    lower, upper = clamp
    ratios = [r for o, s in zip(observed, simulated, strict=True) if (r := safe_ratio(o, s)) is not None and lower < r < upper]
    return float(statistics.fmean(ratios)) if ratios else 1.0


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def fmt_float(x: float, *, digits: int = 2) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"

