from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, Sequence


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    return sum(lhs * rhs for lhs, rhs in zip(left, right))


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


def chunked(values: Sequence[float], chunk_size: int) -> list[list[float]]:
    return [list(values[index : index + chunk_size]) for index in range(0, len(values), chunk_size)]


def stable_hash_int(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def sliding_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    averaged: list[float] = []
    for index in range(len(values)):
        lower = max(0, index - window + 1)
        averaged.append(mean(values[lower : index + 1]))
    return averaged


def gaussian_sample(mean_value: float, stddev: float, rng: random.Random) -> float:
    if stddev <= 0.0:
        return mean_value
    return rng.gauss(mean_value, stddev)


def flatten(rows: Iterable[Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for row in rows:
        flattened.extend(row)
    return flattened

