from __future__ import annotations

import math
import statistics


def safe_ratio(numer: float, denom: float) -> float | None:
    if not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return None
    return numer / denom


def ratio_mean(observed: list[float], simulated: list[float], *, clamp: tuple[float, float] = (0.01, 100.0)) -> float:
    ratios: list[float] = []
    lower, upper = clamp
    for o, s in zip(observed, simulated, strict=True):
        r = safe_ratio(o, s)
        if r is not None and lower < r < upper:
            ratios.append(r)
    return float(statistics.fmean(ratios)) if ratios else 1.0


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def fmt_float(x: float, *, digits: int = 2) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


__all__ = ["fmt_float", "ratio_mean", "safe_ratio", "sample_std"]

