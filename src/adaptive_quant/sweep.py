"""Hyperparameter sweep planning: grid expansion, trial naming, and ranking."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from adaptive_quant.cli.startup_overrides import (
    merge_override,
    normalize_override_key,
    parse_override_value,
)
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import safe_json_loads

SweepDirection = Literal["maximize", "minimize"]

SWEEP_META_KEYS = frozenset(
    {
        "base_config",
        "grid",
        "trials",
        "objective",
        "direction",
        "seed",
    }
)

DEFAULT_OBJECTIVE = "evaluation.mean_reward"


@dataclass(frozen=True)
class SweepSpec:
    objective: str = DEFAULT_OBJECTIVE
    direction: SweepDirection = "maximize"
    seed: int | None = None
    grid: dict[str, tuple[Any, ...]] | None = None
    trials: tuple[dict[str, Any], ...] | None = None
    base_config_path: str | None = None


@dataclass(frozen=True)
class SweepTrialPlan:
    trial_id: int
    overrides: dict[str, Any]
    run_name_suffix: str


@dataclass(frozen=True)
class SweepTrialResult:
    plan: SweepTrialPlan
    summary: dict[str, Any]
    summary_path: str
    objective_value: float | None


def parse_value_list(raw: str) -> list[Any]:
    text = raw.strip()
    if not text:
        return []
    return [parse_override_value(part.strip()) for part in text.split(",") if part.strip()]


def parse_vary_argument(raw: str) -> tuple[str, tuple[Any, ...]]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=val1,val2,... got {raw!r}")
    raw_key, values_text = raw.split("=", 1)
    key = normalize_override_key(raw_key)
    values = parse_value_list(values_text)
    if not values:
        raise ValueError(f"No values provided for sweep parameter {key!r}")
    return key, tuple(values)


def normalize_trial_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw_key, value in raw.items():
        key = normalize_override_key(str(raw_key))
        merge_override(overrides, key, value)
    return overrides


def expand_grid(grid: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        trial: dict[str, Any] = {}
        for key, value in zip(keys, values, strict=True):
            merge_override(trial, key, value)
        combos.append(trial)
    return combos


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return re.sub(r"[^a-zA-Z0-9]+", "p", text)[:16]


def trial_run_suffix(overrides: dict[str, Any], *, max_len: int = 48) -> str:
    if not overrides:
        return "default"
    parts: list[str] = []
    for key in sorted(overrides):
        value = overrides[key]
        if isinstance(value, dict):
            for field, field_value in sorted(value.items()):
                parts.append(f"{field}_{_short_value(field_value)}")
            continue
        short_key = key.split(".")[-1] if "." in key else key
        parts.append(f"{short_key}_{_short_value(value)}")
    slug = "_".join(parts)
    slug = re.sub(r"[^a-zA-Z0-9_]+", "", slug.replace(".", "_"))
    return slug[:max_len] or "default"


def build_trial_plans(
    *,
    grid: dict[str, tuple[Any, ...]] | None,
    explicit_trials: list[dict[str, Any]] | None,
) -> list[SweepTrialPlan]:
    if grid and explicit_trials:
        raise ValueError("Provide either grid or trials, not both.")
    if grid:
        trial_dicts = expand_grid(grid)
    elif explicit_trials:
        trial_dicts = [normalize_trial_overrides(trial) for trial in explicit_trials]
    else:
        raise ValueError("Sweep requires a parameter grid or explicit trial list.")

    plans: list[SweepTrialPlan] = []
    for trial_id, overrides in enumerate(trial_dicts, start=1):
        plans.append(
            SweepTrialPlan(
                trial_id=trial_id,
                overrides=overrides,
                run_name_suffix=trial_run_suffix(overrides),
            )
        )
    return plans


def rank_trials(
    results: list[SweepTrialResult],
    *,
    objective: str,
    direction: SweepDirection,
) -> list[SweepTrialResult]:
    reverse = direction == "maximize"

    def sort_key(result: SweepTrialResult) -> tuple[int, float]:
        value = result.objective_value
        if value is None:
            return (1, 0.0)
        return (0, -value if reverse else value)

    return sorted(results, key=sort_key)


def _parse_sweep_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    label = f"Sweep file {path}"
    if suffix == ".json":
        data = safe_json_loads(text, label=label)
    elif suffix in (".toml", ".tml"):
        from tomllib import loads as toml_loads

        data = toml_loads(text)
    else:
        raise ValueError(f"Unsupported sweep file extension {suffix!r} (use .json or .toml)")
    if not isinstance(data, dict):
        raise TypeError(f"Sweep file root must be an object/dict, got {type(data).__name__}")
    return dict(data)


def _coerce_grid(raw: object) -> dict[str, tuple[Any, ...]]:
    if not isinstance(raw, dict):
        raise TypeError("grid must be a mapping of parameter names to value lists")
    grid: dict[str, tuple[Any, ...]] = {}
    for key, values in raw.items():
        if not isinstance(key, str):
            raise TypeError("grid keys must be strings")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"grid[{key!r}] must be a non-empty list of values")
        grid[normalize_override_key(key)] = tuple(values)
    return grid


def _coerce_trials(raw: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise TypeError("trials must be a non-empty list of override mappings")
    trials: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError("each trial must be a mapping of override keys to values")
        trials.append(dict(entry))
    return tuple(trials)


def load_sweep_file(path: str | Path) -> tuple[SweepSpec, FrameworkConfig | None]:
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Sweep file not found: {raw_path}")
    payload = _parse_sweep_file(raw_path)

    meta = {key: payload.pop(key) for key in list(payload) if key in SWEEP_META_KEYS}

    base_config_raw = meta.get("base_config")
    if base_config_raw is not None and not isinstance(base_config_raw, str):
        raise TypeError("base_config must be a string path")
    base_config_path: str | None = base_config_raw

    objective = str(meta.get("objective", DEFAULT_OBJECTIVE))
    direction_raw = str(meta.get("direction", "maximize")).strip().lower()
    if direction_raw not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    direction: SweepDirection = direction_raw  # type: ignore[assignment]

    seed_raw = meta.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None

    grid_raw = meta.get("grid")
    trials_raw = meta.get("trials")
    grid = _coerce_grid(grid_raw) if grid_raw is not None else None
    trials = _coerce_trials(trials_raw) if trials_raw is not None else None

    base_config: FrameworkConfig | None = None
    if base_config_path:
        base_config = FrameworkConfig.from_file(base_config_path)
    if payload:
        base_config = FrameworkConfig.from_mapping(payload, base=base_config, strict=True)

    spec = SweepSpec(
        objective=objective,
        direction=direction,
        seed=seed,
        grid=grid,
        trials=trials,
        base_config_path=base_config_path,
    )
    return spec, base_config


__all__ = [
    "DEFAULT_OBJECTIVE",
    "SWEEP_META_KEYS",
    "SweepDirection",
    "SweepSpec",
    "SweepTrialPlan",
    "SweepTrialResult",
    "build_trial_plans",
    "load_sweep_file",
    "parse_vary_argument",
    "rank_trials",
    "trial_run_suffix",
]
