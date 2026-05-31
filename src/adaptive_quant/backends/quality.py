from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import load_jsonl, read_json
from adaptive_quant.types import BackendMetricDict, EpisodeState


class ExternalQualityScores:
    """Prompt-level external quality scores loaded from JSON/JSONL."""

    def __init__(self, scores_by_prompt_id: Mapping[str, float], *, metric: str, path: str) -> None:
        self.scores_by_prompt_id = dict(scores_by_prompt_id)
        self.metric = metric
        self.path = path

    @classmethod
    def from_config(cls, config: FrameworkConfig) -> ExternalQualityScores | None:
        path = config.external_quality_path
        if not path:
            return None
        metric = str(config.external_quality_metric or "perplexity")
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
            raise ValueError(
                f"External quality file has no finite `{metric}` scores keyed by prompt_id: {source}"
            )
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
