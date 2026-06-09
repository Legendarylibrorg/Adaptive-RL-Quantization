"""Load preference datasets for DPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_quant.logging_utils import read_json


def load_preference_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load prompt/chosen/rejected examples from JSON or JSONL."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Preference dataset not found: {source}")

    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(_normalize_preference_row(row, label=f"{source}:{line_no}"))
        return rows

    payload = read_json(source, label="preference dataset")
    if isinstance(payload, list):
        return [_normalize_preference_row(row, label=source) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return [
            _normalize_preference_row(row, label=source)
            for row in payload["examples"]
        ]
    raise ValueError(
        f"Unsupported preference dataset shape in {source}; "
        "expected a JSON list or {{'examples': [...]}}."
    )


def _normalize_preference_row(row: Any, *, label: str | Path) -> dict[str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"Preference row in {label} must be an object.")
    missing = [key for key in ("prompt", "chosen", "rejected") if key not in row]
    if missing:
        raise ValueError(f"Preference row in {label} missing fields: {missing}")
    return {
        "prompt": str(row["prompt"]),
        "chosen": str(row["chosen"]),
        "rejected": str(row["rejected"]),
    }
