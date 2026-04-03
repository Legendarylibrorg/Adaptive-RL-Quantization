from __future__ import annotations

import atexit
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Local-use bounds (stdlib only): avoid accidental multi-GB reads on analysis / small JSON sidecars.
MAX_LOCAL_READ_BYTES = 256 << 20
MAX_JSONL_LINES = 2_000_000


def enforce_local_read_limit(path: str | Path, *, label: str = "File") -> None:
    p = Path(path)
    if p.is_file() and p.stat().st_size > MAX_LOCAL_READ_BYTES:
        raise ValueError(f"{label} exceeds local read limit ({MAX_LOCAL_READ_BYTES} bytes): {p}")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class JsonlLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        atexit.register(self.close)

    def log(self, record: dict[str, Any]) -> None:
        if self._handle.closed:
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._handle.write(json.dumps(to_jsonable(record), sort_keys=True))
        self._handle.write("\n")

    def close(self) -> None:
        if hasattr(self, "_handle") and not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def write_json(path: str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    enforce_local_read_limit(source, label="JSONL")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= MAX_JSONL_LINES:
                raise ValueError(f"JSONL exceeds local line limit ({MAX_JSONL_LINES}): {source}")
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return ["| " + " | ".join(headers) + " |", sep] + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
