from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TextIO

# Local-use bounds (stdlib only): avoid accidental multi-GB reads on analysis / small JSON sidecars.
MAX_LOCAL_READ_BYTES = 256 << 20
MAX_JSONL_LINES = 2_000_000
# Single-line cap: one pathological line cannot exhaust memory before the line count limit trips.
MAX_JSONL_LINE_BYTES = 4 << 20


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

    def log(self, record: dict[str, Any]) -> None:
        # Re-open per append so temporary test directories and partially initialized
        # trainers do not keep Windows file handles alive past their cleanup scope.
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True))
            handle.write("\n")

    def close(self) -> None:
        return


class NullJsonlLogger:
    path: Path | None = None

    def log(self, record: dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


class NullJsonlLogger:
    path: Path | None = None

    def log(self, record: dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


def write_json(path: str | Path, payload: Any) -> None:
    def _write(handle: TextIO) -> None:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)

    _write_text_atomically(path, _write)


def write_text_file(path: str | Path, text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    def _write(handle: TextIO) -> None:
        handle.write(text)

    _write_text_atomically(path, _write)


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
            raw_len = len(line.encode("utf-8"))
            if raw_len > MAX_JSONL_LINE_BYTES:
                raise ValueError(
                    f"JSONL line exceeds byte limit ({MAX_JSONL_LINE_BYTES}): line {i + 1} in {source}"
                )
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return ["| " + " | ".join(headers) + " |", sep] + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _write_text_atomically(path: str | Path, writer: Callable[[TextIO], None]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
