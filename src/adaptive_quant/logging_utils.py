from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import tempfile
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO, cast

# Local-use bounds (stdlib only): avoid accidental multi-GB reads on analysis / small JSON sidecars.
MAX_LOCAL_READ_BYTES = 256 << 20
MAX_JSONL_LINES = 2_000_000
# Single-line cap: one pathological line cannot exhaust memory before the line count limit trips.
MAX_JSONL_LINE_BYTES = 4 << 20
# After ``json.loads`` / ``tomllib``: block hostile nesting and huge container graphs (DoS / gadget hardening).
MAX_JSON_NESTING_DEPTH = 64
MAX_JSON_CONTAINER_NODES = 500_000
MAX_JSON_OBJECT_KEYS = 10_000
MAX_JSON_ARRAY_LENGTH = 2_000_000
# Per UTF-8 string segment (dict key or value); aligns with JSONL single-line budget.
MAX_JSON_STRING_BYTES = 4 << 20
# Sum of all string key/value UTF-8 bytes in the tree (memory-DoS guard for wide short strings).
MAX_JSON_AGGREGATE_STRING_BYTES = 96 << 20
_JSONL_INTEGRITY_CHAIN_ENV = "ADAPTIVE_RL_JSONL_INTEGRITY_CHAIN"


def enforce_local_read_limit(path: str | Path, *, label: str = "File") -> None:
    p = Path(path)
    if p.is_file() and p.stat().st_size > MAX_LOCAL_READ_BYTES:
        raise ValueError(f"{label} exceeds local read limit ({MAX_LOCAL_READ_BYTES} bytes): {p}")


def enforce_safe_parsed_json(value: Any, *, label: str = "JSON") -> None:
    """Reject pathological dict/list shapes from untrusted files (nested / wide JSON-like trees).

    Intended for artifacts that may come from third parties or compromised experiment dirs: configs,
    JSONL telemetry, checkpoint sidecars, external quality tables, and analysis inputs. Uses an
    iterative walk so adversarial depth does not rely on Python recursion limits. Also bounds
    string payloads, aggregate UTF-8 from all strings, rejects non-finite floats, and rejects
    ambiguous leaf types (e.g. bytes) after ``to_jsonable`` conversion for writes.
    """
    max_depth = MAX_JSON_NESTING_DEPTH
    max_nodes = MAX_JSON_CONTAINER_NODES
    max_keys = MAX_JSON_OBJECT_KEYS
    max_array = MAX_JSON_ARRAY_LENGTH
    max_str = MAX_JSON_STRING_BYTES
    max_str_sum = MAX_JSON_AGGREGATE_STRING_BYTES

    str_total = 0

    def consume_str(s: str) -> None:
        nonlocal str_total
        raw = len(s.encode("utf-8"))
        if raw > max_str:
            raise ValueError(
                f"{label}: string segment exceeds byte limit ({max_str}); "
                "refusing hostile or pathological JSON-like data."
            )
        str_total += raw
        if str_total > max_str_sum:
            raise ValueError(
                f"{label}: aggregate string UTF-8 bytes exceed limit ({max_str_sum}); "
                "refusing hostile or pathological JSON-like data."
            )

    def check_leaf(v: Any) -> None:
        if isinstance(v, str):
            consume_str(v)
        elif isinstance(v, bool):
            # Must precede ``int`` because ``bool`` subclasses ``int``.
            return
        elif isinstance(v, int):
            return
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(
                    f"{label}: non-finite float in JSON-like data; "
                    "refusing hostile or non-standard numeric payloads."
                )
        elif v is None:
            return
        elif isinstance(v, datetime.date):
            # ``tomllib`` (config files) can attach dates; ``json.loads`` never produces these.
            return
        else:
            raise TypeError(
                f"{label}: unsupported JSON-like leaf type {type(v).__name__}; "
                "refusing ambiguous serialized data."
            )

    if not isinstance(value, (dict, list)):
        check_leaf(value)
        return

    q: deque[tuple[Any, int]] = deque([(value, 0)])
    nodes = 0
    while q:
        obj, depth = q.popleft()
        if isinstance(obj, dict):
            if depth >= max_depth:
                raise ValueError(
                    f"{label}: exceeds maximum nesting depth ({max_depth}); "
                    "refusing hostile or pathological JSON-like data."
                )
            n_keys = len(obj)
            if n_keys > max_keys:
                raise ValueError(
                    f"{label}: object has {n_keys} keys (limit {max_keys}); "
                    "refusing hostile or pathological JSON-like data."
                )
            nodes += 1
            if nodes > max_nodes:
                raise ValueError(
                    f"{label}: exceeds maximum container count ({max_nodes}); "
                    "refusing hostile or pathological JSON-like data."
                )
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise TypeError(
                        f"{label}: object keys must be str, got {type(k).__name__}; "
                        "refusing ambiguous serialized data."
                    )
                consume_str(k)
                if isinstance(v, (dict, list)):
                    q.append((v, depth + 1))
                else:
                    check_leaf(v)
        elif isinstance(obj, list):
            if depth >= max_depth:
                raise ValueError(
                    f"{label}: exceeds maximum nesting depth ({max_depth}); "
                    "refusing hostile or pathological JSON-like data."
                )
            n_el = len(obj)
            if n_el > max_array:
                raise ValueError(
                    f"{label}: array length {n_el} exceeds limit ({max_array}); "
                    "refusing hostile or pathological JSON-like data."
                )
            nodes += 1
            if nodes > max_nodes:
                raise ValueError(
                    f"{label}: exceeds maximum container count ({max_nodes}); "
                    "refusing hostile or pathological JSON-like data."
                )
            for v in obj:
                if isinstance(v, (dict, list)):
                    q.append((v, depth + 1))
                else:
                    check_leaf(v)


def safe_json_loads(data: str, *, label: str = "JSON") -> Any:
    """Parse JSON from a string and apply :func:`enforce_safe_parsed_json`."""
    parsed = json.loads(data)
    enforce_safe_parsed_json(parsed, label=label)
    return parsed


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(cast(Any, value)).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class JsonlLogger:
    def __init__(self, path: str, *, buffered: bool = False, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffered = bool(buffered)
        self._flush_every = max(1, int(flush_every))
        self._handle: TextIO | None = None
        self._pending = 0
        self._integrity_chain = _jsonl_integrity_chain_enabled()
        self._prev_integrity_hash = ""

    def log(self, record: dict[str, Any]) -> None:
        # Bound outbound records so a poisoned in-process dict cannot write multi-GB lines.
        safe = to_jsonable(record)
        enforce_safe_parsed_json(safe, label="JsonlLogger record")
        if self._integrity_chain:
            safe = dict(safe)
            safe["_integrity_prev"] = self._prev_integrity_hash
            line_core = json.dumps(safe, sort_keys=True, separators=(",", ":"))
            line_hash = hashlib.sha256(line_core.encode("utf-8")).hexdigest()
            safe["_integrity_hash"] = line_hash
            self._prev_integrity_hash = line_hash
        payload = json.dumps(safe, sort_keys=True) + "\n"
        line_bytes = len(payload.encode("utf-8"))
        if line_bytes > MAX_JSONL_LINE_BYTES:
            raise ValueError(
                f"JsonlLogger record serializes to {line_bytes} bytes (limit {MAX_JSONL_LINE_BYTES})."
            )
        if not self._buffered:
            # Re-open per append so temporary test directories and partially initialized
            # trainers do not keep Windows file handles alive past their cleanup scope.
            with self.path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(payload)
            return

        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("a", encoding="utf-8")
            self._pending = 0

        self._handle.write(payload)
        self._pending += 1
        if self._pending >= self._flush_every:
            self._handle.flush()
            self._pending = 0

    def close(self) -> None:
        if self._handle is not None:
            try:
                if self._pending:
                    self._handle.flush()
            finally:
                self._handle.close()
            self._pending = 0
        return


class NullJsonlLogger:
    path: Path | None = None

    def log(self, record: dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


def write_json(path: str | Path, payload: Any) -> None:
    safe = to_jsonable(payload)
    enforce_safe_parsed_json(safe, label="write_json")

    def _write(handle: TextIO) -> None:
        json.dump(safe, handle, indent=2, sort_keys=True)

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
                line_label = f"JSONL {source} line {i + 1}"
                records.append(safe_json_loads(line, label=line_label))
    return records


def read_json(path: str | Path, *, label: str = "JSON") -> Any:
    source = Path(path)
    enforce_local_read_limit(source, label=label)
    return safe_json_loads(source.read_text(encoding="utf-8"), label=label)


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return ["| " + " | ".join(headers) + " |", sep] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def _jsonl_integrity_chain_enabled() -> bool:
    raw = os.environ.get(_JSONL_INTEGRITY_CHAIN_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
