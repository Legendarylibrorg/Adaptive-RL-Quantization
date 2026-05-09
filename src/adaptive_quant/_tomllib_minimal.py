"""
Compatibility shim for Python < 3.11.

The project targets Python >= 3.11 (see pyproject.toml), where `tomllib` is in the stdlib.
Some development/test environments still run older Python versions; in that case, importing
`tomllib` fails and blocks even basic config loading and script tests.

This module provides a small, self-contained TOML reader that supports the subset of TOML
used by this repository (including `pyproject.toml` and simple config TOML files).
"""

from __future__ import annotations

import io
import re
from typing import Any, BinaryIO, TextIO

_TABLE_RE = re.compile(r"^\[(?P<name>[A-Za-z0-9_.-]+)\]\s*$")


def load(fp: BinaryIO, /) -> dict[str, Any]:
    data = fp.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    return loads(text)


def loads(s: str, /) -> dict[str, Any]:
    parser = _TomlParser(s)
    return parser.parse()


class _TomlParser:
    def __init__(self, text: str) -> None:
        self._lines = text.splitlines()
        self._root: dict[str, Any] = {}
        self._current: dict[str, Any] = self._root

    def parse(self) -> dict[str, Any]:
        pending_array_key: str | None = None
        pending_array_items: list[Any] | None = None

        for raw in self._lines:
            line = _strip_comment(raw).strip()
            if not line:
                continue

            # Multiline array continuation.
            if pending_array_key is not None:
                assert pending_array_items is not None
                pending_array_items.extend(_parse_array_items(line))
                if "]" in line:
                    # finalize
                    self._current[pending_array_key] = pending_array_items
                    pending_array_key = None
                    pending_array_items = None
                continue

            m = _TABLE_RE.match(line)
            if m:
                self._current = _ensure_table(self._root, m.group("name"))
                continue

            if "=" not in line:
                raise ValueError(f"Invalid TOML line (expected key = value): {raw!r}")
            key, value_raw = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid TOML key: {raw!r}")
            value_raw = value_raw.strip()

            # Start of a multiline array?
            if value_raw.startswith("[") and "]" not in value_raw:
                pending_array_key = key
                pending_array_items = _parse_array_items(value_raw)
                continue

            self._current[key] = _parse_value(value_raw)

        if pending_array_key is not None:
            raise ValueError("Unterminated TOML array")
        return self._root


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    out_chars: list[str] = []
    for ch in line:
        if escaped:
            out_chars.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out_chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out_chars.append(ch)
            continue
        if ch == "#" and not in_string:
            break
        out_chars.append(ch)
    return "".join(out_chars)


def _ensure_table(root: dict[str, Any], dotted_name: str) -> dict[str, Any]:
    current: dict[str, Any] = root
    for part in dotted_name.split("."):
        if part not in current:
            current[part] = {}
        node = current[part]
        if not isinstance(node, dict):
            raise ValueError(f"Cannot create table {dotted_name!r}: {part!r} is not a table")
        current = node
    return current


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty TOML value")

    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return _parse_string(raw)
    if raw.startswith("[") and raw.endswith("]"):
        return _parse_array(raw)
    if raw.startswith("{") and raw.endswith("}"):
        return _parse_inline_table(raw)
    if raw in ("true", "false"):
        return raw == "true"

    # int / float (minimal)
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        # As a pragmatic fallback, keep as string (covers bare words used rarely here).
        return raw


def _parse_string(raw: str) -> str:
    # Only handles basic double-quoted strings used in this repo.
    assert raw.startswith('"') and raw.endswith('"')
    inner = raw[1:-1]
    return bytes(inner, "utf-8").decode("unicode_escape")


def _parse_array(raw: str) -> list[Any]:
    items = _parse_array_items(raw)
    return items


def _parse_array_items(raw: str) -> list[Any]:
    # Parse a (possibly partial) array expression.
    text = raw.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]

    items: list[Any] = []
    token = ""
    depth_inline = 0
    in_string = False
    escaped = False

    def flush() -> None:
        nonlocal token
        t = token.strip()
        token = ""
        if not t:
            return
        items.append(_parse_value(t))

    for ch in text:
        if escaped:
            token += ch
            escaped = False
            continue
        if in_string and ch == "\\":
            token += ch
            escaped = True
            continue
        if ch == '"':
            token += ch
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                depth_inline += 1
            elif ch in "}]":
                depth_inline = max(0, depth_inline - 1)
            elif ch == "," and depth_inline == 0:
                flush()
                continue
        token += ch
    flush()
    return items


def _parse_inline_key(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return _parse_string(raw)
    return raw


def _parse_inline_table(raw: str) -> dict[str, Any]:
    # Supports { key = "value", key2 = 1 } (no nested inline tables required here).
    text = raw.strip()[1:-1].strip()
    if not text:
        return {}
    parts = _split_top_level_commas(text)
    out: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid inline table entry: {part!r}")
        k, v = part.split("=", 1)
        key = _parse_inline_key(k)
        out[key] = _parse_value(v.strip())
    return out


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    token = ""
    in_string = False
    escaped = False
    depth = 0

    def flush() -> None:
        nonlocal token
        t = token.strip()
        token = ""
        if t:
            parts.append(t)

    for ch in text:
        if escaped:
            token += ch
            escaped = False
            continue
        if in_string and ch == "\\":
            token += ch
            escaped = True
            continue
        if ch == '"':
            token += ch
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                flush()
                continue
        token += ch
    flush()
    return parts

