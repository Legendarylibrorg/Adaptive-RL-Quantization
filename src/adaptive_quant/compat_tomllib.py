"""TOML parsing for config files (stdlib ``tomllib``; requires Python 3.11+ per ``pyproject.toml``)."""

from __future__ import annotations

from tomllib import load, loads

__all__ = ["load", "loads"]
