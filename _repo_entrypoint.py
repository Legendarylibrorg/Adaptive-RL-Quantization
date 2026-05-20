"""Shared ``sys.path`` bootstrap for repo-root ``run_*.py`` shims."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

_SRC = Path(__file__).resolve().parent / "src"


def ensure_src_on_path() -> None:
    root = str(_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_main(module: str, *, attr: str = "main") -> Callable[..., Any]:
    ensure_src_on_path()
    return getattr(import_module(module), attr)


def run_cli(module: str, *, attr: str = "main") -> None:
    load_main(module, attr=attr)()
