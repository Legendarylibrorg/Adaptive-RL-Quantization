"""Repo-root `sys.path` fix for `python analysis/<script>.py`."""

from __future__ import annotations

import sys
from pathlib import Path


def _prepare(caller_file: str) -> None:
    root = str(Path(caller_file).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_cli_main(caller_file: str, key: str) -> None:
    _prepare(caller_file)
    from analysis.analyzers import run_cli

    run_cli(key)


def dispatch_cli(caller_file: str, key: str) -> None:
    run_cli_main(caller_file, key)
