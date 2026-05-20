"""Shared bootstrap for analysis CLIs (legacy script paths and ``python -m analysis``)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_shim_main(caller_file: str) -> None:
    """Run the analysis CLI keyed by ``Path(caller_file).stem``."""
    ensure_src_on_path()
    from analysis.analyzers import run_cli

    run_cli(Path(caller_file).stem)
