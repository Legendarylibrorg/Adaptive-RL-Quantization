"""Shared ``if __name__ == '__main__'`` entry for per-topic analysis script shims."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def run_as_main(caller_file: str) -> None:
    bootstrap = Path(caller_file).resolve().parent / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_analysis_bootstrap", bootstrap)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load analysis bootstrap from {bootstrap}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_legacy_main(caller_file)
