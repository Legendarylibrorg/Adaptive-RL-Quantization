"""Bootstrap for legacy ``python src/analysis/<name>.py`` invocations."""

from __future__ import annotations

import sys
from pathlib import Path


def run_legacy_main(caller_file: str) -> None:
    src = Path(caller_file).resolve().parent.parent
    root = str(src)
    if root not in sys.path:
        sys.path.insert(0, root)
    from analysis.shim_support import run_shim_main

    run_shim_main(caller_file)
