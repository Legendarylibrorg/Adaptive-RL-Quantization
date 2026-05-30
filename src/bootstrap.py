"""Shared ``sys.path`` bootstrap for source checkouts (runners, analysis CLI, tests)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_paths(repo_root: Path) -> None:
    """Prepend ``src/`` and the repo root so ``adaptive_quant``, ``analysis``, and ``config`` import."""
    root = repo_root.resolve()
    src = root / "src"
    for entry in (str(src), str(root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
