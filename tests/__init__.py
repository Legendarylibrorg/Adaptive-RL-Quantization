"""Test package bootstrap: allow ``python3 -m unittest discover -s tests -t .`` without ``pip install -e .``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bootstrap import ensure_repo_paths

ensure_repo_paths(_REPO_ROOT)
