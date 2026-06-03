"""Shared ``sys.path`` bootstrap for repo-root ``run_*.py`` shims."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bootstrap import ensure_repo_paths

ensure_repo_paths(_REPO_ROOT)


def _cli_module_for_script(caller_file: str) -> str:
    script = Path(caller_file).name
    if not script.startswith("run_") or not script.endswith(".py"):
        raise ValueError(f"Unknown runner script {script!r}; expected run_<command>.py")
    module = f"adaptive_quant.cli.{script.removeprefix('run_').removesuffix('.py')}"
    if find_spec(module) is None:
        raise ValueError(f"Unknown runner script {script!r}; expected matching module {module!r}")
    return module


def main_for_script(caller_file: str, *, attr: str = "main") -> Callable[..., Any]:
    """Return the CLI entrypoint for a repo-root ``run_*.py`` file (for imports and tests)."""
    return getattr(import_module(_cli_module_for_script(caller_file)), attr)


def run_script_main(caller_file: str, *, attr: str = "main") -> None:
    """Run the CLI ``main`` for a repo-root ``run_*.py`` shim."""
    try:
        main_for_script(caller_file, attr=attr)()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
