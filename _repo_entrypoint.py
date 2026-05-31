"""Shared ``sys.path`` bootstrap for repo-root ``run_*.py`` shims."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bootstrap import ensure_repo_paths

ensure_repo_paths(_REPO_ROOT)

# ``run_<name>.py`` filename → ``adaptive_quant.cli.*`` module providing ``main``.
_RUN_SCRIPT_MODULES: dict[str, str] = {
    "run_calibrate_llama_cpp.py": "adaptive_quant.cli.calibrate_llama_cpp",
    "run_moe_research.py": "adaptive_quant.cli.moe_research",
    "run_multiseed.py": "adaptive_quant.cli.multiseed",
    "run_sweep.py": "adaptive_quant.cli.sweep",
    "run_online_learning.py": "adaptive_quant.cli.online_learning",
    "run_pytorch.py": "adaptive_quant.cli.pytorch",
    "run_research.py": "adaptive_quant.cli.research",
    "run_replay.py": "adaptive_quant.cli.replay",
    "run_route_learning.py": "adaptive_quant.cli.route_learning",
}


def _cli_module_for_script(caller_file: str) -> str:
    script = Path(caller_file).name
    module = _RUN_SCRIPT_MODULES.get(script)
    if module is None:
        known = ", ".join(sorted(_RUN_SCRIPT_MODULES))
        raise ValueError(f"Unknown runner script {script!r}; expected one of: {known}")
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
