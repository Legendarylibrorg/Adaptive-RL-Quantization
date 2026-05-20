"""Shared ``sys.path`` bootstrap for repo-root ``run_*.py`` shims."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

_SRC = Path(__file__).resolve().parent / "src"

# ``run_<name>.py`` filename → ``adaptive_quant.cli.*`` module providing ``main``.
_RUN_SCRIPT_MODULES: dict[str, str] = {
    "run_calibrate_llama_cpp.py": "adaptive_quant.cli.calibrate_llama_cpp",
    "run_moe_research.py": "adaptive_quant.cli.moe_research",
    "run_multiseed.py": "adaptive_quant.cli.multiseed",
    "run_online_learning.py": "adaptive_quant.cli.online_learning",
    "run_pytorch.py": "adaptive_quant.cli.pytorch",
    "run_research.py": "adaptive_quant.cli.research",
    "run_route_learning.py": "adaptive_quant.cli.route_learning",
}


def ensure_src_on_path() -> None:
    root = str(_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_main(module: str, *, attr: str = "main") -> Callable[..., Any]:
    ensure_src_on_path()
    return getattr(import_module(module), attr)


def main_for_script(caller_file: str, *, attr: str = "main") -> Callable[..., Any]:
    """Return the CLI entrypoint for a repo-root ``run_*.py`` file (for imports and tests)."""
    script = Path(caller_file).name
    module = _RUN_SCRIPT_MODULES.get(script)
    if module is None:
        known = ", ".join(sorted(_RUN_SCRIPT_MODULES))
        raise ValueError(f"Unknown runner script {script!r}; expected one of: {known}")
    return load_main(module, attr=attr)


def run_script_main(caller_file: str, *, attr: str = "main") -> None:
    """Run the CLI ``main`` for a repo-root ``run_*.py`` shim."""
    script = Path(caller_file).name
    module = _RUN_SCRIPT_MODULES.get(script)
    if module is None:
        known = ", ".join(sorted(_RUN_SCRIPT_MODULES))
        raise SystemExit(f"Unknown runner script {script!r}; expected one of: {known}")
    load_main(module, attr=attr)()
