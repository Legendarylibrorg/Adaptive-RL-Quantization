from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def resolve_python_bin(root: Path) -> str:
    requested = os.environ.get("PYTHON_BIN")
    if requested:
        return requested
    candidate = venv_python_path(root / ".venv")
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def run(cmd: list[str], *, cwd: Path | None = None, timeout: float | None = None) -> None:
    """Thin ``subprocess.run`` wrapper that bubbles non-zero exit codes.

    ``timeout`` is opt-in: leave it unset for long-running commands like ``pip
    install -e .`` or ``unittest discover``; pass a value (seconds) for the
    fast diagnostic helpers (``git diff``, etc.) so a hung/slow filesystem
    cannot wedge the dev-tool indefinitely. ``subprocess.TimeoutExpired`` is
    surfaced to the caller unchanged.
    """
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        timeout=timeout,
    )


def bash_path() -> str | None:
    return shutil.which("bash")
