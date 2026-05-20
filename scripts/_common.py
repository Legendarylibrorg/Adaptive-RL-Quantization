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


def venv_console_command(venv_dir: Path, name: str) -> Path | None:
    """Return the installed console script in a venv, if present."""
    if os.name == "nt":
        scripts = venv_dir / "Scripts"
        for candidate in (scripts / f"{name}.exe", scripts / name):
            if candidate.is_file():
                return candidate
        return None
    candidate = venv_dir / "bin" / name
    return candidate if candidate.is_file() else None


def venv_cli_hint(
    venv_dir: Path,
    *,
    root: Path | None = None,
    name: str = "adaptive-rl-quant",
) -> str:
    """Display path for a venv console script (relative to repo root when possible)."""
    cli = venv_console_command(venv_dir, name)
    if cli is None:
        return name
    base = root if root is not None else repo_root()
    try:
        return str(cli.relative_to(base))
    except ValueError:
        return str(cli)


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
    if (
        not isinstance(cmd, list)
        or not cmd
        or not all(isinstance(item, str) and item for item in cmd)
    ):
        raise TypeError("cmd must be a non-empty list[str]")
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        timeout=timeout,
    )


def bash_path() -> str | None:
    return shutil.which("bash")
