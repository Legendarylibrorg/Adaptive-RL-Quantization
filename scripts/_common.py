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


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def bash_path() -> str | None:
    return shutil.which("bash")
