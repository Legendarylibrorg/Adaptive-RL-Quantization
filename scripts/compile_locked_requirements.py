#!/usr/bin/env python3
"""Regenerate hash-pinned requirement lockfiles under requirements/.

Uses ``uv pip compile`` when ``uv`` is on PATH (recommended on macOS for the Linux
``pytorch-cpu.txt`` lock), otherwise ``pip-tools`` (``pip install pip-tools``).

Usage (from repo root):
  python scripts/compile_locked_requirements.py
"""

from __future__ import annotations

import shutil
import sys

from _common import repo_root, run

_PYTORCH_EXTRA_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"
_PYTORCH_LOCK = "requirements/pytorch-cpu.txt"


def _ensure_pytorch_extra_index(root) -> None:
    path = root / _PYTORCH_LOCK
    text = path.read_text(encoding="utf-8")
    if _PYTORCH_EXTRA_INDEX in text:
        return
    lines = text.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (
        not lines[insert_at].strip() or lines[insert_at].startswith("#")
    ):
        insert_at += 1
    lines[insert_at:insert_at] = ["", _PYTORCH_EXTRA_INDEX, ""]
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def main() -> int:
    root = repo_root()
    if shutil.which("uv"):
        commands = [
            [
                "uv",
                "pip",
                "compile",
                "--generate-hashes",
                "--extra",
                "dev",
                "--output-file",
                "requirements/dev.txt",
                "pyproject.toml",
            ],
            [
                "uv",
                "pip",
                "compile",
                "--generate-hashes",
                "--python-platform",
                "linux",
                "--index-strategy",
                "unsafe-best-match",
                "--output-file",
                _PYTORCH_LOCK,
                "requirements/pytorch.in",
            ],
        ]
    else:
        commands = [
            [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                "--generate-hashes",
                "--extra",
                "dev",
                "--output-file",
                "requirements/dev.txt",
                "pyproject.toml",
            ],
            [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                "--generate-hashes",
                "--allow-unsafe",
                "--output-file",
                _PYTORCH_LOCK,
                "requirements/pytorch.in",
            ],
        ]
    for cmd in commands:
        print("+", " ".join(cmd), flush=True)
        run(cmd, cwd=root)
    _ensure_pytorch_extra_index(root)
    print("OK: compile_locked_requirements.py finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
