#!/usr/bin/env python3
"""Regenerate hash-pinned requirement lockfiles under requirements/.

Requires: pip install pip-tools (not a runtime project dependency).

Usage (from repo root):
  python scripts/compile_locked_requirements.py
"""

from __future__ import annotations

import sys

from _common import repo_root, run


def main() -> int:
    root = repo_root()
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
            "--output-file",
            "requirements/audit.txt",
            "requirements/audit.in",
        ],
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--generate-hashes",
            "--output-file",
            "requirements/pytorch-cpu.txt",
            "requirements/pytorch.in",
        ],
    ]
    for cmd in commands:
        print("+", " ".join(cmd), flush=True)
        run(cmd, cwd=root)
    print("OK: compile_locked_requirements.py finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
