#!/usr/bin/env python3
"""Run hardware-aware setup tests (curated unittest subset)."""
from __future__ import annotations

import argparse
import sys

from _common import repo_root, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-install setup tests: core config/CLI modules on every host, "
            "plus torch and NVIDIA modules when applicable."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full unittest discover suite (contributor/CI parity).",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Skip torch trainer modules even when PyTorch is installed.",
    )
    parser.add_argument(
        "--no-nvidia",
        action="store_true",
        help="Skip NVIDIA boundary/install modules even on Linux + NVIDIA hosts.",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Interpreter to run unittest (default: current).",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    sys.path.insert(0, str(root / "src"))

    if args.full:
        cmd = [
            args.python_bin,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-t",
            ".",
            "-q",
        ]
    else:
        from adaptive_quant.setup_tests import resolve_setup_test_modules, unittest_command

        modules = resolve_setup_test_modules(
            include_torch=False if args.no_torch else None,
            include_nvidia=False if args.no_nvidia else None,
        )
        print("Setup tests:", ", ".join(modules))
        cmd = unittest_command(args.python_bin, modules, quiet=True)

    run(cmd, cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
