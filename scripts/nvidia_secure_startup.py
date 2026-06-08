#!/usr/bin/env python3
"""NVIDIA secure-boundary gate for shell startup scripts (setup, 4090 pipeline, CUDA install)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adaptive_quant.nvidia_secure_boundary import (  # noqa: E402
    enforce_nvidia_secure_boundary,
    nvidia_boundary_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce the NVIDIA secure-boundary policy on Linux hosts with a visible GPU. "
            "No-op on macOS/Windows and in CI."
        )
    )
    parser.add_argument(
        "--context",
        default="startup",
        help="Short label for logs and error messages (default: startup).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print boundary report JSON fields and exit without enforcing.",
    )
    args = parser.parse_args(argv)

    if args.report_only:
        report = nvidia_boundary_report()
        for key, value in report.items():
            print(f"{key}: {value}")
        return 0

    report = enforce_nvidia_secure_boundary(context=args.context)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
