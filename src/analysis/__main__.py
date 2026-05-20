"""``python -m analysis <command> ...`` — unified analysis CLI."""

from __future__ import annotations

import sys

from analysis.analyzers import CLI_COMMANDS, run_cli
from analysis.shim_support import ensure_src_on_path


def _usage() -> str:
    commands = ", ".join(sorted(CLI_COMMANDS))
    return (
        "Usage: python -m analysis <command> <log_or_history> <output_dir> [--phase ...]\n"
        f"Commands: {commands}"
    )


def main() -> None:
    ensure_src_on_path()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)
    command = sys.argv[1]
    if command not in CLI_COMMANDS:
        raise SystemExit(f"Unknown analysis command: {command!r}\n\n{_usage()}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    run_cli(command)


if __name__ == "__main__":
    main()
