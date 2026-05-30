"""``python -m analysis <command> ...`` — unified post-hoc analysis CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bootstrap import ensure_repo_paths

ensure_repo_paths(_SRC.parent)


def _usage(commands: frozenset[str]) -> str:
    names = ", ".join(sorted(commands))
    return (
        "Usage: python -m analysis <command> <log_or_history> <output_dir> [--phase ...]\n"
        f"Commands: {names}"
    )


def main() -> None:
    from analysis.analyzers import CLI_COMMANDS, run_cli

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage(CLI_COMMANDS))
        raise SystemExit(0)
    command = sys.argv[1]
    if command not in CLI_COMMANDS:
        raise SystemExit(f"Unknown analysis command: {command!r}\n\n{_usage(CLI_COMMANDS)}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    run_cli(command)


if __name__ == "__main__":
    main()
