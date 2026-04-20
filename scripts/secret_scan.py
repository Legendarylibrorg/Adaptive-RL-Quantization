#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from _common import repo_root

PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key_block",
        re.compile(r"BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PGP)\s+PRIVATE\s+KEY"),
        r"BEGIN[[:space:]]+(RSA|OPENSSH|EC|DSA|PGP)[[:space:]]+PRIVATE[[:space:]]+KEY",
    ),
    (
        "generic_private_key_block",
        re.compile(r"BEGIN\s+PRIVATE\s+KEY"),
        r"BEGIN[[:space:]]+PRIVATE[[:space:]]+KEY",
    ),
    ("github_token", re.compile(r"ghp_[0-9A-Za-z]{36}"), r"ghp_[0-9A-Za-z]{36}"),
    ("github_pat", re.compile(r"github_pat_[0-9A-Za-z_]{20,}"), r"github_pat_[0-9A-Za-z_]{20,}"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), r"AKIA[0-9A-Z]{16}"),
    (
        "slack_webhook",
        re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
        r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+",
    ),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z\-_]{35}"), r"AIza[0-9A-Za-z\-_]{35}"),
)


def scan_tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    matches: list[str] = []
    for raw_path in completed.stdout.split(b"\x00"):
        if not raw_path:
            continue
        path = root / raw_path.decode("utf-8", errors="ignore")
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\x00" in content:
            continue
        text = content.decode("utf-8", errors="ignore")
        rel = path.relative_to(root).as_posix()
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern, _history_pattern in PATTERNS:
                if pattern.search(line):
                    matches.append(f"{rel}:{line_no}: {label}: {line}")
    return matches


def scan_history(root: Path) -> list[str]:
    matches: list[str] = []
    for label, _pattern, history_pattern in PATTERNS:
        completed = subprocess.run(
            ["git", "log", "--all", "--format=%H", "-G", history_pattern],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        commits = [commit for commit in completed.stdout.splitlines() if commit]
        for commit in commits:
            matches.append(f"history: {label}: {commit}")
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Secret scan across tracked files, with optional reachable-history checks using Python + git."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan reachable git history for high-signal secret patterns (requires full clone history).",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    matches = scan_tracked_files(root)
    history_matches = scan_history(root) if args.history else []
    if matches or history_matches:
        print("== Possible secret matched pattern (redact before sharing logs): ==", file=sys.stderr)
        for line in matches:
            print(line, file=sys.stderr)
        for line in history_matches:
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print("secret_scan.py: failing — remove or rotate leaked material, then re-run.", file=sys.stderr)
        return 1
    if args.history:
        print("OK: secret_scan.py — no high-signal patterns in tracked files or reachable git history.")
    else:
        print("OK: secret_scan.py — no high-signal patterns in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
