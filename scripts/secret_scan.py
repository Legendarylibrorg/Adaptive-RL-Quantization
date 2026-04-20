#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from _common import repo_root

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PGP)\s+PRIVATE\s+KEY")),
    ("generic_private_key_block", re.compile(r"BEGIN\s+PRIVATE\s+KEY")),
    ("github_token", re.compile(r"ghp_[0-9A-Za-z]{36}")),
    ("github_pat", re.compile(r"github_pat_[0-9A-Za-z_]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
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
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    matches.append(f"{rel}:{line_no}: {label}: {line}")
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight secret scan across tracked text files using Python + git."
    )
    parser.parse_args(argv)

    root = repo_root()
    matches = scan_tracked_files(root)
    if matches:
        print("== Possible secret matched pattern (redact before sharing logs): ==", file=sys.stderr)
        for line in matches:
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print("secret_scan.py: failing — remove or rotate leaked material, then re-run.", file=sys.stderr)
        return 1
    print("OK: secret_scan.py — no high-signal patterns in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
