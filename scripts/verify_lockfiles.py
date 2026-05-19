#!/usr/bin/env python3
"""Verify pip-compile lockfiles under requirements/ include inline hashes for every pin."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _common import repo_root

_PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\[\]-]*==")
_HASH_CONTINUATION_RE = re.compile(r"^\s+--hash=sha256:")
_ALLOWED_OPTION_PREFIXES = (
    "--extra-index-url ",
    "--index-url ",
    "--find-links ",
    "--trusted-host ",
    "-i ",
    "-f ",
)


def _verify_lockfile(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            index += 1
            continue
        if _HASH_CONTINUATION_RE.match(line):
            index += 1
            continue
        if not _PIN_RE.match(stripped):
            if stripped.startswith(("-", "--")):
                if not any(stripped.startswith(prefix) for prefix in _ALLOWED_OPTION_PREFIXES):
                    errors.append(f"{path}:{index + 1}: unexpected option line in lockfile")
            index += 1
            continue
        if "\\" not in line:
            errors.append(f"{path}:{index + 1}: package pin missing trailing backslash")
            index += 1
            continue
        hash_index = index + 1
        if hash_index >= len(lines) or not _HASH_CONTINUATION_RE.match(lines[hash_index]):
            errors.append(f"{path}:{index + 1}: package pin missing --hash=sha256 line")
        index += 1
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    root = repo_root()
    lockfiles = sorted(root.glob("requirements/*.txt"))
    lockfiles = [p for p in lockfiles if p.name != "ci.txt"]
    if not lockfiles:
        print("verify_lockfiles.py: no requirements/*.txt lockfiles found.", file=sys.stderr)
        return 1
    errors: list[str] = []
    for path in lockfiles:
        errors.extend(_verify_lockfile(path))
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    print(f"OK: verify_lockfiles.py — {len(lockfiles)} lockfile(s) include inline hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
