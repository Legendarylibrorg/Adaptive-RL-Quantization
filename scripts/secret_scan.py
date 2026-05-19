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
    ("github_oauth", re.compile(r"gho_[0-9A-Za-z]{36}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("gcp_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack_token", re.compile(r"xox[abprs]-[0-9A-Za-z-]{10,}")),
    ("gitlab_pat", re.compile(r"glpat-[0-9A-Za-z_-]{20,}")),
    ("pypi_token", re.compile(r"pypi-AgE[0-9A-Za-z_-]{40,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-(?:api|admin)\d+-[A-Za-z0-9_-]{20,}")),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b")),
)


# Files larger than this are skipped (not scanned). Secrets in big binary blobs
# are caught by the binary-detection (NUL byte) heuristic anyway, and this keeps
# the scan from OOM'ing on accidental large checkins.
_MAX_FILE_BYTES = 4 << 20
_SKIP_SUFFIXES = frozenset(
    {
        ".bin",
        ".gguf",
        ".ico",
        ".jpeg",
        ".jpg",
        ".pkl",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".svg",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
)


def _redact_match(line: str, pattern: re.Pattern[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if len(value) <= 8:
            return "<redacted>"
        return f"{value[:4]}...{value[-4:]}<redacted>"

    return pattern.sub(replace, line)


def scan_tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        check=True,
        capture_output=True,
        timeout=10.0,
    )
    matches: list[str] = []
    for raw_path in completed.stdout.split(b"\x00"):
        if not raw_path:
            continue
        rel_posix = raw_path.decode("utf-8", errors="ignore")
        path = root / rel_posix
        if not path.is_file():
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        try:
            file_size = path.stat().st_size
        except OSError:
            continue
        if file_size > _MAX_FILE_BYTES:
            continue
        content = path.read_bytes()
        if b"\x00" in content:
            continue
        text = content.decode("utf-8", errors="ignore")
        rel = path.relative_to(root).as_posix()
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    matches.append(f"{rel}:{line_no}: {label}: {_redact_match(line, pattern)}")
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight secret scan across tracked text files using Python + git."
    )
    parser.parse_args(argv)

    root = repo_root()
    matches = scan_tracked_files(root)
    if matches:
        print("== Possible secret matched pattern (matched values redacted): ==", file=sys.stderr)
        for line in matches:
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "secret_scan.py: failing — remove or rotate leaked material, then re-run.",
            file=sys.stderr,
        )
        return 1
    print("OK: secret_scan.py — no high-signal patterns in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
