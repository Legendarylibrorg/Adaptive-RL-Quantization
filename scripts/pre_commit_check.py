#!/usr/bin/env python3
from __future__ import annotations

import argparse
import compileall
import os
import py_compile
import subprocess
from pathlib import Path

from _common import bash_path, repo_root, resolve_python_bin, run
from secret_scan import scan_tracked_files
from verify_hashes import render_hashed_requirements

# Bound the dev-tool git invocations so a hung/slow filesystem (e.g. NFS,
# WSL2 over a Windows mount) cannot wedge the quality gate indefinitely.
_GIT_TIMEOUT_S = 30.0


def _has_staged_changes(root: Path) -> bool:
    """Return True if the staged tree differs from HEAD.

    ``git diff --cached --quiet`` exits 0 (no changes), 1 (changes), or other
    on a real error; we treat anything that is not exit 1 as "no changes" so
    a missing/broken git still lets pre-commit continue with its other checks.
    """
    try:
        completed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(root),
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False
    return completed.returncode == 1


def _python_compile(root: Path) -> None:
    print("== Python compile (syntax) ==")
    if not compileall.compile_dir(str(root / "src" / "adaptive_quant"), quiet=1):
        raise SystemExit(1)
    if not compileall.compile_dir(str(root / "src" / "analysis"), quiet=1):
        raise SystemExit(1)
    for path in sorted(root.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def _bash_syntax(root: Path) -> None:
    bash = bash_path()
    if bash is None:
        print("== Bash scripts (syntax) ==\nskip: bash not available on PATH")
        return
    print("== Bash scripts (syntax) ==")
    for path in sorted((root / "scripts").glob("*.sh")):
        run([bash, "-n", str(path)], cwd=root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform quality gate: git whitespace checks, secret scan, syntax, and unittest."
    )
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git diff whitespace checks.")
    parser.add_argument("--skip-secret-scan", action="store_true", help="Skip tracked-file secret scan.")
    parser.add_argument("--skip-hash-checks", action="store_true", help="Skip dependency hash manifest checks.")
    parser.add_argument("--skip-bash-syntax", action="store_true", help="Skip bash syntax checks.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unittest.")
    args = parser.parse_args(argv)

    root = repo_root()
    python_bin = resolve_python_bin(root)

    if not args.skip_git_check:
        print("== git diff --check (unstaged whitespace / conflict markers) ==")
        run(["git", "diff", "--check"], cwd=root, timeout=_GIT_TIMEOUT_S)
        if _has_staged_changes(root):
            print("== git diff --cached --check (staged; re-stage after fixes: git add -u) ==")
            run(["git", "diff", "--cached", "--check"], cwd=root, timeout=_GIT_TIMEOUT_S)

    if not args.skip_secret_scan:
        print("== Secret pattern scan (tracked files; heuristic) ==")
        matches = scan_tracked_files(root)
        if matches:
            for line in matches:
                print(line)
            raise SystemExit("secret scan failed")
        print("OK: secret_scan.py — no high-signal patterns in tracked files.")

    if not args.skip_hash_checks:
        print("== Dependency hash manifest ==")
        try:
            _, errors, manifest_path = render_hashed_requirements(root)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"dependency hash verification failed: {exc}") from exc
        if errors:
            for error in errors:
                print(error)
            raise SystemExit("dependency hash verification failed")
        print(f"OK: verify_hashes.py — dependency hashes match {manifest_path.relative_to(root)}.")

    _python_compile(root)

    if not args.skip_bash_syntax:
        _bash_syntax(root)

    if not args.skip_tests:
        print("== unittest ==")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [python_bin, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=str(root),
            env=env,
            check=True,
        )

    print("OK: pre_commit_check.py finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
