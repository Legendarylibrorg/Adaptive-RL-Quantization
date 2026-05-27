#!/usr/bin/env python3
"""Fast CI precheck: supply-chain verification, secret scan, syntax, and CLI security invariants."""

from __future__ import annotations

import argparse
import compileall
import os
import py_compile
import sys
from pathlib import Path

from _common import repo_root, resolve_python_bin
from secret_scan import scan_tracked_files
from verify_hashes import render_hashed_requirements


def _python_compile(root: Path) -> None:
    print("== Python compile (syntax) ==")
    if not compileall.compile_dir(str(root / "src" / "adaptive_quant"), quiet=1):
        raise SystemExit(1)
    if not compileall.compile_dir(str(root / "src" / "analysis"), quiet=1):
        raise SystemExit(1)
    for path in sorted(root.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def _dependency_integrity(root: Path) -> None:
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
    print("== pip-compile lockfile hashes ==")
    from verify_lockfiles import main as verify_lockfiles_main

    if verify_lockfiles_main() != 0:
        raise SystemExit("lockfile verification failed")


def _secret_scan(root: Path) -> None:
    print("== Secret pattern scan (tracked files; heuristic) ==")
    matches = scan_tracked_files(root)
    if matches:
        for line in matches:
            print(line)
        raise SystemExit("secret scan failed")
    print("OK: secret_scan.py — no high-signal patterns in tracked files.")


def _ci_security_env() -> None:
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        print("== CI security env (skip outside GitHub Actions) ==")
        return
    print("== CI security env ==")
    abort = os.environ.get("ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS", "").strip().lower()
    if abort not in {"1", "true", "yes", "on"}:
        raise SystemExit(
            "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS must be enabled in GitHub Actions CI."
        )
    print("OK: ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS is active.")


def _cli_startup_override_smoke(root: Path, python_bin: str) -> None:
    print("== CLI startup override security smoke ==")
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES", None)
    code = (
        "import argparse, os, sys\n"
        "from unittest import mock\n"
        "from adaptive_quant.cli.common import add_config_override_arguments, apply_config_overrides\n"
        "from adaptive_quant.presets.baseline import CONFIG\n"
        "parser = argparse.ArgumentParser()\n"
        "add_config_override_arguments(parser)\n"
        "safe_args = parser.parse_args(['--training-episodes', '8'])\n"
        "cfg = apply_config_overrides(CONFIG, safe_args)\n"
        "assert cfg.training_episodes == 8, cfg.training_episodes\n"
        "privileged_args = parser.parse_args(['--set', 'backend=llama_cpp'])\n"
        "env = os.environ.copy()\n"
        "env.pop('ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES', None)\n"
        "try:\n"
        "    with mock.patch.dict(os.environ, env, clear=True):\n"
        "        apply_config_overrides(CONFIG, privileged_args)\n"
        "except SystemExit:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('privileged CLI override was not blocked')\n"
        "deep = 1\n"
        "for _ in range(80):\n"
        "    deep = [deep]\n"
        "import json\n"
        "try:\n"
        "    parser.parse_args(['--set', f'hardware_modes={json.dumps(deep)}'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('nested JSON override was not rejected')\n"
    )
    import subprocess

    completed = subprocess.run(
        [python_bin, "-c", code],
        cwd=str(root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)
    print("OK: privileged overrides blocked and safe overrides accepted.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fast CI precheck: dependency integrity, secret scan, syntax, CI env, "
            "and CLI startup override security invariants."
        )
    )
    parser.add_argument(
        "--skip-cli-smoke",
        action="store_true",
        help="Skip CLI startup override security smoke (requires PYTHONPATH=src).",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    python_bin = resolve_python_bin(root)

    _secret_scan(root)
    _dependency_integrity(root)
    _python_compile(root)
    _ci_security_env()
    if not args.skip_cli_smoke:
        _cli_startup_override_smoke(root, python_bin)

    print("OK: ci_precheck.py finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
