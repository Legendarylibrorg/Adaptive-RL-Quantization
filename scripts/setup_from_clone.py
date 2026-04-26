#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from _common import repo_root, run, venv_python_path

_NETWORK_PIP_BOOTSTRAP_ENV = "ADAPTIVE_RL_ALLOW_NETWORK_PIP_BOOTSTRAP"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _ensure_pip(python_bin: str) -> None:
    probe = subprocess.run([python_bin, "-m", "pip", "--version"], check=False, capture_output=True)
    if probe.returncode == 0:
        return

    ensurepip = subprocess.run([python_bin, "-m", "ensurepip", "--upgrade"], check=False)
    if ensurepip.returncode == 0:
        return

    if os.environ.get(_NETWORK_PIP_BOOTSTRAP_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        raise SystemExit(
            f"pip is missing and `python -m ensurepip` failed for {python_bin!r}. "
            "Install pip via your OS package manager or rerun this script with "
            f"{_NETWORK_PIP_BOOTSTRAP_ENV}=1 to download {_GET_PIP_URL} over HTTPS as a last resort."
        )

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "get-pip.py"
        urllib.request.urlretrieve(_GET_PIP_URL, target)  # noqa: S310 - opt-in HTTPS pip bootstrap
        run([python_bin, str(target)])


def _ensure_build_backend(python_bin: str) -> None:
    probe = subprocess.run(
        [python_bin, "-c", "import setuptools, setuptools.build_meta"],
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0:
        return
    run([python_bin, "-m", "pip", "install", "setuptools>=61"])


def _install_editable(python_bin: str, root: Path) -> None:
    _ensure_build_backend(python_bin)
    run([python_bin, "-m", "pip", "install", "--no-build-isolation", "-e", "."], cwd=root)


def _activation_hint(venv_dir: Path) -> str:
    root = repo_root()
    try:
        display_dir = venv_dir.relative_to(root)
    except ValueError:
        display_dir = venv_dir
    if sys.platform.startswith("win"):
        return str(display_dir / "Scripts" / "activate")
    return f"source {(display_dir / 'bin' / 'activate').as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform bootstrap: venv, editable install, tests, and RL smoke run."
    )
    parser.add_argument("--venv-dir", default=".venv", help="Virtualenv directory relative to repo root.")
    parser.add_argument(
        "--config",
        default="config.e2e_smoke.json",
        help="Config path for the smoke pipeline, relative to repo root.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip unittest.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip run_research smoke execution.")
    args = parser.parse_args(argv)

    root = repo_root()
    venv_dir = (root / args.venv_dir).resolve()
    config_path = (root / args.config).resolve()

    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)], cwd=root)

    venv_python = venv_python_path(venv_dir)
    if not venv_python.is_file():
        raise SystemExit(f"venv python missing: {venv_python}")

    _ensure_pip(str(venv_python))
    run([str(venv_python), "-m", "pip", "install", "-U", "pip"], cwd=root)
    _install_editable(str(venv_python), root)

    if not args.skip_tests:
        run([str(venv_python), "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=root)
    if not args.skip_smoke:
        run([str(venv_python), "run_research.py", "--config", str(config_path)], cwd=root)

    print("")
    print("OK: venv, editable install, tests, and reproducible E2E RL smoke finished.")
    print(f"   Activate: {_activation_hint(venv_dir)}")
    print("   Then run: python run_research.py --config my.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
