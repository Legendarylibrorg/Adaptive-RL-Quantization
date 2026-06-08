#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from _common import repo_root, run, venv_cli_hint, venv_console_command, venv_python_path
from verify_hashes import render_hashed_requirements

_NETWORK_PIP_BOOTSTRAP_ENV = "ADAPTIVE_RL_ALLOW_NETWORK_PIP_BOOTSTRAP"
_NETWORK_PIP_BOOTSTRAP_SHA_ENV = "ADAPTIVE_RL_PIP_BOOTSTRAP_SHA256"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
_GET_PIP_TIMEOUT_S = 30.0
# Refuse to download more than this many bytes; current get-pip.py is ~2 MiB.
_GET_PIP_MAX_BYTES = 16 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SETUPTOOLS_PIN = "82.0.1"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_expected_sha256(raw: str | None) -> str:
    if not raw:
        raise SystemExit(
            f"{_NETWORK_PIP_BOOTSTRAP_ENV}=1 also requires {_NETWORK_PIP_BOOTSTRAP_SHA_ENV} "
            f"to be set to the expected sha256 of {_GET_PIP_URL}. Compute it once and pin it: "
            f"e.g. `curl -fsSL {_GET_PIP_URL} | shasum -a 256` (or sha256sum on Linux). "
            "This avoids running an unverified bootstrap script over the network."
        )
    candidate = raw.strip().lower()
    if not _SHA256_RE.match(candidate):
        raise SystemExit(
            f"{_NETWORK_PIP_BOOTSTRAP_SHA_ENV} must be a 64-character hex sha256 digest; "
            f"got {raw!r}."
        )
    return candidate


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
            f"{_NETWORK_PIP_BOOTSTRAP_ENV}=1 (and {_NETWORK_PIP_BOOTSTRAP_SHA_ENV} pinned to the "
            f"expected sha256) to download {_GET_PIP_URL} over HTTPS as a last resort."
        )

    expected_sha256 = _require_expected_sha256(os.environ.get(_NETWORK_PIP_BOOTSTRAP_SHA_ENV))

    if not _GET_PIP_URL.startswith("https://"):
        raise SystemExit(f"Refusing to bootstrap pip from non-HTTPS URL: {_GET_PIP_URL!r}")
    ssl_context = ssl.create_default_context()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "get-pip.py"
        with urllib.request.urlopen(  # noqa: S310 - opt-in HTTPS pip bootstrap
            _GET_PIP_URL,
            timeout=_GET_PIP_TIMEOUT_S,
            context=ssl_context,
        ) as response:
            payload = response.read(_GET_PIP_MAX_BYTES + 1)
        if len(payload) > _GET_PIP_MAX_BYTES:
            raise SystemExit(
                f"Refusing to execute {_GET_PIP_URL}: payload exceeds "
                f"{_GET_PIP_MAX_BYTES} byte cap (got >= {len(payload)})."
            )
        actual_sha256 = _hash_bytes(payload)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"sha256 mismatch for {_GET_PIP_URL}: expected {expected_sha256}, got "
                f"{actual_sha256}. Refusing to execute an unverified bootstrap script."
            )
        target.write_bytes(payload)
        run([python_bin, str(target)])


def _ensure_build_backend(python_bin: str, root: Path) -> None:
    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import setuptools, setuptools.build_meta; "
                f"raise SystemExit(0 if setuptools.__version__ == '{_SETUPTOOLS_PIN}' else 1)"
            ),
        ],
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0:
        return
    rendered, errors, _ = render_hashed_requirements(
        root, requirement_path=root / "requirements" / "ci.txt"
    )
    if errors:
        raise SystemExit("dependency hash verification failed:\n" + "\n".join(errors))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="-requirements-ci.txt", delete=False, encoding="utf-8"
    ) as handle:
        handle.write("\n".join(rendered) + "\n")
        hashed_path = handle.name
    try:
        run([python_bin, "-m", "pip", "install", "--require-hashes", "-r", hashed_path])
    finally:
        Path(hashed_path).unlink(missing_ok=True)


def _install_editable(python_bin: str, root: Path) -> None:
    """Prefer a normal editable install; fall back to hash-pinned setuptools for minimal envs."""
    probe = subprocess.run(
        [python_bin, "-m", "pip", "install", "-e", "."],
        cwd=root,
        check=False,
    )
    if probe.returncode == 0:
        return
    _ensure_build_backend(python_bin, root)
    run([python_bin, "-m", "pip", "install", "--no-build-isolation", "-e", "."], cwd=root)


def _require_python_311_plus() -> None:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        raise SystemExit(
            f"Python 3.11+ is required; this interpreter is {sys.version.split()[0]} "
            f"({sys.executable})."
        )


def _smoke_command(venv_dir: Path, venv_python: Path, config_path: Path) -> list[str]:
    cli = venv_console_command(venv_dir, "adaptive-rl-quant")
    if cli is not None:
        return [str(cli), "--config", str(config_path)]
    return [str(venv_python), "run_research.py", "--config", str(config_path)]


def _print_success(*, venv_dir: Path, ran_smoke: bool) -> None:
    root = repo_root()
    cli = venv_cli_hint(venv_dir, root=root)
    print("")
    if ran_smoke:
        print(f"OK. Smoke finished. Full run: {cli}")
    else:
        print(f"OK. Run smoke: {cli} -c config.e2e_smoke.json")
        print(f"   Full run:   {cli}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform bootstrap: venv, editable install, tests, and RL smoke run."
    )
    parser.add_argument(
        "--venv-dir", default=".venv", help="Virtualenv directory relative to repo root."
    )
    parser.add_argument(
        "--config",
        default="config.e2e_smoke.json",
        help="Config path for the smoke pipeline, relative to repo root.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Install only: skip tests and E2E smoke (fastest path).",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip unittest.")
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="Run the full unittest discover suite (default: hardware-aware setup subset).",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true", help="Skip adaptive-rl-quant smoke execution."
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.skip_tests = True
        args.skip_smoke = True

    _require_python_311_plus()

    root = repo_root()
    sys.path.insert(0, str(root / "src"))
    from adaptive_quant.nvidia_secure_boundary import enforce_nvidia_secure_boundary

    enforce_nvidia_secure_boundary(context="setup")
    venv_dir = (root / args.venv_dir).resolve()
    config_path = (root / args.config).resolve()

    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)], cwd=root)

    venv_python = venv_python_path(venv_dir)
    if not venv_python.is_file():
        raise SystemExit(f"venv python missing: {venv_python}")

    _ensure_pip(str(venv_python))
    _install_editable(str(venv_python), root)

    ran_tests = not args.skip_tests
    ran_smoke = not args.skip_smoke

    if ran_tests:
        if args.full_tests:
            run(
                [str(venv_python), "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"],
                cwd=root,
            )
        else:
            from adaptive_quant.setup_tests import resolve_setup_test_modules, unittest_command

            modules = resolve_setup_test_modules()
            print("Setup tests:", ", ".join(modules))
            run(unittest_command(str(venv_python), modules, quiet=True), cwd=root)
    if ran_smoke:
        if not config_path.is_file():
            raise SystemExit(f"Smoke config not found: {config_path}")
        run(_smoke_command(venv_dir, venv_python, config_path), cwd=root)

    _print_success(venv_dir=venv_dir, ran_smoke=ran_smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
