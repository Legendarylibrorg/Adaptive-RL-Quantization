"""Tests for hardware-aware setup test selection."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class SetupTestSelectionTests(unittest.TestCase):
    def test_resolve_modules_always_includes_core_and_cli(self) -> None:
        from adaptive_quant.setup_tests import resolve_setup_test_modules

        modules = resolve_setup_test_modules(include_torch=False, include_nvidia=False)
        self.assertIn("tests.test_presets_and_shims", modules)
        self.assertIn("tests.test_easy_config", modules)
        self.assertIn("tests.test_cli_behavior", modules)
        self.assertNotIn("tests.test_torch_trainer", modules)
        self.assertNotIn("tests.test_nvidia_secure_boundary", modules)

    def test_resolve_modules_adds_torch_when_requested(self) -> None:
        from adaptive_quant.setup_tests import resolve_setup_test_modules

        modules = resolve_setup_test_modules(include_torch=True, include_nvidia=False)
        self.assertIn("tests.test_torch_trainer", modules)

    def test_resolve_modules_adds_nvidia_when_requested(self) -> None:
        from adaptive_quant.setup_tests import resolve_setup_test_modules

        modules = resolve_setup_test_modules(include_torch=False, include_nvidia=True)
        self.assertIn("tests.test_nvidia_secure_boundary", modules)
        self.assertIn("tests.test_install_cuda_torch", modules)

    def test_run_setup_tests_script_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "run_setup_tests.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--full", proc.stdout)
        self.assertIn("--no-torch", proc.stdout)

    def test_run_setup_tests_script_runs_core_subset(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "run_setup_tests.py"),
                "--no-torch",
                "--no-nvidia",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    if importlib.util.find_spec("adaptive_quant") is None:
        sys.path.insert(0, str(_REPO_ROOT / "src"))
    unittest.main()
