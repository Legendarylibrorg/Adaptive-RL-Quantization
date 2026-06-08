from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class InstallCudaTorchScriptTests(unittest.TestCase):
    def test_help_lists_cuda_indices(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "install_cuda_torch.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("cu130", proc.stdout)
        self.assertIn("cu126", proc.stdout)
        self.assertIn("--force-reinstall", proc.stdout)
        self.assertIn("--check-only", proc.stdout)

    def test_check_only_exits_without_installing(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "install_cuda_torch.py"),
                "--check-only",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("torch_installed", proc.stdout)
        self.assertIn("nvidia_smi_visible", proc.stdout)
        self.assertIn(proc.returncode, (0, 1))


if __name__ == "__main__":
    unittest.main()
