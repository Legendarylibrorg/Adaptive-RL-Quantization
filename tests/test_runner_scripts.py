from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class RunnerScriptCliTests(unittest.TestCase):
    def test_run_pytorch_config_requires_pytorch_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.json"
            path.write_text(json.dumps({"preset": "minimal", "run_name": "cli_test"}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(_REPO_ROOT / "run_pytorch.py"), "--config", str(path)],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("training_backend", proc.stderr)


if __name__ == "__main__":
    unittest.main()
