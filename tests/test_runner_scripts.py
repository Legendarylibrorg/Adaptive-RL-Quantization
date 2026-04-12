from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class RunnerScriptCliTests(unittest.TestCase):
    def test_ci_uses_hash_verified_bootstrap_install(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/verify_hashes.py", workflow_text)
        self.assertIn("--require-hashes", workflow_text)
        self.assertIn("--no-build-isolation -e .", workflow_text)

    def test_dependabot_covers_root_and_requirements(self) -> None:
        config_text = (_REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        self.assertIn("package-ecosystem: pip", config_text)
        self.assertIn('directory: "/"', config_text)
        self.assertIn('directory: "/requirements"', config_text)

    def test_dependency_review_workflow_exists(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "dependency-review.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("actions/dependency-review-action@", workflow_text)

    def test_pre_commit_config_uses_isolated_python_hook(self) -> None:
        config_text = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        self.assertIn("language: python", config_text)
        self.assertNotIn("language: system", config_text)

    def test_setup_from_clone_activation_hint_keeps_nested_path(self) -> None:
        script_path = _REPO_ROOT / "scripts" / "setup_from_clone.py"
        sys.path.insert(0, str(script_path.parent))
        try:
            spec = importlib.util.spec_from_file_location("setup_from_clone_module", script_path)
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        hint = module._activation_hint(Path(".venvs") / "project-a")
        self.assertIn(".venvs", hint)
        self.assertIn("project-a", hint)

    def test_setup_from_clone_python_wrapper_has_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "setup_from_clone.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--skip-smoke", proc.stdout)

    def test_pre_commit_check_python_wrapper_has_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "pre_commit_check.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--skip-tests", proc.stdout)
        self.assertIn("--skip-hash-checks", proc.stdout)

    def test_secret_scan_python_wrapper_has_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "secret_scan.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("secret", proc.stdout.lower())

    def test_verify_hashes_python_wrapper_has_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "verify_hashes.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--requirement", proc.stdout)

    def test_verify_hashes_renders_require_hashes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "requirements.lock"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "verify_hashes.py"),
                    "--output",
                    str(output_path),
                ],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            rendered = output_path.read_text(encoding="utf-8")
            self.assertIn("setuptools==82.0.1", rendered)
            self.assertIn("--hash=sha256:", rendered)

    def test_run_pytorch_help_lists_gpu_presets(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "run_pytorch.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("4090-universal", proc.stdout)

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
