from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"


def _repo_pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _write_fake_console_script(venv: Path, name: str) -> Path:
    if os.name == "nt":
        scripts = venv / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        cli = scripts / f"{name}.exe"
    else:
        bindir = venv / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        cli = bindir / name
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    return cli


def _write_fake_venv_python(venv: Path) -> Path:
    if os.name == "nt":
        python_bin = venv / "Scripts" / "python.exe"
    else:
        python_bin = venv / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    return python_bin


class RunnerScriptCliTests(unittest.TestCase):
    @staticmethod
    def _pyproject_scripts() -> dict[str, str]:
        with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
            payload = tomllib.load(handle)
        return payload["project"]["scripts"]

    def _load_setup_from_clone_module(self):
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
        return module

    def _load_repo_entrypoint_module(self):
        script_path = _REPO_ROOT / "_repo_entrypoint.py"
        spec = importlib.util.spec_from_file_location("repo_entrypoint_module", script_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _installed_command(self, name: str) -> list[str] | None:
        bin_dir = Path(sysconfig.get_path("scripts"))
        candidates = [
            bin_dir / name,
            bin_dir / f"{name}.exe",
            bin_dir / f"{name}-script.py",
        ]
        for candidate in candidates:
            if candidate.is_file():
                if candidate.suffix == ".py":
                    return [sys.executable, str(candidate)]
                return [str(candidate)]
        return None

    def _entrypoint_command(self, name: str) -> list[str]:
        installed = self._installed_command(name)
        if installed is not None:
            return installed

        entrypoint = self._pyproject_scripts()[name]
        module_name, separator, callable_name = entrypoint.partition(":")
        self.assertEqual(separator, ":")
        self.assertEqual(callable_name, "main")

        module_path = _REPO_ROOT / f"{module_name}.py"
        if module_path.is_file():
            return [sys.executable, str(module_path)]
        nested = _REPO_ROOT / "src" / f"{module_name.replace('.', '/')}.py"
        if nested.is_file():
            return [sys.executable, str(nested)]
        if "." in module_name:
            return [sys.executable, "-m", module_name]
        return [sys.executable, "-m", module_name]

    def test_ci_uses_hash_verified_bootstrap_install(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/verify_hashes.py", workflow_text)
        self.assertIn("--require-hashes", workflow_text)
        self.assertIn("--no-build-isolation -e .", workflow_text)
        self.assertIn("core.autocrlf false", workflow_text)
        self.assertNotIn("python -m pip install -U pip", workflow_text)
        self.assertIn("safe.directory", workflow_text)
        self.assertIn("ubuntu-latest", workflow_text)
        self.assertIn('python-version: "3.12"', workflow_text)

    def test_ci_runs_pip_audit_on_bootstrap_requirements(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("precheck:", workflow_text)
        self.assertIn("scripts/ci_precheck.py", workflow_text)
        self.assertIn("needs: precheck", workflow_text)
        self.assertIn("pip-audit:", workflow_text)
        self.assertIn("pip_audit", workflow_text)
        self.assertIn("verify_lockfiles.py", workflow_text)

    def test_ci_installs_hash_pinned_dev_lockfile(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("requirements/dev.txt", workflow_text)
        self.assertNotIn("pip_audit -r requirements/pytorch-cpu.txt", workflow_text)
        self.assertNotIn("pytorch-smoke:", workflow_text)
        self.assertNotIn('pip install -e ".[torch,dev]"', workflow_text)

    def test_dependabot_covers_root_and_requirements(self) -> None:
        config_text = (_REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        self.assertIn("package-ecosystem: pip", config_text)
        self.assertIn("package-ecosystem: github-actions", config_text)
        self.assertIn('directory: "/"', config_text)
        self.assertIn('directory: "/requirements"', config_text)

    def test_dependency_review_workflow_exists(self) -> None:
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "dependency-review.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("github.event.repository.private == false", workflow_text)
        self.assertIn("actions/dependency-review-action@", workflow_text)
        self.assertIn(
            "actions/dependency-review-action@dcd589ca9f7a6ded22e224ca2e288beb6bf9846b",
            workflow_text,
        )
        self.assertIn("fail-on-severity: high", workflow_text)

    def test_pre_commit_config_uses_isolated_python_hook(self) -> None:
        config_text = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        self.assertIn("language: python", config_text)
        self.assertNotIn("language: system", config_text)

    def test_venv_cli_hint_keeps_nested_path(self) -> None:
        scripts_dir = str(_REPO_ROOT / "scripts")
        sys.path.insert(0, scripts_dir)
        try:
            from _common import venv_cli_hint
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / ".venvs" / "project-a"
            _write_fake_console_script(venv, "adaptive-rl-quant")
            hint = venv_cli_hint(venv, root=root)
        self.assertIn(".venvs", hint)
        self.assertIn("project-a", hint)
        if os.name == "nt":
            self.assertIn("Scripts", hint)
        else:
            self.assertIn("bin", hint)

    def test_setup_from_clone_bootstraps_setuptools_when_missing(self) -> None:
        module = self._load_setup_from_clone_module()
        commands: list[tuple[list[str], Path | None]] = []
        with mock.patch.object(
            module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["python"], 1, stdout=b"", stderr=b""),
        ):
            with mock.patch.object(
                module,
                "render_hashed_requirements",
                return_value=(["setuptools==82.0.1 \\\n    --hash=sha256:abc"], [], {}),
            ):
                with mock.patch.object(
                    module, "run", side_effect=lambda cmd, cwd=None: commands.append((cmd, cwd))
                ):
                    module._ensure_build_backend("/tmp/python", _REPO_ROOT)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0][:4], ["/tmp/python", "-m", "pip", "install"])
        self.assertEqual(commands[0][0][4:6], ["--require-hashes", "-r"])
        self.assertTrue(commands[0][0][6].endswith("-requirements-ci.txt"))

    def test_setup_from_clone_editable_install_prefers_standard_pip(self) -> None:
        module = self._load_setup_from_clone_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_python = root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            with mock.patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as pip_mock:
                with mock.patch.object(module, "_ensure_build_backend") as backend_mock:
                    module._install_editable(str(venv_python), root)
            pip_mock.assert_called_once_with(
                [str(venv_python), "-m", "pip", "install", "-e", "."],
                cwd=root,
                check=False,
            )
            backend_mock.assert_not_called()

    def test_setup_from_clone_editable_install_falls_back_to_hash_pin(self) -> None:
        module = self._load_setup_from_clone_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_python = root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            commands: list[tuple[list[str], Path | None]] = []
            with mock.patch.object(module, "repo_root", return_value=root):
                with mock.patch.object(module, "venv_python_path", return_value=venv_python):
                    with mock.patch.object(module, "_ensure_pip"):
                        with mock.patch.object(module, "_ensure_build_backend"):
                            with mock.patch.object(
                                module.subprocess,
                                "run",
                                side_effect=lambda cmd, **kwargs: subprocess.CompletedProcess(
                                    cmd, 1 if kwargs.get("check") is False else 0
                                ),
                            ):
                                with mock.patch.object(
                                    module,
                                    "run",
                                    side_effect=lambda cmd, cwd=None: commands.append((cmd, cwd)),
                                ):
                                    with contextlib.redirect_stdout(io.StringIO()):
                                        code = module.main(
                                            [
                                                "--venv-dir",
                                                ".venv",
                                                "--skip-tests",
                                                "--skip-smoke",
                                            ]
                                        )
            self.assertEqual(code, 0)
            self.assertIn(
                (
                    [str(venv_python), "-m", "pip", "install", "--no-build-isolation", "-e", "."],
                    root,
                ),
                commands,
            )

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
        self.assertIn("--quick", proc.stdout)

    def test_root_setup_sh_delegates_to_python_script(self) -> None:
        script = _REPO_ROOT / "setup.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("setup_from_clone.py", text)

    def test_setup_from_clone_smoke_prefers_installed_cli(self) -> None:
        module = self._load_setup_from_clone_module()
        with tempfile.TemporaryDirectory() as tmp:
            venv_dir = Path(tmp) / ".venv"
            venv_dir.mkdir()
            cli = _write_fake_console_script(venv_dir, "adaptive-rl-quant")
            venv_python = _write_fake_venv_python(venv_dir)
            config = Path(tmp) / "smoke.json"
            config.write_text("{}", encoding="utf-8")
            cmd = module._smoke_command(venv_dir, venv_python, config)
        self.assertEqual(cmd[0], str(cli))
        self.assertIn("--config", cmd)

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

    def test_ci_precheck_python_wrapper_has_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "ci_precheck.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--skip-cli-smoke", proc.stdout)

    def test_ci_precheck_passes_locally(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "ci_precheck.py")],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("ci_precheck.py finished successfully", proc.stdout)

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

    def test_cli_help_is_cp1252_safe(self) -> None:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "cp1252"
        env["PYTHONUTF8"] = "0"
        for script in ("run_research.py", "run_moe_research.py"):
            with self.subTest(script=script):
                proc = subprocess.run(
                    [sys.executable, str(_REPO_ROOT / script), "--help"],
                    cwd=str(_REPO_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="cp1252",
                    env=env,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn("train -> evaluate ->", proc.stdout)

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

    def test_pyproject_declares_open_source_metadata_and_console_scripts(self) -> None:
        with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
            payload = tomllib.load(handle)
        project = payload["project"]
        # PEP 639: SPDX expression + explicit license-files entry.
        self.assertEqual(project["license"], "MIT")
        self.assertIn("LICENSE", project["license-files"])
        self.assertIn("Repository", project["urls"])
        self.assertIn("Issues", project["urls"])
        extras = project["optional-dependencies"]
        self.assertIn("huggingface_hub>=1.18.0", extras["hub"])
        self.assertIn("transformers>=5.10.2", extras["router"])
        self.assertNotIn("transformers>=5.10.2", extras["hub"])
        scripts = project["scripts"]
        self.assertEqual(scripts["adaptive-rl-quant"], "adaptive_quant.cli.research:main")
        self.assertEqual(scripts["adaptive-rl-quant-pytorch"], "adaptive_quant.cli.pytorch:main")
        self.assertEqual(
            scripts["adaptive-rl-quant-online"], "adaptive_quant.cli.online_learning:main"
        )
        self.assertEqual(
            scripts["adaptive-rl-quant-route"], "adaptive_quant.cli.route_learning:main"
        )
        self.assertEqual(scripts["adaptive-rl-quant-replay"], "adaptive_quant.cli.replay:main")
        setuptools_cfg = payload["tool"]["setuptools"]
        self.assertEqual(setuptools_cfg["package-dir"], {"": "src"})
        self.assertEqual(setuptools_cfg["packages"]["find"]["where"], ["src"])
        py_modules = setuptools_cfg["py-modules"]
        self.assertEqual(py_modules, ["config"])
        self.assertNotIn("run_research", py_modules)

    def test_repo_runner_shims_resolve_by_filename_convention(self) -> None:
        entrypoint = self._load_repo_entrypoint_module()
        for path in sorted(_REPO_ROOT.glob("run_*.py")):
            with self.subTest(path=path.name):
                module_name = entrypoint._cli_module_for_script(str(path))
                self.assertEqual(
                    module_name,
                    f"adaptive_quant.cli.{path.stem.removeprefix('run_')}",
                )

        with self.assertRaisesRegex(ValueError, "expected run_<command>.py"):
            entrypoint._cli_module_for_script("research.py")
        with self.assertRaisesRegex(ValueError, "expected matching module"):
            entrypoint._cli_module_for_script("run_missing.py")

    def test_console_entrypoints_have_help(self) -> None:
        for command in self._pyproject_scripts():
            with self.subTest(command=command):
                proc = subprocess.run(
                    [*self._entrypoint_command(command), "--help"],
                    cwd=str(_REPO_ROOT),
                    env=_repo_pythonpath_env(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertTrue(proc.stdout.strip())

    def test_run_replay_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "run_replay.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("manifest", proc.stdout.lower())

    def test_run_pytorch_help_lists_gpu_presets(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "run_pytorch.py"), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("3090", proc.stdout)
        self.assertIn("4090-universal", proc.stdout)

    def test_run_pytorch_config_requires_pytorch_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.json"
            path.write_text(
                json.dumps({"preset": "minimal", "run_name": "cli_test"}), encoding="utf-8"
            )
            proc = subprocess.run(
                [sys.executable, str(_REPO_ROOT / "run_pytorch.py"), "--config", str(path)],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("training_backend", proc.stderr)

    def test_pytorch_trainer_factory_reports_missing_torch_cleanly(self) -> None:
        code = r"""
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("blocked torch for test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.trainer import build_trainer

try:
    build_trainer(FrameworkConfig(training_backend="pytorch", run_name="missing_torch_test"))
except ImportError as exc:
    print(str(exc))
else:
    raise SystemExit("expected ImportError")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REPO_ROOT),
            env=_repo_pythonpath_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("PyTorch is required", proc.stdout)

    def test_run_research_rejects_unknown_config_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps({"preset": "minimal", "run_name": "cli_bad", "training_episode": 4}),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(_REPO_ROOT / "run_research.py"), "--config", str(path)],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Unknown FrameworkConfig keys", proc.stderr)


if __name__ == "__main__":
    unittest.main()
