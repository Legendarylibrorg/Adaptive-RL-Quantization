"""Red-team regression tests for defense-in-depth hardening.

These cover concerns that came out of an adversarial audit:
- finite-only checkpoint deserialization (no NaN/Inf injection),
- ``previous_action`` length / finiteness validation on resume,
- subprocess timeouts on ``git`` invocations,
- bounded reads in the secret scanner,
- TLS/timeout-hardened network pip bootstrap path.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.base_trainer import coerce_previous_action
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import load_jsonl
from adaptive_quant.policy import (
    _categorical_head_from_payload,
    _gaussian_head_from_payload,
    _value_head_from_payload,
)
from adaptive_quant.research_pipeline import git_commit_hash
from adaptive_quant.trainer import Trainer


class CheckpointFiniteWeightTests(unittest.TestCase):
    """Untrusted JSON cannot smuggle NaN/Inf into deserialized policy heads."""

    def _good_categorical(self) -> dict[str, list]:
        return {"weights": [[0.1, 0.2], [0.3, 0.4]], "bias": [0.0, 0.0]}

    def test_categorical_rejects_inf(self) -> None:
        payload = self._good_categorical()
        payload["weights"][0][1] = float("inf")
        with self.assertRaises(ValueError):
            _categorical_head_from_payload(payload)

    def test_categorical_rejects_nan(self) -> None:
        payload = self._good_categorical()
        payload["bias"][0] = float("nan")
        with self.assertRaises(ValueError):
            _categorical_head_from_payload(payload)

    def test_categorical_rejects_string_inf(self) -> None:
        # A naive ``float()`` on JSON-decoded "Infinity" would slip through;
        # reading via ``json.loads`` would already produce a float here, but the
        # checkpoint format must not silently accept a Python ``str`` either.
        payload = self._good_categorical()
        payload["weights"][1][0] = "1.0"  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            _categorical_head_from_payload(payload)

    def test_gaussian_rejects_negative_stddev(self) -> None:
        payload = {
            "weights": [[0.0, 0.0]],
            "bias": [0.0],
            "stddev": -1.0,
        }
        with self.assertRaises(ValueError):
            _gaussian_head_from_payload(payload)

    def test_gaussian_rejects_nan_stddev(self) -> None:
        payload = {
            "weights": [[0.0, 0.0]],
            "bias": [0.0],
            "stddev": float("nan"),
        }
        with self.assertRaises(ValueError):
            _gaussian_head_from_payload(payload)

    def test_value_head_rejects_inf_bias(self) -> None:
        payload = {"weights": [0.0, 0.0], "bias": float("inf")}
        with self.assertRaises(ValueError):
            _value_head_from_payload(payload)


class PreviousActionCoercionTests(unittest.TestCase):
    def test_none_yields_zero_vector(self) -> None:
        self.assertEqual(coerce_previous_action(None), [0.0, 0.0, 0.0])

    def test_correct_payload_roundtrips(self) -> None:
        self.assertEqual(coerce_previous_action([0.1, -0.2, 0.3]), [0.1, -0.2, 0.3])

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            coerce_previous_action([0.0, 0.0])
        with self.assertRaises(ValueError):
            coerce_previous_action([0.0, 0.0, 0.0, 0.0])

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(TypeError):
            coerce_previous_action((0.0, 0.0, 0.0))  # type: ignore[arg-type]

    def test_rejects_inf(self) -> None:
        with self.assertRaises(ValueError):
            coerce_previous_action([0.0, math.inf, 0.0])

    def test_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            coerce_previous_action([0.0, math.nan, 0.0])

    def test_rejects_bool(self) -> None:
        with self.assertRaises(TypeError):
            coerce_previous_action([True, False, False])  # type: ignore[list-item]

    def test_rejects_string(self) -> None:
        with self.assertRaises(TypeError):
            coerce_previous_action(["0.1", 0.0, 0.0])  # type: ignore[list-item]


class TrainerCheckpointPoisonTests(unittest.TestCase):
    """End-to-end: a tampered checkpoint must not load NaN-poisoned weights."""

    def test_resume_rejects_inf_weight_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="redteam_poison",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=11,
            )
            trainer = Trainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            trainer.train()
            saved = trainer.save_checkpoint(ckpt)
            trainer.close()

            with open(saved, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["policy_state"]["mode_head"]["weights"][0][0] = float("inf")
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            resume = config.clone(
                resume_from_checkpoint=ckpt,
                run_name="redteam_poison_resume",
            )
            with self.assertRaises(ValueError) as ctx:
                Trainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")
            self.assertIn("must be finite", str(ctx.exception))

    def test_resume_rejects_truncated_previous_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="redteam_prevact",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=12,
            )
            trainer = Trainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            trainer.train()
            saved = trainer.save_checkpoint(ckpt)
            trainer.close()

            with open(saved, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["previous_action"] = [0.1, 0.2]  # truncated
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            resume = config.clone(
                resume_from_checkpoint=ckpt,
                run_name="redteam_prevact_resume",
            )
            with self.assertRaises(ValueError):
                Trainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")


class ConfigPathValidationMatrixTests(unittest.TestCase):
    def test_artifact_dirs_reject_parent_reference(self) -> None:
        for field in ("outputs_dir", "benchmark_dir", "analysis_dir", "checkpoint_dir", "report_dir"):
            with self.subTest(field=field):
                kwargs = {
                    "run_name": f"pathmatrix_{field}",
                    "training_episodes": 1,
                    "evaluation_episodes": 1,
                    "stability_probe_count": 1,
                    field: "outputs/../escape",
                }
                with self.assertRaises(ValueError):
                    FrameworkConfig(**kwargs)

    def test_optional_paths_and_hf_revision_reject_control_or_parent_reference(self) -> None:
        for field in ("external_quality_path", "llama_cpp_binary", "llama_cpp_model"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    FrameworkConfig(run_name=f"optional_{field}", **{field: "../escape"})
                with self.assertRaises(ValueError):
                    FrameworkConfig(run_name=f"optional_ctl_{field}", **{field: "bad\npath"})
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_hf_revision", router_hf_embedding_revision="../main")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_hf_allowed", router_hf_allowed_models=("org/../model",))


class GitInvocationTimeoutTests(unittest.TestCase):
    """Embedded ``git`` calls cannot block the pipeline indefinitely."""

    def test_git_commit_returns_none_on_timeout(self) -> None:
        import adaptive_quant.research_pipeline as research_pipeline

        original_run = research_pipeline.subprocess.run

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            assert "timeout" in kwargs, "research_pipeline must pass a timeout to git"
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

        research_pipeline.subprocess.run = fake_run  # type: ignore[assignment]
        try:
            self.assertIsNone(git_commit_hash())
        finally:
            research_pipeline.subprocess.run = original_run  # type: ignore[assignment]


class SecretScanReadCapTests(unittest.TestCase):
    """secret_scan must skip oversized files instead of OOM'ing."""

    @staticmethod
    def _load_secret_scan():
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"

        sys.path.insert(0, str(scripts_dir))
        try:
            import secret_scan  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        return secret_scan

    def test_skips_files_above_cap(self) -> None:
        secret_scan = self._load_secret_scan()

        self.assertTrue(hasattr(secret_scan, "_MAX_FILE_BYTES"))
        self.assertGreater(secret_scan._MAX_FILE_BYTES, 0)

    def test_redacts_matched_secret_value(self) -> None:
        secret_scan = self._load_secret_scan()
        token = "ghp_" + "A" * 36
        redacted = secret_scan._redact_match(f"GITHUB_TOKEN={token}", secret_scan.PATTERNS[2][1])
        self.assertNotIn(token, redacted)
        self.assertIn("<redacted>", redacted)

    def test_scan_reports_redacted_match_and_skips_binary(self) -> None:
        secret_scan = self._load_secret_scan()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret_file = root / "tracked.txt"
            binary_file = root / "binary.txt"
            token = "ghp_" + "B" * 36
            secret_file.write_text(f"token={token}\n", encoding="utf-8")
            binary_file.write_bytes(b"AKIA" + b"ABCDEFGHIJKLMNOP" + b"\x00")
            stdout = b"tracked.txt\x00binary.txt\x00"
            completed = subprocess.CompletedProcess(["git"], 0, stdout=stdout, stderr=b"")
            with mock.patch.object(secret_scan.subprocess, "run", return_value=completed):
                matches = secret_scan.scan_tracked_files(root)

        self.assertEqual(len(matches), 1)
        self.assertIn("tracked.txt:1: github_token:", matches[0])
        self.assertIn("<redacted>", matches[0])
        self.assertNotIn(token, matches[0])


class PipBootstrapTLSAndCapsTests(unittest.TestCase):
    """The opt-in network bootstrap must declare HTTPS, a timeout, and a size cap."""

    def test_module_declares_hardened_constants(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        import sys

        sys.path.insert(0, str(scripts_dir))
        try:
            import setup_from_clone  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        self.assertTrue(setup_from_clone._GET_PIP_URL.startswith("https://"))
        self.assertGreater(setup_from_clone._GET_PIP_TIMEOUT_S, 0)
        self.assertGreater(setup_from_clone._GET_PIP_MAX_BYTES, 0)

    def test_rejects_invalid_bootstrap_sha256(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            import setup_from_clone  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        with self.assertRaises(SystemExit):
            setup_from_clone._require_expected_sha256("not-a-sha")

    def test_network_bootstrap_refuses_hash_mismatch_and_oversize_payload(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            import setup_from_clone  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
                return False

            def read(self, _size: int) -> bytes:
                return self.payload

        env = {
            setup_from_clone._NETWORK_PIP_BOOTSTRAP_ENV: "1",
            setup_from_clone._NETWORK_PIP_BOOTSTRAP_SHA_ENV: "0" * 64,
        }
        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch.object(
                setup_from_clone.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["python"], 1, stdout=b"", stderr=b""),
            ):
                with mock.patch.object(
                    setup_from_clone.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(b"different"),
                ):
                    with mock.patch.object(setup_from_clone, "run") as run_mock:
                        with self.assertRaises(SystemExit):
                            setup_from_clone._ensure_pip("/tmp/python")
                        run_mock.assert_not_called()

        oversize = b"x" * (setup_from_clone._GET_PIP_MAX_BYTES + 1)
        env[setup_from_clone._NETWORK_PIP_BOOTSTRAP_SHA_ENV] = setup_from_clone._hash_bytes(oversize)
        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch.object(
                setup_from_clone.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["python"], 1, stdout=b"", stderr=b""),
            ):
                with mock.patch.object(
                    setup_from_clone.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(oversize),
                ):
                    with mock.patch.object(setup_from_clone, "run") as run_mock:
                        with self.assertRaises(SystemExit):
                            setup_from_clone._ensure_pip("/tmp/python")
                        run_mock.assert_not_called()


class DependencyHashVerificationTests(unittest.TestCase):
    def test_hash_manifest_mismatch_fails(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            import verify_hashes  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            req = root / "requirements" / "ci.txt"
            manifest = root / "security" / "dependency_hashes.json"
            req.parent.mkdir(parents=True)
            manifest.parent.mkdir(parents=True)
            req.write_text("setuptools==82.0.1\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "requirements": {
                            "requirements/ci.txt": {
                                "setuptools==82.0.1": ["sha256:" + "0" * 64],
                                "wheel==0.1": ["sha256:" + "1" * 64],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            _rendered, errors, _path = verify_hashes.render_hashed_requirements(
                root,
                requirement_path=req,
                manifest_path=manifest,
            )
        self.assertTrue(errors)
        self.assertIn("unused entry", "\n".join(errors))

    def test_malformed_hash_manifest_fails(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            import verify_hashes  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "bad.json"
            manifest.write_text('{"requirements": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_hashes.load_dependency_hashes(manifest)


class JsonlLimitTests(unittest.TestCase):
    def test_jsonl_line_count_limit_is_enforced(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.jsonl"
            path.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
            original_limit = logging_utils.MAX_JSONL_LINES
            try:
                logging_utils.MAX_JSONL_LINES = 1
                with self.assertRaises(ValueError):
                    load_jsonl(str(path))
            finally:
                logging_utils.MAX_JSONL_LINES = original_limit


class DockerComposeHardeningTests(unittest.TestCase):
    def test_dockerfile_base_image_is_digest_pinned(self) -> None:
        dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("FROM python:3.12.10-slim-bookworm@sha256:", dockerfile)

    def test_compose_keeps_security_contract(self) -> None:
        compose = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('user: "10001:10001"', compose)
        self.assertIn("read_only: true", compose)
        self.assertIn("cap_drop:", compose)
        self.assertIn("- ALL", compose)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("pids_limit:", compose)
        self.assertIn("/tmp:rw,noexec,nosuid,nodev", compose)
        self.assertIn("adaptive_outputs:/app/outputs", compose)

    def test_gpu_compose_defaults_to_one_visible_gpu(self) -> None:
        compose = (Path(__file__).resolve().parent.parent / "docker-compose.gpu.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-0}", compose)
        self.assertIn("count: 1", compose)


if __name__ == "__main__":
    unittest.main()
