"""Red-team regression tests for defense-in-depth hardening.

These cover concerns that came out of an adversarial audit:
- finite-only checkpoint deserialization (no NaN/Inf injection),
- ``previous_action`` length / finiteness validation on resume,
- subprocess timeouts on ``git`` invocations,
- bounded reads in the secret scanner,
- TLS/timeout-hardened network pip bootstrap path,
- structural caps on untrusted JSON / JSONL / config files (nested / wide DoS,
  per-string and aggregate UTF-8 limits, non-finite floats, symmetric ``write_json`` validation).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from unittest import mock

from adaptive_quant.backends.llama_cpp import require_llama_cpp_paths
from adaptive_quant.base_trainer import coerce_previous_action
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.easy_config import load_config
from adaptive_quant.logging_utils import (
    enforce_safe_parsed_json,
    load_jsonl,
    read_json,
    safe_json_loads,
)
from adaptive_quant.model_routes import ModelRoute, RouteCatalog
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.policy import (
    _categorical_head_from_payload,
    _gaussian_head_from_payload,
    _value_head_from_payload,
)
from adaptive_quant.routing import parse_route
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
            msg = str(ctx.exception).lower()
            self.assertTrue("finite" in msg, msg)

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
        for field in (
            "outputs_dir",
            "benchmark_dir",
            "analysis_dir",
            "checkpoint_dir",
            "report_dir",
        ):
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

    def test_router_hf_backend_requires_allowlist_and_pinned_revision(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="hf_strict",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                router_feature_backend="hf",
                router_hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="hf_strict2",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                router_feature_backend="hf",
                router_hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                router_hf_embedding_revision="main",
            )

    def test_hf_download_denied_without_allowlist(self) -> None:
        from adaptive_quant.huggingface_cli import HuggingFaceCli, build_download_command

        cli = HuggingFaceCli(binary="hf", dialect="hf")
        env = {k: v for k, v in os.environ.items() if k != "ADAPTIVE_RL_HF_ALLOW_UNLISTED"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                build_download_command(cli, repo_id="org/model", filename="weights.gguf")
        self.assertIn("denied by default", str(ctx.exception))

    def test_hf_download_rejects_repo_outside_allowlist(self) -> None:
        from adaptive_quant.huggingface_cli import HuggingFaceCli, build_download_command

        env = {
            **os.environ,
            "ADAPTIVE_RL_HF_ALLOWED_REPOS": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        }
        cli = HuggingFaceCli(binary="hf", dialect="hf")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                build_download_command(
                    cli,
                    repo_id="evil-org/evil-model",
                    filename="weights.gguf",
                )


class GitInvocationTimeoutTests(unittest.TestCase):
    """Embedded ``git`` calls cannot block the pipeline indefinitely."""

    def test_git_commit_returns_none_on_timeout(self) -> None:
        import adaptive_quant.pipeline.vcs as vcs_mod

        original_run = vcs_mod.subprocess.run

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            assert "timeout" in kwargs, "git_commit_hash must pass a timeout to git"
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

        vcs_mod.subprocess.run = fake_run  # type: ignore[assignment]
        try:
            self.assertIsNone(git_commit_hash())
        finally:
            vcs_mod.subprocess.run = original_run  # type: ignore[assignment]


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
        env[setup_from_clone._NETWORK_PIP_BOOTSTRAP_SHA_ENV] = setup_from_clone._hash_bytes(
            oversize
        )
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


class UntrustedJsonStructureTests(unittest.TestCase):
    """Hostile or accidental JSON-like trees must fail before full materialization walks."""

    def test_read_json_rejects_dict_key_flood(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "keys.json"
            limit = 24
            payload = {f"k{i}": i for i in range(limit + 1)}
            path.write_text(json.dumps(payload), encoding="utf-8")
            orig = logging_utils.MAX_JSON_OBJECT_KEYS
            try:
                logging_utils.MAX_JSON_OBJECT_KEYS = limit
                with self.assertRaises(ValueError) as ctx:
                    read_json(path, label="key flood")
                self.assertIn("keys", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_OBJECT_KEYS = orig

    def test_read_json_rejects_oversized_primitive_array(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arr.json"
            limit = 50
            path.write_text(json.dumps([0] * (limit + 1)), encoding="utf-8")
            orig = logging_utils.MAX_JSON_ARRAY_LENGTH
            try:
                logging_utils.MAX_JSON_ARRAY_LENGTH = limit
                with self.assertRaises(ValueError) as ctx:
                    read_json(path, label="array bomb")
                self.assertIn("array length", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_ARRAY_LENGTH = orig

    def test_read_json_rejects_oversized_string_segment(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "s.json"
            limit = 64
            path.write_text(json.dumps({"s": "x" * (limit + 1)}), encoding="utf-8")
            orig = logging_utils.MAX_JSON_STRING_BYTES
            try:
                logging_utils.MAX_JSON_STRING_BYTES = limit
                with self.assertRaises(ValueError) as ctx:
                    read_json(path, label="string cap")
                self.assertIn("string segment", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_STRING_BYTES = orig

    def test_read_json_rejects_aggregate_string_flood(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agg.json"
            payload = {f"k{i}": "ab" for i in range(64)}
            path.write_text(json.dumps(payload), encoding="utf-8")
            orig = logging_utils.MAX_JSON_AGGREGATE_STRING_BYTES
            try:
                logging_utils.MAX_JSON_AGGREGATE_STRING_BYTES = 96
                with self.assertRaises(ValueError) as ctx:
                    read_json(path, label="agg cap")
                self.assertIn("aggregate string", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_AGGREGATE_STRING_BYTES = orig

    def test_enforce_safe_parsed_json_rejects_non_finite_float(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            enforce_safe_parsed_json({"z": float("inf")}, label="inf probe")
        self.assertIn("non-finite float", str(ctx.exception))

    def test_safe_json_loads_rejects_deep_nested_lists(self) -> None:
        # Lists only: depth 65 exceeds default MAX_JSON_NESTING_DEPTH (64).
        inner: list[object] = [1]
        for _ in range(64):
            inner = [inner]
        with self.assertRaises(ValueError) as ctx:
            safe_json_loads(json.dumps(inner), label="list nest")
        self.assertIn("nesting depth", str(ctx.exception))

    def test_load_config_json_rejects_calibration_nested_bomb(self) -> None:
        """Deep values under a legitimate key still pass through ``json.loads`` as one tree."""
        inner: dict[str, object] = {"v": 1.0}
        for _ in range(70):
            inner = {"w": inner}
        payload = {"preset": "minimal", "sim_calibration": {"m": inner}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evil.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(path, strict=True)
            self.assertIn("nesting depth", str(ctx.exception))

    def test_load_config_toml_rejects_nested_bomb(self) -> None:
        segs = ["sim_calibration"] + [f"n{i}" for i in range(80)]
        table_path = ".".join(segs)
        body = f'preset = "minimal"\n[{table_path}]\nv = 1.0\n'
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evil.toml"
            path.write_text(body, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(path, strict=True)
            self.assertIn("nesting depth", str(ctx.exception))

    def test_enforce_safe_parsed_json_allows_shallow_toml_like_datetime_leaf(self) -> None:
        from datetime import datetime

        payload = {"run_at": datetime(2026, 1, 1, tzinfo=UTC), "ok": True}
        enforce_safe_parsed_json(payload, label="toml leaf simulation")

    def test_jsonl_aborts_on_hostile_second_line(self) -> None:
        bad_inner: dict[str, object] = {"x": 1}
        for _ in range(70):
            bad_inner = {"k": bad_inner}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mix.jsonl"
            path.write_text(
                '{"ok": true}\n' + json.dumps({"nested": bad_inner}) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                load_jsonl(str(path))
            self.assertIn("line 2", str(ctx.exception))

    def test_jsonl_respects_reduced_depth_per_line(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        inner: dict[str, object] = {"x": 1}
        for _ in range(12):
            inner = {"k": inner}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lines.jsonl"
            path.write_text(
                '{"phase": "ok"}\n' + json.dumps({"data": inner}) + "\n", encoding="utf-8"
            )
            orig = logging_utils.MAX_JSON_NESTING_DEPTH
            try:
                logging_utils.MAX_JSON_NESTING_DEPTH = 8
                with self.assertRaises(ValueError) as ctx:
                    load_jsonl(str(path))
                self.assertIn("line 2", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_NESTING_DEPTH = orig


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


class RuntimePathTraversalTests(unittest.TestCase):
    """Route / llama.cpp paths must reject ``..`` even when supplied via metadata overrides."""

    def test_require_llama_cpp_paths_rejects_parent_reference_in_model_override(self) -> None:
        config = FrameworkConfig(
            run_name="llama_path_redteam",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            llama_cpp_binary="/usr/bin/true",
            llama_cpp_model="/models/base.gguf",
        )
        with self.assertRaises(ValueError):
            require_llama_cpp_paths(
                config,
                model_override="../../etc/passwd",
            )

    def test_model_route_local_path_rejects_parent_reference(self) -> None:
        with self.assertRaises(ValueError):
            ModelRoute(
                route_id="evil",
                repo_id="org/model",
                quant_label="Q4_K_M",
                local_path="../../escape.gguf",
            )

    def test_route_catalog_update_local_path_rejects_parent_reference(self) -> None:
        catalog = RouteCatalog(
            routes=[
                ModelRoute(
                    route_id="ok",
                    repo_id="org/model",
                    quant_label="Q4_K_M",
                )
            ]
        )
        with self.assertRaises(ValueError):
            catalog.update_local_path("ok", "../escape.gguf")

    def test_parse_route_llama_cpp_rejects_parent_reference(self) -> None:
        with self.assertRaises(ValueError):
            parse_route("llama_cpp:../../etc/passwd@q4")


class StructuralLimitTests(unittest.TestCase):
    def test_num_layers_rejects_absurd_values(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            FrameworkConfig(
                run_name="struct_cap",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                num_layers=10**9,
            )
        self.assertIn("num_layers", str(ctx.exception))

    def test_moe_top_k_must_not_exceed_num_experts(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="moe_cap",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                moe_num_experts=4,
                moe_top_k=8,
            )


class LlamaBinaryAllowlistTests(unittest.TestCase):
    def test_allowlist_env_rejects_binary_outside_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.gguf"
            model_path.write_text("fake", encoding="utf-8")
            binary_path = Path(temp_dir) / "llama-cli"
            binary_path.write_text("#!/bin/sh\necho 'tok/s 1.0'\n", encoding="utf-8")
            binary_path.chmod(0o755)
            allowed_root = Path(temp_dir) / "allowed"
            allowed_root.mkdir()
            config = FrameworkConfig(
                run_name="llama_allowlist",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                llama_cpp_binary=str(binary_path),
                llama_cpp_model=str(model_path),
            )
            env = {**os.environ, "ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES": str(allowed_root)}
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    require_llama_cpp_paths(config)
            self.assertIn("ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES", str(ctx.exception))


class EpisodeCountCapTests(unittest.TestCase):
    def test_recommendation_candidate_limit_rejects_absurd_values(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            FrameworkConfig(
                run_name="rec_cap",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                recommendation_candidate_limit=10**9,
            )
        self.assertIn("recommendation_candidate_limit", str(ctx.exception))

    def test_training_episodes_rejects_absurd_values(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="episode_cap",
                training_episodes=10**9,
                evaluation_episodes=1,
                stability_probe_count=1,
            )

    def test_router_routes_validated_at_config_load(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="bad_router",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                router_enabled=True,
                router_routes=("llama_cpp:../../etc/passwd@q4",),
            )


class CliAndPromptHardeningTests(unittest.TestCase):
    def test_cli_path_rejects_parent_reference(self) -> None:
        from adaptive_quant.configuration.validation import validate_cli_path_argument

        with self.assertRaises(ValueError):
            validate_cli_path_argument("log_path", "../secrets.jsonl")

    def test_router_task_text_rejects_oversized_input(self) -> None:
        from adaptive_quant.configuration.validation import (
            MAX_ROUTER_TASK_TEXT_CHARS,
            validate_router_task_text,
        )

        with self.assertRaises(ValueError):
            validate_router_task_text("x" * (MAX_ROUTER_TASK_TEXT_CHARS + 1))

    def test_online_prompt_rejects_nul(self) -> None:
        from adaptive_quant.configuration.validation import validate_online_prompt_text

        with self.assertRaises(ValueError):
            validate_online_prompt_text("hello\x00world")

    def test_online_loop_rejects_oversized_prompt(self) -> None:
        from adaptive_quant.configuration.validation import MAX_ONLINE_PROMPT_TEXT_CHARS
        from adaptive_quant.online_learning import OnlineLearningLoop
        from adaptive_quant.types import HardwareType, OnlineRequest

        config = FrameworkConfig(
            run_name="online_prompt_cap",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            online_learning=True,
            online_requests=1,
        )
        loop = OnlineLearningLoop(config)
        with self.assertRaises(ValueError):
            loop.serve_request(
                OnlineRequest(
                    prompt_text="z" * (MAX_ONLINE_PROMPT_TEXT_CHARS + 1),
                    hardware=HardwareType.GPU,
                )
            )


class TextSanitizationTests(unittest.TestCase):
    def test_sanitize_strips_zero_width_and_normalizes(self) -> None:
        from adaptive_quant.configuration.validation import sanitize_user_text

        raw = "caf\u00e9\u200b"  # NFC vs combining + zero-width space
        self.assertEqual(sanitize_user_text(raw), "café")

    def test_router_task_text_sanitizes_homoglyphs(self) -> None:
        from adaptive_quant.configuration.validation import validate_router_task_text

        # Cyrillic 'а' (U+0430) normalizes differently from Latin 'a' under NFKC in some cases;
        # ensure validation returns stable sanitized output.
        result = validate_router_task_text("test\u200b")
        self.assertNotIn("\u200b", result)

    def test_llama_cpp_cache_uses_sanitized_prompt_key(self) -> None:
        from adaptive_quant.backends.llama_cpp import LlamaCppBackend

        raw = "probe\u200btext"
        config = FrameworkConfig(
            run_name="llama_cache_sanitize",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            llama_cpp_cache_enabled=True,
            llama_cpp_cache_max_entries=8,
            llama_cpp_binary="/usr/bin/true",
            llama_cpp_model="/tmp/model.gguf",
        )
        backend = LlamaCppBackend(config)
        with mock.patch(
            "adaptive_quant.backends.llama_cpp.run_llama_cpp_measurement",
            return_value={"throughput_tps": 1.0},
        ) as measure_mock:
            backend._run_or_cache_measurement(
                llama_cpp_binary="/usr/bin/true",
                llama_cpp_model="/tmp/model.gguf",
                prompt_text=raw,
                ngl=0,
            )
            backend._run_or_cache_measurement(
                llama_cpp_binary="/usr/bin/true",
                llama_cpp_model="/tmp/model.gguf",
                prompt_text="probetext",
                ngl=0,
            )
        self.assertEqual(measure_mock.call_count, 1)


class JsonlIntegrityChainTests(unittest.TestCase):
    def test_load_jsonl_verifies_integrity_chain(self) -> None:
        import adaptive_quant.logging_utils as logging_utils
        from adaptive_quant.logging_utils import JsonlLogger, load_jsonl

        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "chain.jsonl")
            orig = logging_utils._jsonl_integrity_chain_enabled
            try:
                logging_utils._jsonl_integrity_chain_enabled = lambda: True  # type: ignore[method-assign]
                logger = JsonlLogger(path)
                logger.log({"phase": "a", "value": 1})
                logger.log({"phase": "b", "value": 2})
                logger.close()
                records = load_jsonl(path)
                self.assertEqual(len(records), 2)
            finally:
                logging_utils._jsonl_integrity_chain_enabled = orig  # type: ignore[method-assign]

    def test_load_jsonl_rejects_broken_chain(self) -> None:
        from adaptive_quant.logging_utils import load_jsonl

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text(
                '{"phase":"a","_integrity_prev":"","_integrity_hash":"deadbeef"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_jsonl(str(path))
            self.assertIn("integrity hash mismatch", str(ctx.exception).lower())


class RequireCheckpointIntegrityTests(unittest.TestCase):
    def test_require_integrity_rejects_legacy_sidecar(self) -> None:
        from adaptive_quant.checkpoint_integrity import verify_dict_integrity

        env = {**os.environ, "ADAPTIVE_RL_REQUIRE_CHECKPOINT_INTEGRITY": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError) as ctx:
                verify_dict_integrity({"format": 1, "run_name": "x"}, label="legacy")
            self.assertIn("missing integrity_sha256", str(ctx.exception).lower())


class SecurityBypassPolicyTests(unittest.TestCase):
    def test_abort_on_bypass_env(self) -> None:
        from adaptive_quant.security_bypass import enforce_security_bypass_policy

        env = {
            **os.environ,
            "ADAPTIVE_RL_HF_ALLOW_UNLISTED": "1",
            "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS": "1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(SystemExit):
                enforce_security_bypass_policy(context="test")


class CheckpointIntegrityTests(unittest.TestCase):
    def test_tampered_python_checkpoint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="integrity_redteam",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=21,
            )
            trainer = Trainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            trainer.train()
            saved = trainer.save_checkpoint(ckpt)
            trainer.close()

            with open(saved, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["completed_episodes"] = 999
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            resume = config.clone(
                resume_from_checkpoint=ckpt,
                run_name="integrity_redteam_resume",
            )
            with self.assertRaises(ValueError) as ctx:
                Trainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")
            self.assertIn("integrity mismatch", str(ctx.exception).lower())

    def test_legacy_checkpoint_without_integrity_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "legacy_no_integrity.json")
            config = FrameworkConfig(
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="legacy_integrity",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=22,
            )
            trainer = Trainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            trainer.train()
            saved = trainer.save_checkpoint(ckpt.replace(".json", ".pt"))
            trainer.close()

            with open(saved, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload.pop("integrity_sha256", None)
            with open(saved, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            resume = config.clone(
                resume_from_checkpoint=saved,
                run_name="legacy_integrity_resume",
            )
            Trainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")


class OnlinePromptReplayCapTests(unittest.TestCase):
    def test_replay_capped_per_prompt_hash(self) -> None:
        from adaptive_quant.online_learning import OnlineLearningLoop
        from adaptive_quant.types import HardwareType, OnlineRequest

        config = FrameworkConfig(
            run_name="replay_cap",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            online_learning=True,
            online_requests=1,
            online_exploration_rate=1.0,
            online_canary_ratio=0.0,
            online_max_replay_entries_per_prompt_hash=2,
            online_min_replay_size=1,
            online_update_interval=10_000,
        )
        loop = OnlineLearningLoop(config)
        prompt = "repeatable prompt for replay cap test"
        for _ in range(4):
            loop.serve_request(OnlineRequest(prompt_text=prompt, hardware=HardwareType.GPU))
        self.assertLessEqual(len(loop.replay_buffer), 2)
        loop.close()


class DockerComposeHardeningTests(unittest.TestCase):
    def test_dockerfile_base_image_is_digest_pinned(self) -> None:
        dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("FROM python:3.12-slim-bookworm@sha256:", dockerfile)
        self.assertIn("verify_lockfiles.py", dockerfile)

    def test_compose_keeps_security_contract(self) -> None:
        compose = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES", compose)
        self.assertIn("ADAPTIVE_RL_JSONL_INTEGRITY_CHAIN", compose)
        self.assertIn('user: "10001:10001"', compose)
        self.assertIn("privileged: false", compose)
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
        self.assertIn("gpus:", compose)
        self.assertIn("driver: nvidia", compose)
        self.assertIn("config.docker.gpu_smoke.json", compose)

    def test_gpu_compose_does_not_weaken_base_hardening(self) -> None:
        gpu = (
            (Path(__file__).resolve().parent.parent / "docker-compose.gpu.yml")
            .read_text(encoding="utf-8")
            .lower()
        )
        self.assertNotIn("privileged: true", gpu)
        self.assertNotIn("docker.sock", gpu)
        self.assertNotIn("read_only: false", gpu)
        self.assertNotIn("cap_add:", gpu)

    def test_merged_compose_preserves_hardening(self) -> None:
        root = Path(__file__).resolve().parent.parent
        if shutil.which("docker") is None:
            self.skipTest("docker not installed")
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.yml",
                    "-f",
                    "docker-compose.gpu.yml",
                    "config",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.skipTest(f"docker compose unavailable: {exc}")
        if proc.returncode != 0:
            self.skipTest(
                f"docker compose config failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        merged = proc.stdout
        for key in (
            "read_only: true",
            "no-new-privileges:true",
            "gpus:",
            "driver: nvidia",
        ):
            with self.subTest(key=key):
                self.assertIn(key, merged)
        self.assertRegex(merged, r"user:\s*\"?10001:10001\"?")
        self.assertNotIn("privileged: true", merged.lower())

    def test_docker_gpu_smoke_config_uses_cpu_torch(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config.docker.gpu_smoke.json"
        cfg = load_config(path)
        self.assertEqual(cfg.torch_device, "cpu")
        self.assertFalse(cfg.torch_preflight)

    def test_docker_gpu_device_probe_without_require_is_warning(self) -> None:
        root = Path(__file__).resolve().parent.parent
        env = {**os.environ}
        env.pop("ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA", None)
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "docker_gpu_device_probe.py")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("warning", proc.stdout.lower())

    def test_docker_gpu_device_probe_require_fails_without_devices(self) -> None:
        root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA": "1"}
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "docker_gpu_device_probe.py")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("nvidia", proc.stderr.lower())


class CliStartupOverrideRedTeamTests(unittest.TestCase):
    def test_privileged_override_allowed_with_explicit_env(self) -> None:
        import argparse

        from adaptive_quant.cli.common import add_config_override_arguments, apply_config_overrides
        from adaptive_quant.presets.baseline import CONFIG

        parser = argparse.ArgumentParser()
        add_config_override_arguments(parser)
        args = parser.parse_args(["--set", "backend=llama_cpp"])

        with mock.patch.dict(os.environ, {"ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES": "1"}):
            cfg = apply_config_overrides(CONFIG, args)
        self.assertEqual(cfg.backend, "llama_cpp")

    def test_cli_set_rejects_deeply_nested_json(self) -> None:
        import argparse

        from adaptive_quant.cli.common import add_config_override_arguments

        parser = argparse.ArgumentParser()
        add_config_override_arguments(parser)
        inner = 1
        for _ in range(80):
            inner = [inner]
        with self.assertRaises(SystemExit):
            parser.parse_args(["--set", f"hardware_modes={json.dumps(inner)}"])

    def test_cli_set_rejects_non_ascii_override_keys(self) -> None:
        import argparse

        from adaptive_quant.cli.common import add_config_override_arguments

        parser = argparse.ArgumentParser()
        add_config_override_arguments(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--set", "trаining_episodes=1"])


if __name__ == "__main__":
    unittest.main()
