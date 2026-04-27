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
import tempfile
import unittest
from pathlib import Path

from adaptive_quant.base_trainer import coerce_previous_action
from adaptive_quant.configuration import FrameworkConfig
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

    def test_skips_files_above_cap(self) -> None:
        # We import here so the test does not require running from /scripts.
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        import sys

        sys.path.insert(0, str(scripts_dir))
        try:
            import secret_scan  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        self.assertTrue(hasattr(secret_scan, "_MAX_FILE_BYTES"))
        self.assertGreater(secret_scan._MAX_FILE_BYTES, 0)


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


if __name__ == "__main__":
    unittest.main()
