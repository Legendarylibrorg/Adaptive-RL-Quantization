"""Tests for optional Rust CLI simulator hook."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.rust_cli import (
    RustCliError,
    resolve_rust_cli_binary,
    run_rust_sim_eval,
    rust_cli_status,
    rust_simulator_available,
)
from adaptive_quant.types import (
    HardwareProfile,
    HardwareType,
    InputFeatures,
    LayerSensitivity,
    PromptSample,
    QuantizationDecision,
    QuantMode,
)


def _sample_state() -> tuple:
    state = mock.Mock()
    state.hardware_profile = HardwareProfile(
        hardware_type=HardwareType.GPU,
        name="gpu",
        compute_factor=1.0,
        throughput_bias=1.0,
        latency_bias=1.0,
        memory_budget_mb=24_000.0,
        preferred_bits=4.0,
        kernel_uniformity_preference=0.5,
        ngl=0,
    )
    state.prompt = PromptSample(prompt_id="p0", text="hello", domain="test")
    state.input_features = InputFeatures(
        prompt_length=64,
        token_entropy=0.5,
        token_variance=0.5,
        embedding_norm=1.0,
        complexity_score=0.3,
    )
    state.sensitivity = LayerSensitivity(
        attention_sensitivity=0.2,
        ffn_sensitivity=0.2,
        layer_stats=[0.2, 0.2],
    )
    decision = QuantizationDecision(
        mode=QuantMode.LEARNED,
        effective_layer_bits=[4.0, 4.0],
    )
    return state, decision


class RustCliTests(unittest.TestCase):
    def test_rust_simulator_disabled_by_default(self) -> None:
        cfg = FrameworkConfig(run_name="rust_off", detect_host_hardware=False)
        self.assertFalse(rust_simulator_available(cfg))

    def test_rust_simulator_requires_simulator_backend(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="bad",
                backend="llama_cpp",
                rust_simulator_enabled=True,
                detect_host_hardware=False,
            )

    def test_run_rust_sim_eval_parses_json(self) -> None:
        cfg = FrameworkConfig(run_name="rust_mock", detect_host_hardware=False)
        state, decision = _sample_state()
        metrics = {
            "latency_ms": 100.0,
            "throughput_tps": 50.0,
            "perplexity": 6.0,
            "memory_mb": 1000.0,
            "simulator_engine": "rust_cli",
        }
        with mock.patch("adaptive_quant.rust_cli.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps(metrics),
                stderr="",
            )
            out = run_rust_sim_eval(cfg, state, decision, binary="/fake/rust")
        self.assertEqual(out["simulator_engine"], "rust_cli")
        run_mock.assert_called_once()
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], "/fake/rust")
        self.assertEqual(args[1], "sim-eval")

    def test_run_rust_sim_eval_raises_on_failure(self) -> None:
        cfg = FrameworkConfig(run_name="rust_fail", detect_host_hardware=False)
        state, decision = _sample_state()
        with mock.patch("adaptive_quant.rust_cli.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=1, stdout="", stderr="boom")
            with self.assertRaises(RustCliError):
                run_rust_sim_eval(cfg, state, decision, binary="/fake/rust")

    def test_resolve_binary_from_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "adaptive-rl-quant-rust"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            cfg = FrameworkConfig(
                run_name="explicit",
                rust_cli_binary=str(binary),
                detect_host_hardware=False,
            )
            self.assertEqual(resolve_rust_cli_binary(cfg), str(binary.resolve()))

    def test_resolve_binary_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "adaptive-rl-quant-rust"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            os.environ["ADAPTIVE_RL_RUST_CLI"] = str(binary)
            try:
                cfg = FrameworkConfig(run_name="env", detect_host_hardware=False)
                self.assertEqual(resolve_rust_cli_binary(cfg), str(binary.resolve()))
            finally:
                os.environ.pop("ADAPTIVE_RL_RUST_CLI", None)

    def test_finalize_adds_variant_churn(self) -> None:
        cfg = FrameworkConfig(run_name="churn", detect_host_hardware=False)
        state, decision = _sample_state()
        decision.metadata["moe_variant_churn"] = 2.5
        metrics = {
            "latency_ms": 10.0,
            "throughput_tps": 5.0,
            "perplexity": 6.0,
            "memory_mb": 500.0,
        }
        with mock.patch("adaptive_quant.rust_cli.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0, stdout=json.dumps(metrics), stderr="")
            out = run_rust_sim_eval(cfg, state, decision, binary="/fake/rust")
        self.assertEqual(out["variant_churn"], 2.5)
        self.assertEqual(out["simulator_engine"], "rust_cli")

    def test_rust_cli_status_reports_repo_root(self) -> None:
        cfg = FrameworkConfig(run_name="status", detect_host_hardware=False)
        status = rust_cli_status(cfg)
        self.assertIn("repo_root", status)
        self.assertIn("build_script", status)


if __name__ == "__main__":
    unittest.main()
