from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from adaptive_quant.backend import LlamaCppBackend
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.types import (
    HardwareType,
    PromptSample,
    QuantizationDecision,
    QuantMode,
)


class LlamaCppHardeningTests(unittest.TestCase):
    def test_llama_cpp_prompt_is_clamped_and_timeout_is_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            binary_path = temp_path / "llama-cli"
            model_path = temp_path / "model.gguf"
            binary_path.write_text("#!/bin/sh\necho 'tok/s 123.0\\nms per token 0.5'\n", encoding="utf-8")
            os.chmod(binary_path, 0o755)
            model_path.write_text("fake", encoding="utf-8")

            config = FrameworkConfig(
                backend="llama_cpp",
                llama_cpp_binary=str(binary_path),
                llama_cpp_model=str(model_path),
                llama_cpp_timeout_s=1.5,
                llama_cpp_max_prompt_chars=32,
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="llama_cpp_hardening_test",
                seed=3,
            )

            env = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/test.jsonl")
            state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
            # Make the prompt very long so clamp triggers.
            state = replace(state, prompt=PromptSample(prompt_id=state.prompt.prompt_id, text="x" * 5000, domain=state.prompt.domain))

            decision = finalize_decision(QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4), state, config)

            captured: dict[str, object] = {}

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                return subprocess.CompletedProcess(cmd, 0, stdout="tok/s 100.0\nms per token 1.0\n", stderr="")

            import adaptive_quant.backend as backend_module

            original_run = backend_module.subprocess.run
            backend_module.subprocess.run = fake_run  # type: ignore[assignment]
            try:
                backend = LlamaCppBackend(config)
                _metrics = backend.evaluate(state, decision)
            finally:
                backend_module.subprocess.run = original_run  # type: ignore[assignment]

            cmd = captured.get("cmd")
            self.assertIsInstance(cmd, list)
            assert isinstance(cmd, list)
            self.assertIn("-p", cmd)
            prompt_index = cmd.index("-p") + 1
            passed_prompt = cmd[prompt_index]
            self.assertLessEqual(len(passed_prompt), config.llama_cpp_max_prompt_chars)

            kwargs = captured.get("kwargs")
            self.assertIsInstance(kwargs, dict)
            assert isinstance(kwargs, dict)
            self.assertIn("timeout", kwargs)
            self.assertEqual(float(kwargs["timeout"]), float(config.llama_cpp_timeout_s))

    def test_llama_cpp_backend_uses_route_local_model_and_generate_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            binary_path = temp_path / "llama-cli"
            fallback_model = temp_path / "fallback.gguf"
            route_model = temp_path / "route-q4.gguf"
            binary_path.write_text("#!/bin/sh\necho 'tok/s 123.0\\nms per token 0.5'\n", encoding="utf-8")
            os.chmod(binary_path, 0o755)
            fallback_model.write_text("fallback", encoding="utf-8")
            route_model.write_text("route", encoding="utf-8")

            config = FrameworkConfig(
                backend="llama_cpp",
                llama_cpp_binary=str(binary_path),
                llama_cpp_model=str(fallback_model),
                llama_cpp_generate_tokens=17,
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="llama_cpp_route_model_test",
                seed=5,
            )

            env = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/test.jsonl")
            state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
            decision = finalize_decision(
                QuantizationDecision(
                    mode=QuantMode.DISCRETE,
                    base_bit_width=4,
                    metadata={"llama_cpp_model_path": str(route_model)},
                ),
                state,
                config,
            )

            captured: dict[str, object] = {}

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured["cmd"] = cmd
                return subprocess.CompletedProcess(cmd, 0, stdout="tok/s 100.0\nms per token 1.0\n", stderr="")

            import adaptive_quant.backend as backend_module

            original_run = backend_module.subprocess.run
            backend_module.subprocess.run = fake_run  # type: ignore[assignment]
            try:
                metrics = LlamaCppBackend(config).evaluate(state, decision)
            finally:
                backend_module.subprocess.run = original_run  # type: ignore[assignment]

            cmd = captured.get("cmd")
            self.assertIsInstance(cmd, list)
            assert isinstance(cmd, list)
            self.assertEqual(cmd[cmd.index("-m") + 1], str(route_model.resolve()))
            self.assertEqual(cmd[cmd.index("-n") + 1], "17")
            self.assertEqual(metrics["latency_source"], "llama_cpp")
            self.assertEqual(metrics["perplexity_source"], "simulator")

    def test_llama_cpp_backend_uses_external_quality_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            binary_path = temp_path / "llama-cli"
            model_path = temp_path / "model.gguf"
            quality_path = temp_path / "quality.json"
            binary_path.write_text("#!/bin/sh\necho 'tok/s 123.0\\nms per token 0.5'\n", encoding="utf-8")
            os.chmod(binary_path, 0o755)
            model_path.write_text("fake", encoding="utf-8")
            quality_path.write_text('{"very_complex": {"perplexity": 7.25}}', encoding="utf-8")

            config = FrameworkConfig(
                backend="llama_cpp",
                llama_cpp_binary=str(binary_path),
                llama_cpp_model=str(model_path),
                external_quality_path=str(quality_path),
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="llama_cpp_external_quality_test",
                seed=5,
            )

            env = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/test.jsonl")
            state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
            decision = finalize_decision(QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4), state, config)

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                return subprocess.CompletedProcess(cmd, 0, stdout="tok/s 100.0\nms per token 1.0\n", stderr="")

            import adaptive_quant.backend as backend_module

            original_run = backend_module.subprocess.run
            backend_module.subprocess.run = fake_run  # type: ignore[assignment]
            try:
                metrics = LlamaCppBackend(config).evaluate(state, decision)
            finally:
                backend_module.subprocess.run = original_run  # type: ignore[assignment]

            self.assertEqual(metrics["perplexity"], 7.25)
            self.assertEqual(metrics["perplexity_source"], "external:perplexity")

    def test_llama_cpp_backend_raises_on_non_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            binary_path = temp_path / "llama-cli"
            model_path = temp_path / "model.gguf"
            binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(binary_path, 0o755)
            model_path.write_text("fake", encoding="utf-8")

            config = FrameworkConfig(
                backend="llama_cpp",
                llama_cpp_binary=str(binary_path),
                llama_cpp_model=str(model_path),
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="llama_cpp_failure_test",
                seed=7,
            )

            env = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/test.jsonl")
            state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
            decision = finalize_decision(QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4), state, config)

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal backend failure")

            import adaptive_quant.backend as backend_module

            original_run = backend_module.subprocess.run
            backend_module.subprocess.run = fake_run  # type: ignore[assignment]
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    LlamaCppBackend(config).evaluate(state, decision)
            finally:
                backend_module.subprocess.run = original_run  # type: ignore[assignment]

            self.assertIn("exit code 1", str(ctx.exception))
            self.assertIn("fatal backend failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
