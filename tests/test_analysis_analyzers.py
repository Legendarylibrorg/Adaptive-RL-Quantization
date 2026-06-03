from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from analysis.analyzers import (
    CLI_COMMANDS,
    analyze_hardware,
    analyze_inputs,
    analyze_moe_cache,
    analyze_moe_experts,
    analyze_online,
    analyze_quant,
    analyze_training_dynamics,
    run_cli,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SRC = Path(__file__).resolve().parent.parent / "src"


class AnalysisAnalyzerTests(unittest.TestCase):
    def test_jsonl_analysis_warns_on_missing_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.jsonl"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                analyze_hardware(str(missing), f"{temp_dir}/out", phase="eval")
            self.assertTrue(any("log not found" in str(w.message).lower() for w in caught))

    def test_analyze_hardware_writes_summary_and_chart(self) -> None:
        fixture = _FIXTURES / "analysis_eval.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_hardware(str(fixture), temp_dir, phase="eval")
            self.assertIn("reward_by_hardware", summary)
            self.assertIn("gpu", summary["reward_by_hardware"])
            self.assertTrue((Path(temp_dir) / "hardware_generalization_summary.json").is_file())
            self.assertTrue((Path(temp_dir) / "hardware_generalization_reward.svg").is_file())

    def test_analyze_inputs_groups_by_complexity(self) -> None:
        fixture = _FIXTURES / "analysis_eval.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_inputs(str(fixture), temp_dir, phase="eval")
            by_complexity = summary["by_complexity"]
            self.assertIsInstance(by_complexity, dict)
            self.assertGreater(by_complexity["high"]["count"], 0)
            self.assertTrue((Path(temp_dir) / "input_adaptation_summary.json").is_file())

    def test_analyze_quant_learned_episodes(self) -> None:
        fixture = _FIXTURES / "analysis_learned.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_quant(str(fixture), temp_dir, phase="eval")
            self.assertEqual(summary["learned_episode_count"], 1)
            self.assertTrue((Path(temp_dir) / "quant_function_behavior_summary.json").is_file())

    def test_analyze_moe_cache_and_experts(self) -> None:
        fixture = _FIXTURES / "analysis_moe.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_summary = analyze_moe_cache(str(fixture), temp_dir, phase="eval")
            self.assertIn("mean_cache_miss_count", cache_summary)
            expert_summary = analyze_moe_experts(str(fixture), temp_dir, phase="eval")
            self.assertIn("variant_usage", expert_summary)
            self.assertTrue((Path(temp_dir) / "moe_cache_behavior_summary.json").is_file())
            self.assertTrue((Path(temp_dir) / "moe_expert_behavior_summary.json").is_file())

    def test_analyze_training_dynamics_from_history(self) -> None:
        fixture = _FIXTURES / "training_history.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_training_dynamics(str(fixture), temp_dir)
            self.assertEqual(summary["records"], 3)
            self.assertTrue((Path(temp_dir) / "training_dynamics_summary.json").is_file())
            self.assertTrue((Path(temp_dir) / "training_reward_curve.svg").is_file())

    def test_analyze_online_telemetry(self) -> None:
        fixture = _FIXTURES / "online_telemetry.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_online(str(fixture), temp_dir)
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["rollback_count"], 1)
            self.assertTrue((Path(temp_dir) / "online_learning_summary.json").is_file())

    def test_run_cli_usage_on_missing_args(self) -> None:
        argv_backup = sys.argv
        try:
            sys.argv = ["hardware_generalization"]
            with self.assertRaises(SystemExit):
                run_cli("hardware_generalization")
        finally:
            sys.argv = argv_backup

    def test_cli_commands_matches_cli_registry(self) -> None:
        self.assertIn("hardware_generalization", CLI_COMMANDS)
        self.assertIn("online_learning", CLI_COMMANDS)
        self.assertEqual(len(CLI_COMMANDS), 7)


class AnalysisModuleMainTests(unittest.TestCase):
    def test_module_help_lists_commands(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "analysis", "--help"],
            cwd=_SRC.parent,
            env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("hardware_generalization", proc.stdout)
        self.assertIn("online_learning", proc.stdout)

    def test_module_unknown_command_exits_nonzero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "analysis", "not_a_command"],
            cwd=_SRC.parent,
            env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown analysis command", proc.stderr + proc.stdout)

    def test_module_hardware_generalization_smoke(self) -> None:
        fixture = _FIXTURES / "analysis_eval.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "analysis",
                    "hardware_generalization",
                    str(fixture),
                    temp_dir,
                ],
                cwd=_SRC.parent,
                env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Wrote hardware analysis", proc.stdout)


class AdaptiveQuantLazyExportTests(unittest.TestCase):
    def test_all_exports_are_eager_or_lazy_symbols(self) -> None:
        import adaptive_quant

        expected = set(adaptive_quant._EAGER_EXPORTS) | set(adaptive_quant._LAZY)
        self.assertEqual(set(adaptive_quant.__all__), expected)
        self.assertEqual(dir(adaptive_quant), sorted(expected))

    def test_lazy_exports_research_pipeline(self) -> None:
        import adaptive_quant

        self.assertIs(adaptive_quant.ResearchPipeline, adaptive_quant.ResearchPipeline)
        self.assertTrue(callable(adaptive_quant.run_pipeline_entrypoint))


class RegisterBackendTests(unittest.TestCase):
    def test_register_backend_builds_custom_backend(self) -> None:
        from adaptive_quant.backends.registry import build_backend, register_backend
        from adaptive_quant.configuration import FrameworkConfig
        from adaptive_quant.types import BackendMetricDict, EpisodeState, QuantizationDecision

        class _StubBackend:
            def evaluate(
                self, state: EpisodeState, decision: QuantizationDecision
            ) -> BackendMetricDict:
                return {"reward": 1.0, "latency_ms": 1.0, "throughput_tps": 1.0, "perplexity": 1.0}

        register_backend("stub_test", lambda _cfg: _StubBackend())
        config = FrameworkConfig(backend="stub_test", run_name="stub_backend_test")
        backend = build_backend(config)
        self.assertIsInstance(backend, _StubBackend)

    def test_unknown_backend_lists_registered_names(self) -> None:
        from adaptive_quant.backends.registry import build_backend
        from adaptive_quant.configuration import FrameworkConfig

        with self.assertRaises(ValueError):
            FrameworkConfig(backend="definitely_missing_xyz", run_name="bad_backend")
        config = FrameworkConfig(backend="simulator", run_name="bad_backend")
        object.__setattr__(config, "backend", "definitely_missing_xyz")
        with self.assertRaises(ValueError) as ctx:
            build_backend(config)
        self.assertIn("register_backend", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
