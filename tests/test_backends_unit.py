from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_quant.backends.quality import (
    ExternalQualityScores,
    _quality_scores_from_payload,
    _quality_scores_from_rows,
    apply_external_quality,
)
from adaptive_quant.backends.simulator import SimulatorBackend
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.types import HardwareType, QuantizationDecision, QuantMode


class ExternalQualityTests(unittest.TestCase):
    def test_scores_from_json_object_and_list(self) -> None:
        obj_scores = _quality_scores_from_payload(
            {"a": {"perplexity": 3.5}, "b": 4.0},
            metric="perplexity",
        )
        self.assertEqual(obj_scores["a"], 3.5)
        self.assertEqual(obj_scores["b"], 4.0)

        row_scores = _quality_scores_from_rows(
            [
                {"prompt_id": "x", "perplexity": 2.0},
                {"prompt_id": "", "perplexity": 9.0},
                {"prompt_id": "y", "perplexity": "bad"},
            ],
            metric="perplexity",
        )
        self.assertEqual(row_scores, {"x": 2.0})

    def test_from_config_loads_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            path.write_text('{"prompt_a": {"perplexity": 6.5}}', encoding="utf-8")
            cfg = FrameworkConfig(
                run_name="quality_backend_test",
                external_quality_path=str(path),
            )
            scores = ExternalQualityScores.from_config(cfg)
            assert scores is not None
            self.assertEqual(scores.score_for_prompt("prompt_a"), 6.5)

    def test_apply_external_quality_overrides_perplexity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.jsonl"
            path.write_text(
                '{"prompt_id": "very_complex", "perplexity": 7.25}\n',
                encoding="utf-8",
            )
            cfg = FrameworkConfig(
                run_name="apply_quality_test",
                external_quality_path=str(path),
            )
            env = AdaptiveQuantizationEnv(cfg, enable_logging=False)
            try:
                state = env.reset(forced_prompt_id="very_complex")
            finally:
                env.logger.close()
            scores = ExternalQualityScores.from_config(cfg)
            metrics = {
                "latency_ms": 1.0,
                "throughput_tps": 1.0,
                "perplexity": 99.0,
                "memory_mb": 1.0,
            }
            apply_external_quality(metrics, state, scores)
            self.assertEqual(metrics["perplexity"], 7.25)
            self.assertIn("external", str(metrics["perplexity_source"]))


class SimulatorBackendTests(unittest.TestCase):
    def test_evaluate_returns_finite_metrics_for_modes(self) -> None:
        cfg = FrameworkConfig(
            run_name="sim_backend_test",
            training_episodes=2,
            evaluation_episodes=2,
            moe_enabled=True,
        )
        env = AdaptiveQuantizationEnv(cfg, enable_logging=False)
        backend = SimulatorBackend(cfg)
        try:
            state = env.reset(forced_hardware=HardwareType.GPU)
            for mode in (QuantMode.DISCRETE, QuantMode.LEARNED, QuantMode.DYNAMIC):
                decision = finalize_decision(
                    QuantizationDecision(mode=mode, precision_level=6),
                    state,
                    cfg,
                )
                metrics = backend.evaluate(state, decision)
                for key in ("latency_ms", "throughput_tps", "perplexity", "memory_mb"):
                    self.assertTrue(metrics[key] == metrics[key])  # not NaN
                    self.assertGreater(metrics[key], 0.0)
        finally:
            env.logger.close()

    def test_moe_expert_bank_used_when_enabled(self) -> None:
        cfg = FrameworkConfig(run_name="sim_moe_test", moe_enabled=True)
        backend = SimulatorBackend(cfg)
        self.assertIsNotNone(backend.expert_bank)


if __name__ == "__main__":
    unittest.main()
