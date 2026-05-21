from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.cli.common import load_config_or_fallback
from adaptive_quant.presets.baseline import CONFIG as BASELINE_CONFIG
from adaptive_quant.presets.moe import CONFIG_MOE


class CliCommonTests(unittest.TestCase):
    def test_load_config_or_fallback_uses_fallback_when_no_path(self) -> None:
        cfg = load_config_or_fallback(None, BASELINE_CONFIG)
        self.assertIs(cfg, BASELINE_CONFIG)

    def test_load_config_or_fallback_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "override.json"
            path.write_text(
                '{"run_name": "cli_json_override", "training_episodes": 17}',
                encoding="utf-8",
            )
            cfg = load_config_or_fallback(str(path), BASELINE_CONFIG)
            self.assertEqual(cfg.run_name, "cli_json_override")
            self.assertEqual(cfg.training_episodes, 17)


class ResearchCliTests(unittest.TestCase):
    def test_research_main_invokes_pipeline_with_fallback(self) -> None:
        from adaptive_quant.cli import research

        with mock.patch("adaptive_quant.research_pipeline.run_pipeline_entrypoint") as run_pipeline:
            with mock.patch.object(sys, "argv", ["adaptive-rl-quant"]):
                research.main()
            run_pipeline.assert_called_once()
            passed = run_pipeline.call_args[0][0]
            self.assertEqual(passed.run_name, BASELINE_CONFIG.run_name)


class MoeResearchCliTests(unittest.TestCase):
    def test_moe_research_main_uses_moe_fallback(self) -> None:
        from adaptive_quant.cli import moe_research

        with mock.patch("adaptive_quant.research_pipeline.run_pipeline_entrypoint") as run_pipeline:
            with mock.patch.object(sys, "argv", ["adaptive-rl-quant-moe"]):
                moe_research.main()
            passed = run_pipeline.call_args[0][0]
            self.assertEqual(passed.run_name, CONFIG_MOE.run_name)
            self.assertTrue(passed.moe_enabled)


class OnlineLearningCliTests(unittest.TestCase):
    def test_online_learning_main_invokes_online_entrypoint(self) -> None:
        from adaptive_quant.cli import online_learning

        with mock.patch(
            "adaptive_quant.cli.online_learning.run_online_pipeline_entrypoint",
        ) as run_online:
            with mock.patch.object(sys, "argv", ["adaptive-rl-quant-online"]):
                online_learning.main()
            run_online.assert_called_once()
            passed = run_online.call_args[0][0]
            self.assertGreater(passed.online_reward_guard, 0.0)


class PytorchCliTests(unittest.TestCase):
    def test_pytorch_main_rejects_simulator_backend_config(self) -> None:
        from adaptive_quant.cli import pytorch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.json"
            path.write_text(
                json.dumps({"preset": "minimal", "run_name": "bad_pytorch"}),
                encoding="utf-8",
            )
            argv = ["adaptive-rl-quant-pytorch", "--config", str(path)]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as ctx:
                    pytorch.main()
                self.assertIn("training_backend", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
