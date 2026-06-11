"""Integration tests for instability fallback in AdaptiveQuantizationEnv."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.types import QuantizationDecision, QuantMode


class InstabilityFallbackTests(unittest.TestCase):
    def test_fallback_replaces_unstable_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                run_name="fallback_test",
                detect_host_hardware=False,
                stability_probe_count=3,
                instability_threshold=0.01,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
            )
            env = AdaptiveQuantizationEnv(
                config,
                log_path=f"{temp_dir}/logs/fallback.jsonl",
                enable_logging=False,
            )
            env.reset(phase="train", episode_index=0)

            unstable = QuantizationDecision(mode=QuantMode.LEARNED, base_bit_width=2)
            with mock.patch.object(
                env,
                "_stability_penalty",
                side_effect=[2.0, 0.0],
            ):
                result = env.evaluate_current(unstable, episode_index=0, log_episode=False)

            self.assertTrue(result.decision.fallback_applied)
            self.assertTrue(result.decision.unstable)
            self.assertEqual(result.decision.metadata.get("fallback_reason"), "instability")
            pre = result.decision.metadata.get("fallback_pre")
            self.assertIsInstance(pre, dict)


if __name__ == "__main__":
    unittest.main()
