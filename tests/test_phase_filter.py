"""Phase tagging on JSONL logs and phase filtering in analysis/analyzers.

These tests verify that:
  - ``AdaptiveQuantizationEnv._log_episode`` records the phase set by ``reset(...)``.
  - ``analysis.analyzers._filter_phase`` keeps only the requested phase, while remaining
    backward compatible with legacy logs that have no ``phase`` field.
  - High-level analyzers (``analyze_hardware``) drop training rows by default, so reports
    no longer mix train and eval reward distributions.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import load_jsonl
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.types import HardwareType, QuantizationDecision, QuantMode
from analysis.analyzers import (
    DEFAULT_ANALYSIS_PHASE,
    _filter_phase,
    analyze_hardware,
)


class PhaseFilterUnitTests(unittest.TestCase):
    def test_keeps_only_matching_phase(self) -> None:
        records = [
            {"phase": "train", "metrics": {"reward": -1.0}},
            {"phase": "eval", "metrics": {"reward": 1.0}},
            {"phase": "eval", "metrics": {"reward": 2.0}},
        ]
        kept = _filter_phase(records, "eval")
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(record.get("phase") == "eval" for record in kept))

    def test_none_keeps_all(self) -> None:
        records = [
            {"phase": "train", "metrics": {"reward": -1.0}},
            {"phase": "eval", "metrics": {"reward": 1.0}},
        ]
        self.assertEqual(_filter_phase(records, None), records)

    def test_legacy_logs_without_phase_pass_through(self) -> None:
        records = [
            {"metrics": {"reward": 0.0}},
            {"metrics": {"reward": 1.0}},
        ]
        kept = _filter_phase(records, "eval")
        self.assertEqual(kept, records)


class PhaseTaggingEnvironmentTests(unittest.TestCase):
    def test_log_records_include_phase_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = str(Path(temp_dir) / "phase_tag.jsonl")
            config = FrameworkConfig(
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="phase_tag",
            )
            env = AdaptiveQuantizationEnv(config, log_path=log_path)
            try:
                env.reset(
                    forced_hardware=HardwareType.GPU,
                    forced_prompt_id="very_complex",
                    phase="train",
                    episode_index=0,
                )
                env.evaluate_current(
                    QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4),
                    episode_index=0,
                )
                env.reset(
                    forced_hardware=HardwareType.GPU,
                    forced_prompt_id="very_complex",
                    phase="eval",
                    episode_index=1,
                )
                env.evaluate_current(
                    QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4),
                    episode_index=1,
                )
            finally:
                env.logger.close()

            records = load_jsonl(log_path)
            self.assertEqual([r.get("phase") for r in records], ["train", "eval"])


class AnalyzerPhaseDefaultTests(unittest.TestCase):
    def test_analyze_hardware_default_drops_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = str(Path(temp_dir) / "split_phase.jsonl")
            config = FrameworkConfig(
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="split_phase",
            )
            env = AdaptiveQuantizationEnv(config, log_path=log_path)
            try:
                for ep, phase, hw in (
                    (0, "train", HardwareType.LOW_RESOURCE),
                    (1, "train", HardwareType.LOW_RESOURCE),
                    (2, "eval", HardwareType.GPU),
                    (3, "eval", HardwareType.CPU),
                ):
                    state = env.reset(
                        forced_hardware=hw,
                        forced_prompt_id="very_complex",
                        phase=phase,
                        episode_index=ep,
                    )
                    decision = finalize_decision(
                        QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4),
                        state,
                        config,
                    )
                    env.evaluate_current(decision, episode_index=ep)
            finally:
                env.logger.close()

            self.assertEqual(DEFAULT_ANALYSIS_PHASE, "eval")
            default = analyze_hardware(log_path, str(Path(temp_dir) / "out_default"))
            all_rows = analyze_hardware(
                log_path,
                str(Path(temp_dir) / "out_all"),
                phase=None,
            )

            # Default (phase="eval") sees only the eval episodes (GPU + CPU).
            self.assertEqual(set(default["reward_by_hardware"].keys()), {"gpu", "cpu"})
            # phase=None falls through to every record, so the train-only LOW_RESOURCE rows reappear.
            self.assertEqual(
                set(all_rows["reward_by_hardware"].keys()),
                {"gpu", "cpu", "low_resource"},
            )


if __name__ == "__main__":
    unittest.main()
