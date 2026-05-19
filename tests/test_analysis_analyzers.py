from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.analyzers import analyze_hardware, analyze_inputs


class AnalysisAnalyzerTests(unittest.TestCase):
    def test_analyze_hardware_writes_summary_and_chart(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "analysis_eval.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_hardware(str(fixture), temp_dir, phase="eval")
            self.assertIn("reward_by_hardware", summary)
            self.assertIn("gpu", summary["reward_by_hardware"])
            self.assertTrue((Path(temp_dir) / "hardware_generalization_summary.json").is_file())
            self.assertTrue((Path(temp_dir) / "hardware_generalization_reward.svg").is_file())

    def test_analyze_inputs_groups_by_complexity(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "analysis_eval.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = analyze_inputs(str(fixture), temp_dir, phase="eval")
            by_complexity = summary["by_complexity"]
            self.assertIsInstance(by_complexity, dict)
            self.assertGreater(by_complexity["high"]["count"], 0)
            self.assertTrue((Path(temp_dir) / "input_adaptation_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
