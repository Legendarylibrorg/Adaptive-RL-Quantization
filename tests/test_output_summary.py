"""Tests for pipeline output summary helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.output_summary import (
    analysis_takeaway_lines,
    benchmark_metric_rows,
    recommendation_decision_block,
    resolve_analysis_log_path,
    slim_analysis_for_summary,
)


class OutputSummaryTests(unittest.TestCase):
    def test_resolve_analysis_log_path_prefers_specialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(run_name="run_a", outputs_dir=tmp)
            log_dir = Path(cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            specialized = log_dir / "run_a_multi_hw.jsonl"
            primary = log_dir / "run_a.jsonl"
            specialized.write_text("{}\n", encoding="utf-8")
            primary.write_text("{}\n", encoding="utf-8")
            self.assertEqual(resolve_analysis_log_path(cfg, "multi_hw"), str(specialized))

    def test_resolve_analysis_log_path_falls_back_to_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(run_name="run_b", outputs_dir=tmp)
            log_dir = Path(cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            primary = log_dir / "run_b.jsonl"
            primary.write_text("{}\n", encoding="utf-8")
            self.assertEqual(resolve_analysis_log_path(cfg, "dynamic"), str(primary))

    def test_slim_analysis_for_summary_keeps_headlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(run_name="slim", outputs_dir=tmp)
            analysis = {
                "hardware": {
                    "generalization_gap": 0.5,
                    "reward_by_hardware": {"gpu": 1.0},
                    "log_path": "x.jsonl",
                }
            }
            slim = slim_analysis_for_summary(analysis, cfg)
            self.assertIn("hardware", slim)
            hw = slim["hardware"]
            assert isinstance(hw, dict)
            self.assertEqual(hw.get("generalization_gap"), 0.5)
            self.assertIn("artifacts_dir", hw)

    def test_recommendation_decision_prefers_adaptive_when_better(self) -> None:
        block = recommendation_decision_block(
            {
                "adaptive_policy": {"mean_reward": 1.2},
                "recommended_quant": {
                    "signature": "Q4",
                    "evaluation": {"mean_reward": 1.0},
                },
            }
        )
        self.assertTrue(block["use_adaptive_policy"])
        self.assertEqual(block["deploy"], "adaptive_policy")

    def test_benchmark_metric_rows_include_deltas(self) -> None:
        rows = benchmark_metric_rows(
            {
                "static_vs_dynamic": {
                    "evaluation": {
                        "static": {"mean_reward": 1.0},
                        "dynamic": {"mean_reward": 1.1},
                    },
                    "quality_variance_delta": 0.05,
                }
            }
        )
        self.assertTrue(any("quality_variance_delta" in row[0] for row in rows))

    def test_analysis_takeaway_lines_from_hardware(self) -> None:
        lines = analysis_takeaway_lines(
            {
                "hardware": {
                    "generalization_gap": 0.25,
                    "reward_by_hardware": {"gpu": 1.0, "cpu": 0.8},
                }
            }
        )
        self.assertTrue(any("generalization gap" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
