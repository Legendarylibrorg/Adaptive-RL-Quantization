"""Tests for pipeline output summary helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.output_summary import (
    analysis_takeaway_lines,
    benchmark_metric_rows,
    experiment_config_summary,
    headline_summary_for_metrics,
    online_analysis_takeaway_lines,
    recommendation_decision_block,
    resolve_analysis_log_path,
    slim_analysis_for_summary,
    slim_online_analysis_for_summary,
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

    def test_experiment_config_summary_is_compact(self) -> None:
        cfg = FrameworkConfig(run_name="compact", training_episodes=12)
        summary = experiment_config_summary(cfg)
        self.assertEqual(summary["run_name"], "compact")
        self.assertEqual(summary["training_episodes"], 12)
        self.assertNotIn("reward_weights", summary)

    def test_headline_summary_skips_full_config(self) -> None:
        summary = {
            "config": {"run_name": "x", "nested": {"huge": list(range(100))}},
            "evaluation": {"mean_reward": 1.0},
            "analysis": {"hardware": {"generalization_gap": 0.2, "chart_data": [1, 2, 3]}},
        }
        curated = headline_summary_for_metrics(summary)
        self.assertNotIn("config", curated)
        self.assertIn("evaluation", curated)
        hw = curated["analysis"]["hardware"]
        self.assertEqual(hw.get("generalization_gap"), 0.2)
        self.assertNotIn("chart_data", hw)

    def test_slim_online_analysis_keeps_headlines(self) -> None:
        slim = slim_online_analysis_for_summary(
            {
                "log_path": "t.jsonl",
                "records": 10,
                "mean_served_reward": 0.9,
                "reward_by_hardware": {"gpu": 1.0},
                "svg_paths": ["ignored.svg"],
            }
        )
        self.assertEqual(slim["records"], 10)
        self.assertNotIn("svg_paths", slim)

    def test_online_analysis_takeaway_lines(self) -> None:
        lines = online_analysis_takeaway_lines(
            {
                "mean_served_reward": 0.85,
                "candidate_accept_rate": 0.4,
                "rollback_count": 2,
                "reward_by_hardware": {"gpu": 1.0, "cpu": 0.7},
            }
        )
        self.assertTrue(any("served reward" in line for line in lines))
        self.assertTrue(any("Rollbacks" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
