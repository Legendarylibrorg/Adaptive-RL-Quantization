from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import read_json
from adaptive_quant.research_pipeline import ResearchPipeline


class OutputSchemaTests(unittest.TestCase):
    def test_summary_and_recommendation_json_have_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = FrameworkConfig(
                training_episodes=12,
                evaluation_episodes=4,
                benchmark_training_episodes=6,
                benchmark_evaluation_episodes=3,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="schema_test",
                seed=11,
            )

            returned = ResearchPipeline(cfg).run()
            summary_path = Path(cfg.summary_path())
            rec_path = Path(cfg.recommendation_path())

            self.assertTrue(summary_path.is_file())
            self.assertTrue(rec_path.is_file())

            summary = read_json(summary_path, label="summary schema")
            recommendation = read_json(rec_path, label="recommendation schema")

            # Summary file: stable top-level keys downstream tooling relies on.
            for key in (
                "config",
                "git_commit",
                "research",
                "artifact_index",
                "train",
                "evaluation",
                "recommendation",
                "benchmarks",
                "analysis",
                "artifacts",
            ):
                self.assertIn(key, summary)

            research = summary["research"]
            self.assertIsInstance(research, dict)
            self.assertEqual(research["learning_target"]["object"], "quantization_policy")
            self.assertIn("topology", research)

            self.assertIsInstance(summary["config"], dict)
            self.assertIsInstance(summary["train"], dict)
            self.assertIsInstance(summary["evaluation"], dict)
            self.assertIsInstance(summary["benchmarks"], dict)
            self.assertIsInstance(summary["analysis"], dict)
            self.assertIsInstance(summary["artifacts"], dict)

            artifacts = summary["artifacts"]
            for key in ("training_history", "recommendation", "report"):
                self.assertIn(key, artifacts)

            # Paths in artifacts should point at real files when enabled.
            self.assertTrue(Path(artifacts["training_history"]).is_file())
            self.assertTrue(Path(artifacts["recommendation"]).is_file())
            self.assertTrue(Path(artifacts["report"]).is_file())

            # Recommendation file: stable keys and basic types.
            for key in (
                "target_hardware",
                "episodes",
                "adaptive_policy",
                "candidate_count",
                "recommended_quant",
            ):
                self.assertIn(key, recommendation)
            self.assertIsInstance(recommendation["target_hardware"], str)
            self.assertIsInstance(recommendation["episodes"], int)
            self.assertIsInstance(recommendation["adaptive_policy"], dict)

            # The returned in-memory summary should match the serialized one on key metadata.
            returned_rec = Path(returned["artifacts"]["recommendation"])
            self.assertEqual(returned_rec.resolve(), rec_path.resolve())


if __name__ == "__main__":
    unittest.main()
