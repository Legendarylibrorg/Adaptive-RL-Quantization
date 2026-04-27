from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import JsonlLogger, read_json
from adaptive_quant.paper_bundle import (
    aggregate_values,
    create_multiseed_paper_bundle,
    create_pipeline_paper_bundle,
)


class PaperBundleTests(unittest.TestCase):
    def test_pipeline_bundle_writes_manifest_metrics_episodes_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = FrameworkConfig(
                run_name="paper_bundle_smoke",
                backend="llama_cpp",
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                outputs_dir=str(tmpdir / "outputs"),
                log_dir=str(tmpdir / "outputs" / "logs"),
                benchmark_dir=str(tmpdir / "outputs" / "benchmarks"),
                analysis_dir=str(tmpdir / "outputs" / "analysis"),
                checkpoint_dir=str(tmpdir / "outputs" / "checkpoints"),
                report_dir=str(tmpdir / "outputs" / "reports"),
                detect_host_hardware=False,
            )
            telemetry = Path(cfg.log_dir) / f"{cfg.run_name}_route_telemetry.jsonl"
            logger = JsonlLogger(str(telemetry))
            logger.log(
                {
                    "step": 0,
                    "route_id": "local-q4",
                    "metrics": {"latency_ms": 12.5, "throughput_tps": 42.0},
                    "reward": 1.25,
                }
            )
            logger.close()

            summary = {
                "git_commit": "abc123",
                "config": {"run_name": cfg.run_name, "backend": cfg.backend},
                "evaluation": {"mean_reward": 1.25, "mean_latency_ms": 12.5},
                "artifacts": {"route_telemetry": str(telemetry)},
            }
            artifacts = create_pipeline_paper_bundle(
                config=cfg,
                summary=summary,
                telemetry_path=str(telemetry),
            )

            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), path)

            manifest = read_json(artifacts["manifest"], label="Paper bundle manifest")
            self.assertEqual(manifest["backend"], "llama_cpp")
            self.assertEqual(manifest["metric_sources"]["perplexity"], "simulator")

            metrics = read_json(artifacts["metrics_summary_json"], label="Metrics summary")
            self.assertIn("evaluation.mean_latency_ms", metrics)
            self.assertNotIn("evaluation.mean_reward", metrics)

            claims = read_json(artifacts["claims_validation_json"], label="Claims validation")
            self.assertEqual(claims["evidence_level"], "local_llama_cpp")
            self.assertFalse(claims["deployment_grade"])

            with Path(artifacts["episodes_csv"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["route_id"], "local-q4")

    def test_pipeline_bundle_marks_external_quality_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            quality_path = tmpdir / "quality.json"
            quality_path.write_text('{"prompt_a": {"perplexity": 8.0}}', encoding="utf-8")
            cfg = FrameworkConfig(
                run_name="paper_bundle_external_quality",
                backend="llama_cpp",
                external_quality_path=str(quality_path),
                outputs_dir=str(tmpdir / "outputs"),
                log_dir=str(tmpdir / "outputs" / "logs"),
                benchmark_dir=str(tmpdir / "outputs" / "benchmarks"),
                analysis_dir=str(tmpdir / "outputs" / "analysis"),
                checkpoint_dir=str(tmpdir / "outputs" / "checkpoints"),
                report_dir=str(tmpdir / "outputs" / "reports"),
                detect_host_hardware=False,
            )
            summary = {
                "git_commit": "abc123",
                "config": {"run_name": cfg.run_name, "backend": cfg.backend},
                "evaluation": {"mean_latency_ms": 12.5, "mean_perplexity": 8.0},
            }

            artifacts = create_pipeline_paper_bundle(config=cfg, summary=summary)

            manifest = read_json(artifacts["manifest"], label="Paper bundle manifest")
            self.assertEqual(manifest["metric_sources"]["perplexity"], "external:perplexity")
            self.assertEqual(manifest["external_quality"]["metric"], "perplexity")
            self.assertIsNotNone(manifest["external_quality"]["sha256"])

            claims = read_json(artifacts["claims_validation_json"], label="Claims validation")
            self.assertTrue(claims["external_quality"])
            self.assertEqual(claims["external_quality_metric"], "perplexity")

    def test_aggregate_values_adds_ci_and_effect_size(self) -> None:
        stats = aggregate_values([1.0, 2.0, 3.0])
        self.assertEqual(stats["n"], 3)
        self.assertGreater(float(stats["stderr"]), 0.0)
        self.assertLess(float(stats["ci95_low"]), float(stats["mean"]))
        self.assertGreater(float(stats["ci95_high"]), float(stats["mean"]))
        self.assertGreater(float(stats["effect_size_vs_zero"]), 0.0)

    def test_multiseed_bundle_writes_manifest_stats_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = FrameworkConfig(
                run_name="multiseed_base",
                outputs_dir=str(tmpdir / "outputs"),
                log_dir=str(tmpdir / "outputs" / "logs"),
                benchmark_dir=str(tmpdir / "outputs" / "benchmarks"),
                analysis_dir=str(tmpdir / "outputs" / "analysis"),
                checkpoint_dir=str(tmpdir / "outputs" / "checkpoints"),
                report_dir=str(tmpdir / "outputs" / "reports"),
                detect_host_hardware=False,
            )
            payload = {
                "run_name": "multiseed_base_multiseed",
                "git_commit": "abc123",
                "config": {"run_name": cfg.run_name, "backend": cfg.backend},
                "seeds": [1, 2],
                "aggregates": {"evaluation.mean_reward": aggregate_values([1.0, 2.0])},
            }

            artifacts = create_multiseed_paper_bundle(
                config=cfg,
                run_name="multiseed_base_multiseed",
                aggregate_payload=payload,
                aggregate_stats=payload["aggregates"],
                report_path=str(tmpdir / "outputs" / "reports" / "multiseed.md"),
            )

            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), path)

            manifest = read_json(artifacts["manifest"], label="Multiseed manifest")
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertEqual(manifest["seeds"], [1, 2])
            stats = read_json(artifacts["aggregate_stats_json"], label="Aggregate stats")
            self.assertIn("evaluation.mean_reward", stats)


if __name__ == "__main__":
    unittest.main()
