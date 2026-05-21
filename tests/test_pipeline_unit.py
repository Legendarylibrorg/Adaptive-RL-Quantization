from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.artifacts import maybe_save_final_checkpoint, write_training_history
from adaptive_quant.pipeline.analysis_runner import run_research_analysis
from adaptive_quant.pipeline.report_markdown import write_research_report_markdown

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


class PipelineArtifactsTests(unittest.TestCase):
    def test_write_training_history_respects_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(
                run_name="history_flag",
                outputs_dir=tmp,
                write_training_history=False,
            )

            class Trainer:
                training_history = [{"step": 1}]

            self.assertIsNone(write_training_history(cfg, Trainer()))

            cfg_on = FrameworkConfig(
                run_name="history_on",
                outputs_dir=tmp,
                write_training_history=True,
            )
            path = write_training_history(cfg_on, Trainer())
            self.assertIsNotNone(path)
            assert path is not None
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(payload, [{"step": 1}])

    def test_maybe_save_final_checkpoint_calls_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(run_name="ckpt_test", outputs_dir=tmp)

            class Trainer:
                def save_checkpoint(self, path: str) -> str:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text("{}", encoding="utf-8")
                    return path

            path = maybe_save_final_checkpoint(cfg, Trainer())
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(Path(path).is_file())


class AnalysisRunnerTests(unittest.TestCase):
    def _stage_logs(self, log_dir: Path, run_name: str) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(_FIXTURES / "analysis_eval.jsonl", log_dir / f"{run_name}_multi_hw.jsonl")
        shutil.copy(_FIXTURES / "analysis_eval.jsonl", log_dir / f"{run_name}_dynamic.jsonl")
        shutil.copy(_FIXTURES / "analysis_learned.jsonl", log_dir / f"{run_name}_learned.jsonl")
        shutil.copy(_FIXTURES / "analysis_moe.jsonl", log_dir / f"{run_name}.jsonl")

    def test_run_research_analysis_with_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_name = "pipeline_analysis_test"
            log_dir = Path(tmp) / "logs"
            analysis_dir = Path(tmp) / "analysis"
            self._stage_logs(log_dir, run_name)
            history = _FIXTURES / "training_history.json"
            cfg = FrameworkConfig(
                run_name=run_name,
                outputs_dir=tmp,
                log_dir=str(log_dir),
                analysis_dir=str(analysis_dir),
                moe_enabled=True,
            )
            result = run_research_analysis(cfg, str(history))
            self.assertIn("hardware", result)
            self.assertIn("input", result)
            self.assertIn("quant_function", result)
            self.assertIn("moe_experts", result)
            self.assertIn("moe_cache", result)
            self.assertIn("training_dynamics", result)
            self.assertTrue((analysis_dir / run_name / "hardware").is_dir())


class ReportMarkdownTests(unittest.TestCase):
    def test_write_research_report_markdown_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(
                run_name="report_off",
                outputs_dir=tmp,
                write_research_report=False,
            )
            self.assertIsNone(
                write_research_report_markdown(
                    cfg,
                    git_commit=None,
                    train_summary={},
                    eval_summary={},
                    benchmark_summary={},
                    gpu_profile_report=None,
                    preflight_report=None,
                    vram_report=None,
                    analysis={},
                    history_path=None,
                    checkpoint_path=None,
                    recommendation_summary=None,
                )
            )

    def test_write_research_report_markdown_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FrameworkConfig(
                run_name="report_on",
                outputs_dir=tmp,
                write_research_report=True,
            )
            benchmark_summary = {
                "single_vs_multi": {"generalization_gap_improvement": 0.12},
                "static_vs_dynamic": {
                    "evaluation": {
                        "static": {"mean_reward": 1.0, "mean_stability_penalty": 0.1},
                        "dynamic": {"mean_reward": 1.2, "mean_stability_penalty": 0.05},
                    }
                },
                "discrete_vs_learned": {
                    "evaluation": {
                        "discrete": {"mean_reward": 0.9, "mean_latency_ms": 10.0},
                        "learned": {"mean_reward": 1.1, "mean_latency_ms": 9.0},
                    }
                },
            }
            with mock.patch(
                "adaptive_quant.pipeline.report_markdown.Path.exists",
                return_value=False,
            ):
                path = write_research_report_markdown(
                    cfg,
                    git_commit="abc123",
                    train_summary={
                        "episodes": 4,
                        "mean_reward": 1.0,
                        "best_reward": 2.0,
                        "final_reward": 1.5,
                    },
                    eval_summary={
                        "mean_reward": 1.2,
                        "mean_latency_ms": 8.0,
                        "mean_throughput_tps": 100.0,
                        "mean_memory_mb": 500.0,
                        "mean_perplexity": 4.0,
                        "mean_stability_penalty": 0.1,
                    },
                    benchmark_summary=benchmark_summary,
                    gpu_profile_report={"profile": "test"},
                    preflight_report=None,
                    vram_report=None,
                    analysis={"hardware": {"ok": True}},
                    history_path="history.json",
                    checkpoint_path="ckpt.pt",
                    recommendation_summary={
                        "target_hardware": "gpu",
                        "detected_hardware": "gpu",
                        "adaptive_policy": {"mean_reward": 1.2},
                        "recommended_quant": {
                            "signature": "Q4",
                            "evaluation": {"mean_reward": 1.0},
                        },
                    },
                )
            self.assertIsNotNone(path)
            assert path is not None
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("# report_on", text)
            self.assertIn("abc123", text)
            self.assertIn("universal policy gap improvement", text)


if __name__ == "__main__":
    unittest.main()
