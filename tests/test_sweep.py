from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock


class SweepRunnerTests(unittest.TestCase):
    def test_sweep_runner_writes_outputs(self) -> None:
        from adaptive_quant.cli.sweep import main

        with tempfile.TemporaryDirectory() as tmp:
            main(
                [
                    "--preset",
                    "dense",
                    "--run-name",
                    "test_sweep_runner",
                    "--vary",
                    "learning_rate=0.02,0.035",
                    "--episodes",
                    "24",
                    "--outputs-dir",
                    tmp,
                    "--quiet",
                ]
            )

            summary_path = Path(tmp) / "benchmarks" / "test_sweep_runner_sweep_summary.json"
            report_path = Path(tmp) / "reports" / "test_sweep_runner_sweep_report.md"
            csv_path = Path(tmp) / "reports" / "test_sweep_runner_sweep_leaderboard.csv"
            bundle_dir = Path(tmp) / "paper_bundles" / "test_sweep_runner_sweep"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertTrue(bundle_dir.is_dir())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["trials"]), 2)
            self.assertEqual(len(summary["leaderboard"]), 2)

    def test_dry_run_prints_plan_without_running(self) -> None:
        from adaptive_quant.cli.sweep import main

        buffer = StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(sys, "stdout", buffer),
        ):
            main(
                [
                    "--preset",
                    "dense",
                    "--run-name",
                    "dry_run_sweep",
                    "--vary",
                    "learning_rate=0.02,0.035",
                    "--dry-run",
                    "--outputs-dir",
                    tmp,
                ]
            )
        output = buffer.getvalue()
        self.assertIn("2 trial setting(s)", output)
        self.assertIn("learning_rate=0.02", output)
        self.assertNotIn("Run complete", output)


class SweepPlanningTests(unittest.TestCase):
    def test_expand_grid_cartesian_product(self) -> None:
        from adaptive_quant.sweep import build_trial_plans

        plans = build_trial_plans(
            grid={
                "learning_rate": (0.01, 0.05),
                "reward_weights.beta_throughput": (0.04, 0.08),
            },
            explicit_trials=None,
        )
        self.assertEqual(len(plans), 4)
        suffixes = {plan.run_name_suffix for plan in plans}
        self.assertEqual(len(suffixes), 4)

    def test_rank_trials_maximize(self) -> None:
        from adaptive_quant.sweep import SweepTrialPlan, SweepTrialResult, rank_trials

        def result(trial_id: int, value: float | None) -> SweepTrialResult:
            plan = SweepTrialPlan(trial_id=trial_id, overrides={}, run_name_suffix=f"t{trial_id}")
            return SweepTrialResult(
                plan=plan,
                summary={},
                summary_path=f"/tmp/t{trial_id}.json",
                objective_value=value,
            )

        ranked = rank_trials(
            [result(1, 0.2), result(2, 0.9), result(3, None)],
            objective="evaluation.mean_reward",
            direction="maximize",
        )
        self.assertEqual([trial.plan.trial_id for trial in ranked], [2, 1, 3])

    def test_aggregate_objective_values(self) -> None:
        from adaptive_quant.sweep import aggregate_objective_values

        mean, std, count = aggregate_objective_values([0.2, 0.4, 0.6])
        self.assertAlmostEqual(mean, 0.4)
        self.assertAlmostEqual(std, 0.2)
        self.assertEqual(count, 3)

    def test_load_sweep_file_with_seeds(self) -> None:
        from adaptive_quant.sweep import load_sweep_file

        with tempfile.TemporaryDirectory() as tmp:
            sweep_path = Path(tmp) / "sweep.json"
            sweep_path.write_text(
                json.dumps(
                    {
                        "base_config": "config.e2e_smoke.json",
                        "run_name": "seeded_sweep",
                        "objective": "evaluation.mean_reward",
                        "seeds": [11, 13],
                        "grid": {"learning_rate": [0.02, 0.03]},
                    }
                ),
                encoding="utf-8",
            )
            spec, _ = load_sweep_file(sweep_path)
            self.assertEqual(spec.seeds, (11, 13))
            self.assertEqual(len(spec.grid or {}), 1)


if __name__ == "__main__":
    unittest.main()
