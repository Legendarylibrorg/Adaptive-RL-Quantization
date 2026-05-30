from __future__ import annotations

import json
import unittest
from pathlib import Path


class SweepRunnerTests(unittest.TestCase):
    def test_sweep_runner_writes_outputs(self) -> None:
        from run_sweep import main

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
                "--quiet",
            ]
        )

        summary_path = Path("outputs/benchmarks/test_sweep_runner_sweep_summary.json")
        report_path = Path("outputs/reports/test_sweep_runner_sweep_report.md")
        bundle_dir = Path("outputs/paper_bundles/test_sweep_runner_sweep")
        self.assertTrue(summary_path.exists(), "Expected sweep JSON summary to be written")
        self.assertTrue(report_path.exists(), "Expected sweep markdown report to be written")
        self.assertTrue(bundle_dir.exists(), "Expected sweep paper bundle to be written")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("trials", summary)
        self.assertEqual(len(summary["trials"]), 2)
        self.assertIn("leaderboard", summary)
        self.assertEqual(len(summary["leaderboard"]), 2)


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


if __name__ == "__main__":
    unittest.main()
