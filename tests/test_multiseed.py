from __future__ import annotations

import json
import unittest
from pathlib import Path


class MultiSeedRunnerTests(unittest.TestCase):
    def test_multiseed_runner_writes_outputs(self) -> None:
        # Import inside the test so this remains a normal stdlib unittest.
        from adaptive_quant.cli.multiseed import main

        # Two seeds, low episode budget so default `unittest discover` stays quick (see scripts/setup_from_clone.py).
        main(
            [
                "--preset",
                "dense",
                "--seeds",
                "1,2",
                "--run-name",
                "test_multiseed",
                "--episodes",
                "48",
                "--quiet",
            ]
        )

        summary_path = Path("outputs/benchmarks/test_multiseed_multiseed_summary.json")
        report_path = Path("outputs/reports/test_multiseed_multiseed_report.md")
        bundle_dir = Path("outputs/paper_bundles/test_multiseed_multiseed")
        self.assertTrue(summary_path.exists(), "Expected multiseed JSON summary to be written")
        self.assertTrue(report_path.exists(), "Expected multiseed markdown report to be written")
        self.assertTrue(bundle_dir.exists(), "Expected multiseed paper bundle to be written")
        self.assertTrue((bundle_dir / "manifest.json").exists(), "Expected multiseed manifest")
        self.assertTrue((bundle_dir / "aggregate_stats.json").exists(), "Expected aggregate stats")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("per_seed", summary)
        self.assertEqual(len(summary["per_seed"]), 2)
        seeds = {entry["seed"] for entry in summary["per_seed"]}
        self.assertEqual(seeds, {1, 2})


if __name__ == "__main__":
    unittest.main()
