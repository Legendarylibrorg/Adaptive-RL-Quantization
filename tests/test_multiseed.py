from __future__ import annotations

from pathlib import Path
import unittest


class MultiSeedRunnerTests(unittest.TestCase):
    def test_multiseed_runner_writes_outputs(self) -> None:
        # Import inside the test so this remains a normal stdlib unittest.
        from run_multiseed import main

        # Two seeds, low episode budget so default `unittest discover` stays quick (see scripts/setup_from_clone.sh).
        main(["--preset", "dense", "--seeds", "1,2", "--run-name", "test_multiseed", "--episodes", "48"])

        summary_path = Path("outputs/benchmarks/test_multiseed_multiseed_summary.json")
        report_path = Path("outputs/reports/test_multiseed_multiseed_report.md")
        self.assertTrue(summary_path.exists(), "Expected multiseed JSON summary to be written")
        self.assertTrue(report_path.exists(), "Expected multiseed markdown report to be written")


if __name__ == "__main__":
    unittest.main()

