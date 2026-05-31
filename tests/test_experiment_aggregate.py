from __future__ import annotations

import unittest

from adaptive_quant.experiment_aggregate import extract_metric, flatten_numeric


class ExperimentAggregateTests(unittest.TestCase):
    def test_flatten_numeric_nested_paths(self) -> None:
        flat = flatten_numeric({"evaluation": {"mean_reward": 0.42, "tags": ["x"]}})
        self.assertEqual(flat["evaluation.mean_reward"], 0.42)
        self.assertNotIn("evaluation.tags", flat)

    def test_extract_metric_supports_suffix_match(self) -> None:
        summary = {"train": {"mean_reward": 0.1}, "evaluation": {"mean_reward": 0.7}}
        self.assertEqual(extract_metric(summary, "evaluation.mean_reward"), 0.7)


if __name__ == "__main__":
    unittest.main()
