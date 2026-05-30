from __future__ import annotations

import unittest

from adaptive_quant.cli.multiseed import _parse_seeds
from adaptive_quant.experiment_aggregate import (
    AggregateStat,
    aggregate_numeric_maps,
    default_key_filter,
)


class ParseSeedsTests(unittest.TestCase):
    def test_parse_comma_separated(self) -> None:
        self.assertEqual(_parse_seeds("1, 2, 3"), [1, 2, 3])

    def test_parse_inclusive_range(self) -> None:
        self.assertEqual(_parse_seeds("3-5"), [3, 4, 5])

    def test_parse_reversed_range_normalizes(self) -> None:
        self.assertEqual(_parse_seeds("5-3"), [3, 4, 5])

    def test_parse_empty_returns_empty(self) -> None:
        self.assertEqual(_parse_seeds(""), [])
        self.assertEqual(_parse_seeds("   "), [])


class AggregateNumericMapsTests(unittest.TestCase):
    def test_aggregate_computes_mean_std_and_ci(self) -> None:
        maps = [
            {"evaluation.mean_reward": 1.0, "benchmarks.gap": 0.1},
            {"evaluation.mean_reward": 3.0, "benchmarks.gap": 0.3},
        ]
        aggregated = aggregate_numeric_maps(maps)
        reward = aggregated["evaluation.mean_reward"]
        self.assertEqual(reward.n, 2)
        self.assertAlmostEqual(reward.mean, 2.0, places=6)
        self.assertGreater(reward.std, 0.0)
        self.assertLess(reward.ci95_low, reward.mean)
        self.assertGreater(reward.ci95_high, reward.mean)

    def test_aggregate_skips_non_finite_values(self) -> None:
        maps = [
            {"metric": 1.0},
            {"metric": float("nan")},
            {"metric": 3.0},
        ]
        aggregated = aggregate_numeric_maps(maps)
        stat = aggregated["metric"]
        self.assertEqual(stat.n, 2)
        self.assertAlmostEqual(stat.mean, 2.0, places=6)

    def test_aggregate_stat_round_trip(self) -> None:
        stat = AggregateStat(
            mean=1.5,
            std=0.5,
            n=4,
            stderr=0.25,
            ci95_low=1.0,
            ci95_high=2.0,
            effect_size_vs_zero=3.0,
        )
        payload = stat.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {
                "mean",
                "std",
                "n",
                "stderr",
                "ci95_low",
                "ci95_high",
                "effect_size_vs_zero",
            },
        )
        self.assertEqual(payload["n"], 4)


class DefaultKeyFilterTests(unittest.TestCase):
    def test_filters_config_prefix(self) -> None:
        self.assertFalse(default_key_filter("config.training_episodes"))

    def test_keeps_gap_and_mean_reward_keys(self) -> None:
        self.assertTrue(
            default_key_filter("benchmarks.single_vs_multi.generalization_gap_improvement")
        )
        self.assertTrue(default_key_filter("evaluation.mean_reward"))

    def test_delta_suffix_allowed(self) -> None:
        self.assertTrue(default_key_filter("benchmarks.static_vs_dynamic.quality_variance_delta"))


if __name__ == "__main__":
    unittest.main()
