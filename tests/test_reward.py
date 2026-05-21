from __future__ import annotations

import unittest

from adaptive_quant.configuration import RewardWeights
from adaptive_quant.reward import compute_weighted_reward
from adaptive_quant.types import BackendMetricDict


def _metrics(**overrides: float) -> BackendMetricDict:
    return {
        "latency_ms": overrides.get("latency_ms", 100.0),
        "throughput_tps": overrides.get("throughput_tps", 50.0),
        "perplexity": overrides.get("perplexity", 5.0),
        "memory_mb": overrides.get("memory_mb", 2000.0),
        "latency_ms_per_token": overrides.get("latency_ms_per_token", 2.0),
    }


class RewardModuleTests(unittest.TestCase):
    def test_compute_weighted_reward_basic_signs(self) -> None:
        weights = RewardWeights()
        reward = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(),
            stability_penalty=0.5,
        )
        self.assertIsInstance(reward, float)
        # Lower latency and perplexity should improve reward vs higher values.
        better = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(latency_ms=50.0, perplexity=3.0),
            stability_penalty=0.5,
        )
        worse = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(latency_ms=200.0, perplexity=9.0),
            stability_penalty=0.5,
        )
        self.assertGreater(better, worse)

    def test_instability_term_optional(self) -> None:
        weights = RewardWeights(epsilon_instability=2.0)
        with_penalty = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(),
            stability_penalty=1.0,
            include_instability=True,
        )
        without_penalty = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(),
            stability_penalty=1.0,
            include_instability=False,
        )
        self.assertGreater(without_penalty, with_penalty)

    def test_perplexity_reference_hinge(self) -> None:
        weights = RewardWeights(zeta_perplexity_over_ref=1.5)
        at_ref = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(perplexity=4.0),
            perplexity_reference=4.0,
        )
        over_ref = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(perplexity=6.0),
            perplexity_reference=4.0,
        )
        self.assertGreater(at_ref, over_ref)

    def test_token_latency_default_applied(self) -> None:
        weights = RewardWeights(eta_token_latency=0.1)
        explicit = compute_weighted_reward(
            reward_weights=weights,
            metrics=_metrics(latency_ms_per_token=3.0),
        )
        default = compute_weighted_reward(
            reward_weights=weights,
            metrics={
                "latency_ms": 100.0,
                "throughput_tps": 50.0,
                "perplexity": 5.0,
                "memory_mb": 2000.0,
            },
            latency_ms_per_token_default=3.0,
        )
        self.assertAlmostEqual(explicit, default, places=9)


if __name__ == "__main__":
    unittest.main()
