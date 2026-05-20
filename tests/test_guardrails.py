from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.guardrails import passes_online_guardrails, should_fallback_due_to_instability


def _minimal_config() -> FrameworkConfig:
    return FrameworkConfig(
        run_name="guardrails_test",
        training_episodes=4,
        evaluation_episodes=4,
        online_reward_guard=0.05,
        online_max_latency_ratio=1.25,
        online_max_memory_ratio=1.25,
        online_max_perplexity_delta=0.5,
    )


class GuardrailsTests(unittest.TestCase):
    def test_should_fallback_when_penalty_exceeds_threshold(self) -> None:
        self.assertTrue(should_fallback_due_to_instability(1.1, threshold=1.0))
        self.assertFalse(should_fallback_due_to_instability(0.9, threshold=1.0))

    def test_passes_when_candidate_improves_all_metrics(self) -> None:
        config = _minimal_config()
        self.assertTrue(
            passes_online_guardrails(
                config=config,
                candidate_reward=1.0,
                baseline_reward=0.5,
                candidate_latency_ms=10.0,
                baseline_latency_ms=12.0,
                candidate_memory_mb=100.0,
                baseline_memory_mb=110.0,
                candidate_perplexity=1.1,
                baseline_perplexity=1.2,
            )
        )

    def test_rejects_unstable_or_fallback_candidates(self) -> None:
        config = _minimal_config()
        kwargs = dict(
            config=config,
            candidate_reward=2.0,
            baseline_reward=0.5,
            candidate_latency_ms=10.0,
            baseline_latency_ms=12.0,
            candidate_memory_mb=100.0,
            baseline_memory_mb=110.0,
            candidate_perplexity=1.0,
            baseline_perplexity=1.2,
        )
        self.assertFalse(passes_online_guardrails(**kwargs, candidate_unstable=True))
        self.assertFalse(passes_online_guardrails(**kwargs, candidate_fallback_applied=True))

    def test_rejects_reward_regression_beyond_guard(self) -> None:
        config = _minimal_config()
        self.assertFalse(
            passes_online_guardrails(
                config=config,
                candidate_reward=0.4,
                baseline_reward=0.5,
                candidate_latency_ms=10.0,
                baseline_latency_ms=12.0,
                candidate_memory_mb=100.0,
                baseline_memory_mb=110.0,
                candidate_perplexity=1.0,
                baseline_perplexity=1.2,
            )
        )


if __name__ == "__main__":
    unittest.main()
