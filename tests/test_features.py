from __future__ import annotations

import math
import unittest

from adaptive_quant.features import (
    COMPLEXITY_ANALYSIS_THRESHOLDS,
    complexity_bucket,
    estimate_layer_sensitivity,
    extract_input_features,
    summarize_precision_needs,
    tokenize,
)
from adaptive_quant.types import PromptSample


class FeaturesModuleTests(unittest.TestCase):
    def test_tokenize_splits_words_and_punctuation(self) -> None:
        tokens = tokenize("Hello, world! RL_Quant 2026.")
        self.assertIn("hello", tokens)
        self.assertIn(",", tokens)
        self.assertIn("rl_quant", tokens)

    def test_complexity_bucket_respects_thresholds(self) -> None:
        self.assertEqual(
            complexity_bucket(0.1, thresholds=COMPLEXITY_ANALYSIS_THRESHOLDS),
            "low",
        )
        self.assertEqual(
            complexity_bucket(0.5, thresholds=COMPLEXITY_ANALYSIS_THRESHOLDS),
            "medium",
        )
        self.assertEqual(
            complexity_bucket(0.9, thresholds=COMPLEXITY_ANALYSIS_THRESHOLDS),
            "high",
        )
        self.assertEqual(complexity_bucket(float("nan")), "medium")

    def test_extract_input_features_empty_prompt(self) -> None:
        features = extract_input_features(PromptSample("empty", "", "general"))
        self.assertEqual(features.prompt_length, 0)
        self.assertEqual(features.complexity_score, 0.0)

    def test_extract_input_features_non_empty(self) -> None:
        prompt = PromptSample(
            "p1",
            "quantization adapts per layer sensitivity across hardware modes",
            "research",
        )
        features = extract_input_features(prompt)
        self.assertGreater(features.prompt_length, 0)
        self.assertGreater(features.complexity_score, 0.0)
        self.assertTrue(all(math.isfinite(v) for v in features.to_vector()))

    def test_layer_sensitivity_and_precision_summary(self) -> None:
        prompt = PromptSample("p2", "repeat repeat repeat tokens", "code")
        features = extract_input_features(prompt)
        sensitivity = estimate_layer_sensitivity(prompt, features, num_layers=4)
        self.assertEqual(len(sensitivity.layer_stats), 4)
        summary = summarize_precision_needs(features, sensitivity)
        self.assertGreater(summary, 0.0)
        self.assertLessEqual(summary, 1.4)


if __name__ == "__main__":
    unittest.main()
