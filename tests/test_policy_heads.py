from __future__ import annotations

import random
import unittest

from adaptive_quant.policy_heads import (
    CategoricalHead,
    ValueHead,
    _restore_categorical_head,
)


class PolicyHeadTests(unittest.TestCase):
    def test_categorical_head_deterministic_sample_uses_argmax(self) -> None:
        head = CategoricalHead(2, 3, random.Random(1))
        head.weights = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]
        head.bias = [0.0, 0.0, 0.0]

        index, probabilities = head.sample([1.0, 1.0], random.Random(2), deterministic=True)

        self.assertEqual(index, 2)
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_categorical_head_epsilon_samples_uniform_arm(self) -> None:
        head = CategoricalHead(1, 3, random.Random(1))
        head.weights = [[0.0], [0.0], [0.0]]
        head.bias = [0.0, 10.0, 0.0]
        rng = random.Random(5)

        sampled = {head.sample([1.0], rng, epsilon=1.0)[0] for _ in range(20)}

        self.assertGreater(len(sampled), 1)
        self.assertTrue(sampled.issubset({0, 1, 2}))

    def test_value_head_zero_init_is_available_for_router_baselines(self) -> None:
        head = ValueHead(3, random.Random(1), zero_init=True)

        self.assertEqual(head.weights, [0.0, 0.0, 0.0])
        self.assertEqual(head.predict([1.0, 2.0, 3.0]), 0.0)

    def test_categorical_restore_rejects_non_finite_weights(self) -> None:
        head = CategoricalHead(1, 1, random.Random(1))

        with self.assertRaises(ValueError):
            _restore_categorical_head(head, {"weights": [[float("inf")]], "bias": [0.0]})


if __name__ == "__main__":
    unittest.main()
