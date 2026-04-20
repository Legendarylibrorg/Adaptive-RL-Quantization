from __future__ import annotations

import tempfile
import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv


class PromptSplitTests(unittest.TestCase):
    def test_prompt_split_separates_train_and_eval_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                prompt_split_enabled=True,
                prompt_split_seed=123,
                prompt_train_fraction=0.7,
                training_episodes=4,
                evaluation_episodes=4,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="prompt_split_test",
                seed=9,
            )
            env = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/prompt_split.jsonl")
            self.assertIsNotNone(env.train_prompt_ids)
            self.assertIsNotNone(env.eval_prompt_ids)
            assert env.train_prompt_ids is not None
            assert env.eval_prompt_ids is not None
            self.assertTrue(env.train_prompt_ids)
            self.assertTrue(env.eval_prompt_ids)
            self.assertTrue(env.train_prompt_ids.isdisjoint(env.eval_prompt_ids))

            train_ids = {env.reset(phase="train").prompt.prompt_id for _ in range(50)}
            eval_ids = {env.reset(phase="eval").prompt.prompt_id for _ in range(50)}
            self.assertTrue(train_ids.issubset(env.train_prompt_ids))
            self.assertTrue(eval_ids.issubset(env.eval_prompt_ids))

    def test_prompt_split_random_sampling_is_stable_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                prompt_split_enabled=True,
                prompt_split_seed=123,
                prompt_train_fraction=0.7,
                training_episodes=4,
                evaluation_episodes=4,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="prompt_split_repro_test",
                seed=9,
            )
            env_a = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/prompt_split_a.jsonl")
            env_b = AdaptiveQuantizationEnv(config, log_path=f"{temp_dir}/logs/prompt_split_b.jsonl")

            train_seq_a = [env_a.reset(phase="train").prompt.prompt_id for _ in range(20)]
            train_seq_b = [env_b.reset(phase="train").prompt.prompt_id for _ in range(20)]
            eval_seq_a = [env_a.reset(phase="eval").prompt.prompt_id for _ in range(20)]
            eval_seq_b = [env_b.reset(phase="eval").prompt.prompt_id for _ in range(20)]

            self.assertEqual(train_seq_a, train_seq_b)
            self.assertEqual(eval_seq_a, eval_seq_b)


if __name__ == "__main__":
    unittest.main()
