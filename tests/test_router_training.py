from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.presets.post_train import CONFIG_POST_TRAIN
from adaptive_quant.prompts import load_prompt_library_json
from adaptive_quant.router_training import OfflineRouterTrainer, resolve_prompt_library
from adaptive_quant.types import PromptSample, QuantizationDecision, QuantMode


class ResolvePromptLibraryTests(unittest.TestCase):
    def test_loads_post_train_library_from_repo(self) -> None:
        root = Path(__file__).resolve().parent.parent
        config = FrameworkConfig(
            prompt_library_path=str(root / "prompts" / "post_train_library.json")
        )
        library = resolve_prompt_library(config)
        self.assertIsNotNone(library)
        assert library is not None
        self.assertGreaterEqual(len(library.prompts), 20)

    def test_named_json_loader_matches_helper(self) -> None:
        root = Path(__file__).resolve().parent.parent
        path = root / "prompts" / "post_train_library.json"
        library = load_prompt_library_json(path)
        self.assertEqual(
            len(library.prompts),
            len(resolve_prompt_library(FrameworkConfig(prompt_library_path=str(path))).prompts),
        )


class PostTrainPresetTests(unittest.TestCase):
    def test_post_train_preset_enables_long_routed_training(self) -> None:
        self.assertTrue(CONFIG_POST_TRAIN.continuous_training)
        self.assertGreaterEqual(CONFIG_POST_TRAIN.max_training_episodes, 50_000)
        self.assertTrue(CONFIG_POST_TRAIN.router_enabled)
        self.assertEqual(CONFIG_POST_TRAIN.env_sampling_mode, "sequential")
        self.assertIsNotNone(CONFIG_POST_TRAIN.prompt_library_path)


class OfflineRouterTrainerTests(unittest.TestCase):
    def test_prepare_decision_annotates_route_metadata(self) -> None:
        config = FrameworkConfig(
            router_enabled=True,
            router_routes=("hf:openai-community/gpt2@q4", "hf:openai-community/gpt2@q8"),
            router_feature_backend="hash",
            seed=7,
        )
        router = OfflineRouterTrainer(config)
        state = mock.Mock()
        state.prompt = PromptSample("p1", "Plan a long RL post-training run.", "systems")
        policy_decision = QuantizationDecision(
            mode=QuantMode.DISCRETE, base_bit_width=4, metadata={"head": "policy"}
        )
        with mock.patch.object(router.router, "route") as route_mock:
            from adaptive_quant.routing import RouteCandidate, RouterTrace

            route_mock.return_value = (
                RouteCandidate(backend="hf", model_id="openai-community/gpt2", quant_bits=8),
                RouterTrace(
                    feature_vector=[0.1],
                    selected_index=1,
                    probabilities=[0.4, 0.6],
                    value_prediction=0.0,
                ),
            )
            routed = router.prepare_decision(policy_decision, state)
        self.assertEqual(routed.base_bit_width, 8)
        self.assertEqual(routed.metadata.get("route"), "hf:openai-community/gpt2@q8")
        self.assertEqual(routed.metadata.get("head"), "router")

    def test_complete_episode_updates_router_from_baseline_comparison(self) -> None:
        config = FrameworkConfig(
            router_enabled=True,
            router_routes=("hf:openai-community/gpt2@q4", "hf:openai-community/gpt2@q8"),
            router_feature_backend="hash",
            seed=3,
        )
        router = OfflineRouterTrainer(config)
        router._pending_trace = mock.Mock()
        state = mock.Mock()
        state.prompt = PromptSample("p1", "Evaluate quantization tradeoffs.", "systems")
        policy_decision = QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4)
        routed_result = mock.Mock()
        routed_result.metrics.perplexity = 1.2
        routed_result.metrics.memory_mb = 100.0
        routed_result.metrics.latency_ms = 10.0
        baseline_result = mock.Mock()
        baseline_result.metrics.perplexity = 1.5
        env = mock.Mock()
        env.evaluate_current.return_value = baseline_result
        with mock.patch.object(router.router, "update") as update_mock:
            router.complete_episode(
                state=state,
                policy_decision=policy_decision,
                routed_result=routed_result,
                env=env,
                episode_index=12,
            )
        update_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
