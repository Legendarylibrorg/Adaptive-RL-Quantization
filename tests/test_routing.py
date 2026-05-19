from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.routing import EfficientTaskRouter, _router_hf_pretrained_kwargs, parse_route


class RoutingTests(unittest.TestCase):
    def test_router_hf_pretrained_kwargs_safetensors_and_no_trust_remote_code(self) -> None:
        cfg = FrameworkConfig(
            run_name="hf_kw_test",
            router_hf_embedding_revision="main",
            router_hf_local_files_only=True,
        )
        tokenizer_kw, model_kw = _router_hf_pretrained_kwargs(cfg)
        self.assertFalse(tokenizer_kw["trust_remote_code"])
        self.assertEqual(tokenizer_kw["revision"], "main")
        self.assertTrue(tokenizer_kw["local_files_only"])
        self.assertTrue(model_kw["use_safetensors"])
        self.assertFalse(model_kw["trust_remote_code"])
        self.assertEqual(model_kw["revision"], "main")

    def test_parse_route_accepts_model_only_or_model_with_bits(self) -> None:
        self.assertEqual(parse_route("hf:org/model").model_id, "org/model")
        self.assertIsNone(parse_route("hf:org/model").quant_bits)
        parsed = parse_route("hf:org/model@q4")
        self.assertEqual(parsed.model_id, "org/model")
        self.assertEqual(parsed.quant_bits, 4)
        self.assertEqual(parsed.backend, "hf")

    def test_parse_route_accepts_llama_cpp_prefix(self) -> None:
        parsed = parse_route("llama_cpp:/models/foo.gguf@q4")
        self.assertEqual(parsed.backend, "llama_cpp")
        self.assertEqual(parsed.model_id, "/models/foo.gguf")
        self.assertEqual(parsed.quant_bits, 4)

    def test_reward_penalizes_regression(self) -> None:
        cfg = FrameworkConfig(
            run_name="routing_reward_test",
            stability_probe_count=1,
            router_enabled=True,
            router_routes=("hf:org/a@q4", "hf:org/b@q8"),
        )
        router = EfficientTaskRouter(cfg)
        ok = router.reward_from_metrics(memory_mb=800.0, perplexity=10.2, baseline_perplexity=10.0)
        bad = router.reward_from_metrics(memory_mb=300.0, perplexity=20.0, baseline_perplexity=10.0)
        self.assertLess(bad, ok)

    def test_router_learns_prefer_lower_memory_when_quality_ok(self) -> None:
        cfg = FrameworkConfig(
            run_name="routing_learn_test",
            stability_probe_count=1,
            router_enabled=True,
            router_routes=("hf:example/low@q4", "hf:example/high@q8"),
            router_exploration=0.0,
        )
        router = EfficientTaskRouter(cfg)
        task = "Summarize an email thread about deployment risks."
        baseline = 10.0

        # Train: route 0 is good (low memory, ok ppl); route 1 is worse (high memory).
        for _ in range(600):
            route, trace = router.route(task_text=task, deterministic=False)
            if "low" in route.model_id:
                reward = router.reward_from_metrics(
                    memory_mb=300.0, perplexity=10.1, baseline_perplexity=baseline
                )
            else:
                reward = router.reward_from_metrics(
                    memory_mb=900.0, perplexity=10.1, baseline_perplexity=baseline
                )
            router.update(trace, reward=reward)

        # Evaluate: greedy should pick low-memory.
        chosen, _trace = router.route(task_text=task, deterministic=True)
        self.assertIn("low", chosen.model_id)


if __name__ == "__main__":
    unittest.main()
