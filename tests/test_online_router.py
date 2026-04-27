from __future__ import annotations

import tempfile
import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.online_learning import OnlineLearningLoop
from adaptive_quant.trainer import build_trainer
from adaptive_quant.types import HardwareType, OnlineRequest


class OnlineRouterTests(unittest.TestCase):
    def test_online_router_serves_request_without_optional_deps(self) -> None:
        # No torch/transformers assumed in the base test environment.
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = FrameworkConfig(
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="online_router_smoke",
                stability_probe_count=1,
                backend="router",
                router_enabled=True,
                router_routes=("hf:distilgpt2@q8",),
                online_learning=False,
                training_episodes=2,
                evaluation_episodes=1,
            )
            trainer = build_trainer(cfg)
            loop = OnlineLearningLoop(cfg, trainer=trainer)
            try:
                record = loop.serve_request(
                    OnlineRequest(prompt_text="Summarize deployment risk.", hardware=HardwareType.GPU, prompt_id="router_0")
                )
            finally:
                loop.close()
                trainer.close()

        self.assertTrue(record.get("router_enabled"))
        self.assertIsNotNone(record.get("router_selected_route"))
        self.assertIn("served_metrics", record)


if __name__ == "__main__":
    unittest.main()

