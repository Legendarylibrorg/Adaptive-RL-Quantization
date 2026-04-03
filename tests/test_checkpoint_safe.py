from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig

try:
    import torch

    from adaptive_quant.torch_trainer import TorchTrainer, _checkpoint_meta_path
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TorchTrainer = None  # type: ignore[misc,assignment]
    _checkpoint_meta_path = None  # type: ignore[misc,assignment]


@unittest.skipIf(torch is None, "PyTorch not installed")
class CheckpointSafeTests(unittest.TestCase):
    def test_v2_checkpoint_roundtrip_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_backend="pytorch",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                replay_buffer_capacity=0,
                run_name="ckptsafe",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                torch_compile=False,
                torch_preflight=False,
                seed=99,
            )
            t1 = TorchTrainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            t1.global_episode = 42
            t1.update_index = 7
            t1.previous_action = [0.1, 0.2, 0.3]
            t1.training_history = [{"step": 1.0}]
            t1.save_checkpoint(ckpt)

            meta_path = Path(_checkpoint_meta_path(ckpt))
            self.assertTrue(meta_path.is_file())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["format"], 2)
            self.assertEqual(meta["global_episode"], 42)

            resume = config.clone(
                resume_from_checkpoint=ckpt,
                run_name="ckptsafe2",
                allow_legacy_checkpoint_load=False,
            )
            t2 = TorchTrainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")
            self.assertEqual(t2.global_episode, 42)
            self.assertEqual(t2.update_index, 7)
            self.assertEqual(t2.previous_action, [0.1, 0.2, 0.3])
            self.assertEqual(t2.training_history, [{"step": 1.0}])

    def test_legacy_checkpoint_refused_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = Path(temp_dir) / "legacy.pt"
            torch.save(
                {
                    "model_state": {},
                    "optimizer_state": {},
                    "global_episode": 0,
                    "update_index": 0,
                    "previous_action": [0.0, 0.0, 0.0],
                    "training_history": [],
                },
                ckpt,
            )
            config = FrameworkConfig(
                training_backend="pytorch",
                training_episodes=1,
                evaluation_episodes=1,
                stability_probe_count=1,
                replay_buffer_capacity=0,
                run_name="legacytest",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                resume_from_checkpoint=str(ckpt),
                allow_legacy_checkpoint_load=False,
                torch_compile=False,
                torch_preflight=False,
            )
            with self.assertRaises(RuntimeError):
                TorchTrainer(config, log_path=f"{temp_dir}/logs/z.jsonl")

    def test_alternate_policy_objectives_train_one_batch(self) -> None:
        for algo in ("ppo", "vpg", "awr"):
            with self.subTest(algo=algo), tempfile.TemporaryDirectory() as temp_dir:
                config = FrameworkConfig(
                    training_backend="pytorch",
                    training_episodes=2,
                    evaluation_episodes=1,
                    stability_probe_count=1,
                    replay_buffer_capacity=0,
                    run_name=f"algo_{algo}",
                    outputs_dir=temp_dir,
                    log_dir=f"{temp_dir}/logs",
                    benchmark_dir=f"{temp_dir}/benchmarks",
                    analysis_dir=f"{temp_dir}/analysis",
                    checkpoint_dir=f"{temp_dir}/ckpt",
                    torch_compile=False,
                    torch_preflight=False,
                    torch_batch_episodes=2,
                    torch_minibatch_size=2,
                    torch_update_epochs=1,
                    torch_policy_algorithm=algo,
                    seed=101,
                )
                trainer = TorchTrainer(config, log_path=f"{temp_dir}/logs/t.jsonl")
                try:
                    summary = trainer.train()
                finally:
                    trainer.close()
                self.assertIn("mean_reward", summary)

    def test_torch_deterministic_train_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_backend="pytorch",
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                replay_buffer_capacity=8,
                run_name="deterministic_smoke",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                torch_compile=False,
                torch_preflight=False,
                torch_deterministic=True,
                torch_batch_episodes=2,
                torch_minibatch_size=2,
                torch_update_epochs=1,
                seed=202,
            )
            trainer = TorchTrainer(config, log_path=f"{temp_dir}/logs/d.jsonl")
            try:
                summary = trainer.train()
            finally:
                trainer.close()
            self.assertIn("mean_reward", summary)


class LlamaCppPathHardeningTests(unittest.TestCase):
    def test_rejects_newline_in_paths(self) -> None:
        from adaptive_quant.backend import require_llama_cpp_paths

        cfg = FrameworkConfig(
            run_name="path_test",
            backend="llama_cpp",
            llama_cpp_binary="/bin/sh\nevil",
            llama_cpp_model="/tmp/m.gguf",
        )
        with self.assertRaises(ValueError):
            require_llama_cpp_paths(cfg)


if __name__ == "__main__":
    unittest.main()
