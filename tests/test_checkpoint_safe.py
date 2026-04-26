from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.trainer import Trainer
from adaptive_quant.types import HardwareType

try:
    import torch

    from adaptive_quant.torch_policy import TorchActorCritic
    from adaptive_quant.torch_trainer import (
        TorchTrainer,
        _checkpoint_meta_path,
        _torch_load_v2_tensor_file,
    )
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TorchTrainer = None  # type: ignore[misc,assignment]
    TorchActorCritic = None  # type: ignore[misc,assignment]
    _checkpoint_meta_path = None  # type: ignore[misc,assignment]
    _torch_load_v2_tensor_file = None  # type: ignore[misc,assignment]


class PythonCheckpointSafeTests(unittest.TestCase):
    def test_python_checkpoint_roundtrip_restores_policy_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_episodes=3,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="py_ckpt_safe",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=33,
            )
            t1 = Trainer(config, log_path=f"{temp_dir}/logs/x.jsonl")
            t1.train()
            saved_path = t1.save_checkpoint(ckpt)

            state1 = t1.env.reset(
                previous_action=t1.previous_action,
                forced_hardware=HardwareType.GPU,
                forced_prompt_id="very_complex",
                phase="eval",
                episode_index=999,
            )
            decision1, _ = t1.policy.act(state1, deterministic=True)

            resume = config.clone(
                resume_from_checkpoint=ckpt,
                run_name="py_ckpt_safe_resume",
            )
            t2 = Trainer(resume, log_path=f"{temp_dir}/logs/y.jsonl")
            state2 = t2.env.reset(
                previous_action=t2.previous_action,
                forced_hardware=HardwareType.GPU,
                forced_prompt_id="very_complex",
                phase="eval",
                episode_index=999,
            )
            decision2, _ = t2.policy.act(state2, deterministic=True)

            self.assertTrue(saved_path.endswith(".json"))
            self.assertEqual(t2.completed_episodes, 3)
            self.assertEqual(t2.previous_action, t1.previous_action)
            self.assertEqual(t2.training_history, t1.training_history)
            self.assertEqual(decision2, decision1)

    def test_python_checkpoint_rejects_incompatible_policy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = str(Path(temp_dir) / "trial_final.pt")
            config = FrameworkConfig(
                training_episodes=2,
                evaluation_episodes=1,
                stability_probe_count=1,
                run_name="py_ckpt_shape_a",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                seed=19,
            )
            trainer = Trainer(config, log_path=f"{temp_dir}/logs/base.jsonl")
            trainer.train()
            trainer.save_checkpoint(ckpt)
            trainer.close()

            incompatible = config.clone(
                run_name="py_ckpt_shape_b",
                resume_from_checkpoint=ckpt,
                num_layers=config.num_layers + 2,
            )
            with self.assertRaises(ValueError) as ctx:
                Trainer(incompatible, log_path=f"{temp_dir}/logs/resume.jsonl")
            self.assertIn("checkpoint shape mismatch", str(ctx.exception))


@unittest.skipIf(torch is None, "PyTorch not installed")
class TorchCheckpointSafeTests(unittest.TestCase):
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

    def test_v2_loader_does_not_fall_back_to_pickle_by_default(self) -> None:
        """A corrupt v2 tensor file must not silently re-attempt with weights_only=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = Path(temp_dir) / "broken.pt"
            ckpt.write_bytes(b"this is not a torch file")
            # The exact exception type comes from torch's pickle/zip parser; we only require
            # that the loader fails fast rather than silently re-trying with legacy pickle.
            with self.assertRaises(Exception):  # noqa: B017 - intentionally broad
                _torch_load_v2_tensor_file(str(ckpt))

    def test_v2_loader_allow_legacy_does_not_silently_succeed_on_garbage(self) -> None:
        """allow_legacy=True relaxes the fallback path but still surfaces unparsable files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = Path(temp_dir) / "broken.pt"
            ckpt.write_bytes(b"this is not a torch file")
            with self.assertRaises(Exception):  # noqa: B017 - intentionally broad
                _torch_load_v2_tensor_file(str(ckpt), allow_legacy=True)

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

    def test_torch_actor_critic_forward_shape_cpu(self) -> None:
        """Smoke test the actor-critic on CPU: heads emit the expected per-batch dimensions."""
        config = FrameworkConfig(
            training_backend="pytorch",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            run_name="actor_shape",
            torch_compile=False,
            torch_preflight=False,
            torch_mlp_depth=1,
            torch_hidden_dim=16,
            seed=11,
        )
        state_dim = config.state_vector_dim()
        num_bits = len(config.discrete_bit_widths)
        num_modes = len(config.supported_modes())
        model = TorchActorCritic(config)
        model.eval()
        with torch.no_grad():
            x = torch.zeros(3, state_dim, dtype=torch.float32)
            out = model(x)
        self.assertEqual(out["mode_logits"].shape, (3, num_modes))
        self.assertEqual(out["discrete_logits"].shape, (3, num_bits))
        self.assertEqual(out["group_logits"].shape, (3, config.num_groups, num_bits))
        self.assertEqual(out["layer_logits"].shape, (3, config.num_layers, num_bits))
        self.assertEqual(out["learned_mean"].shape, (3, 3))
        self.assertEqual(out["learned_std"].shape, (3, 3))
        self.assertEqual(out["value"].shape, (3,))

    def test_train_continuous_rejects_eval_interval_smaller_than_batch(self) -> None:
        """The continuous-training guard refuses configs that would skip the eval trigger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_backend="pytorch",
                continuous_training=True,
                max_training_episodes=4,
                evaluation_episodes=1,
                stability_probe_count=1,
                replay_buffer_capacity=0,
                run_name="eval_interval_guard",
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/ckpt",
                torch_compile=False,
                torch_preflight=False,
                torch_batch_episodes=4,
                torch_minibatch_size=2,
                torch_update_epochs=1,
                eval_interval=2,
                checkpoint_interval=0,
                seed=303,
            )
            trainer = TorchTrainer(config, log_path=f"{temp_dir}/logs/g.jsonl")
            try:
                with self.assertRaises(ValueError) as ctx:
                    trainer.train()
                self.assertIn("eval_interval", str(ctx.exception))
            finally:
                trainer.close()

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
    def test_rejects_newline_in_paths_at_config(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            FrameworkConfig(
                run_name="path_test",
                backend="llama_cpp",
                llama_cpp_binary="/bin/sh\nevil",
                llama_cpp_model="/tmp/m.gguf",
            )
        self.assertIn("llama_cpp_binary", str(ctx.exception))

    def test_rejects_newline_in_model_path_at_config(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="path_test",
                backend="llama_cpp",
                llama_cpp_binary="/bin/sh",
                llama_cpp_model="/tmp/m\neg",
            )


if __name__ == "__main__":
    unittest.main()
