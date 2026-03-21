from __future__ import annotations

import tempfile
import unittest

from config_4090_universal import CONFIG_4090_UNIVERSAL
from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.online_learning import analyze as analyze_online
from analysis.quant_function_behavior import analyze as analyze_quant
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.gpu_profiles import apply_gpu_profile, infer_gpu_profile
from adaptive_quant.online_learning import OnlineLearningLoop
from adaptive_quant.policy import UniversalQuantizationPolicy
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.research_pipeline import ResearchPipeline
from adaptive_quant.trainer import Trainer, build_trainer
from adaptive_quant.types import HardwareType, OnlineRequest, QuantMode, QuantizationDecision


class FrameworkTests(unittest.TestCase):
    def test_4090_universal_preset_is_multi_hardware(self) -> None:
        self.assertEqual(CONFIG_4090_UNIVERSAL.training_host_label, "rtx4090")
        self.assertTrue(CONFIG_4090_UNIVERSAL.multi_hardware)
        self.assertEqual(CONFIG_4090_UNIVERSAL.hardware_modes, ("gpu", "cpu", "low_resource"))

    def test_state_contains_hardware_and_input_features(self) -> None:
        config = FrameworkConfig(training_episodes=4, evaluation_episodes=2, stability_probe_count=1, run_name="state_test")
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/state_test.jsonl")
        state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
        vector = state.to_vector(config.ordered_hardware())

        self.assertGreater(len(vector), config.num_layers)
        self.assertEqual(vector[0], 1.0)
        self.assertGreater(state.input_features.complexity_score, 0.0)

    def test_learned_quantization_is_clamped(self) -> None:
        config = FrameworkConfig(training_episodes=4, evaluation_episodes=2, stability_probe_count=1, run_name="clamp_test")
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/clamp_test.jsonl")
        state = env.reset(forced_hardware=HardwareType.CPU, forced_prompt_id="very_complex")
        decision = QuantizationDecision(
            mode=QuantMode.LEARNED,
            scale_factor=9.0,
            clipping_range=0.01,
            precision_level=4.5,
        )
        finalized = finalize_decision(decision, state, config)

        self.assertTrue(all(min(config.discrete_bit_widths) <= bit <= max(config.discrete_bit_widths) for bit in finalized.effective_layer_bits))
        self.assertGreaterEqual(finalized.scale_factor, config.scale_bounds[0])
        self.assertLessEqual(finalized.scale_factor, config.scale_bounds[1])

    def test_benchmark_and_analysis_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_episodes=10,
                evaluation_episodes=4,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="bench_test",
                seed=7,
            )
            results = BenchmarkSuite(config).run()
            self.assertIn("single_vs_multi", results)
            self.assertIn("static_vs_dynamic", results)
            self.assertIn("discrete_vs_learned", results)

            log_path = f"{config.log_dir}/{config.run_name}_learned.jsonl"
            hardware = analyze_hardware(log_path, f"{config.analysis_dir}/hardware")
            inputs = analyze_inputs(log_path, f"{config.analysis_dir}/inputs")
            quant = analyze_quant(log_path, f"{config.analysis_dir}/quant")

            self.assertIn("reward_by_hardware", hardware)
            self.assertIn("by_complexity", inputs)
            self.assertIn("learned_episode_count", quant)

    def test_build_trainer_uses_python_backend_by_default(self) -> None:
        config = FrameworkConfig(training_episodes=2, evaluation_episodes=1, stability_probe_count=1, run_name="factory_test")
        trainer = build_trainer(config, log_path=f"{tempfile.gettempdir()}/factory_test.jsonl")
        self.assertIsInstance(trainer, Trainer)

    def test_rollout_carries_previous_action_between_episodes(self) -> None:
        config = FrameworkConfig(training_episodes=2, evaluation_episodes=2, stability_probe_count=1, run_name="rollout_test")
        trainer = build_trainer(config, log_path=f"{tempfile.gettempdir()}/rollout_test.jsonl")
        results = trainer.rollout(2)

        self.assertEqual(len(results), 2)
        expected_previous = results[0].decision.feedback_vector(
            max_bits=max(config.discrete_bit_widths),
            scale_upper=config.scale_bounds[1],
            clip_upper=config.clip_bounds[1],
        )
        self.assertEqual(results[1].state.previous_action, expected_previous)

    def test_single_probe_stability_short_circuits_to_zero(self) -> None:
        config = FrameworkConfig(training_episodes=2, evaluation_episodes=1, stability_probe_count=1, run_name="stability_short_circuit")
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/stability_short_circuit.jsonl")
        state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
        decision = QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4)
        result = env.evaluate_current(decision)

        self.assertEqual(state.prompt.prompt_id, "very_complex")
        self.assertEqual(result.metrics.stability_penalty, 0.0)

    def test_moe_state_and_policy_attach_packed_variants(self) -> None:
        config = FrameworkConfig(
            training_episodes=2,
            evaluation_episodes=1,
            stability_probe_count=1,
            run_name="moe_test",
            moe_enabled=True,
            moe_num_experts=12,
            moe_top_k=2,
        )
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/moe_test.jsonl")
        state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
        self.assertIsNotNone(state.moe_context)
        assert state.moe_context is not None
        self.assertEqual(len(state.moe_context.experts), 2)
        self.assertEqual(len(state.to_vector(config.ordered_hardware())), config.state_vector_dim())

        policy = UniversalQuantizationPolicy(config)
        decision, _trace = policy.act(state, deterministic=True)
        finalized = finalize_decision(decision, state, config)
        self.assertEqual(len(finalized.moe_variant_indices), 2)
        self.assertEqual(len(finalized.moe_variant_names), 2)
        self.assertTrue(all(name in config.moe_variant_names for name in finalized.moe_variant_names))
        self.assertTrue(finalized.metadata["moe_enabled"])

    def test_moe_backend_reports_swap_and_cache_penalties(self) -> None:
        config = FrameworkConfig(
            training_episodes=2,
            evaluation_episodes=1,
            stability_probe_count=1,
            run_name="moe_metrics_test",
            moe_enabled=True,
            moe_num_experts=12,
            moe_top_k=2,
        )
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/moe_metrics_test.jsonl")
        state = env.reset(forced_hardware=HardwareType.LOW_RESOURCE, forced_prompt_id="very_complex")
        decision = finalize_decision(QuantizationDecision(mode=QuantMode.DISCRETE, base_bit_width=4), state, config)
        result = env.evaluate_current(decision)
        self.assertGreaterEqual(result.metrics.swap_cost_ms, 0.0)
        self.assertGreaterEqual(result.metrics.cache_miss_count, 0.0)
        self.assertGreaterEqual(result.metrics.variant_churn, 0.0)

    def test_gpu_profile_inference_and_application(self) -> None:
        self.assertEqual(infer_gpu_profile("NVIDIA GeForce RTX 4090", 24.0), "rtx4090")
        self.assertEqual(infer_gpu_profile("NVIDIA A100-SXM4-80GB", 80.0), "a100_80gb")
        self.assertEqual(infer_gpu_profile("Generic 12GB GPU", 12.0), "rtx4070")

        config = FrameworkConfig(training_backend="pytorch", torch_gpu_profile="auto", run_name="gpu_profile_test")
        tuned, metadata = apply_gpu_profile(config, device_name="NVIDIA GeForce RTX 4080", total_memory_gb=16.0)
        self.assertEqual(tuned.torch_gpu_profile, "rtx4080")
        self.assertEqual(metadata["selected_gpu_profile"], "rtx4080")
        self.assertGreaterEqual(tuned.torch_batch_episodes, tuned.torch_minibatch_size)

    def test_online_learning_loop_updates_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_episodes=4,
                evaluation_episodes=2,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                run_name="online_test",
                online_learning=True,
                online_exploration_rate=1.0,
                online_canary_ratio=1.0,
                online_replay_capacity=32,
                online_min_replay_size=2,
                online_update_interval=2,
                online_batch_size=4,
                online_drift_window=4,
                online_drift_reward_delta=999.0,
                seed=11,
            )
            trainer = build_trainer(config)
            loop = OnlineLearningLoop(config, trainer=trainer)
            requests = [
                OnlineRequest(prompt_text=f"Summarize deployment risk {index}.", hardware=HardwareType.GPU, prompt_id=f"online_{index}")
                for index in range(6)
            ]

            try:
                summary = loop.run_stream(requests)
            finally:
                loop.close()
                trainer.close()

            analysis = analyze_online(config.online_telemetry_path(), f"{config.analysis_dir}/online")
            self.assertEqual(summary["requests"], 6)
            self.assertGreater(summary["total_updates"], 0)
            self.assertIn("candidate_accept_rate", analysis)
            self.assertIn("reward_by_hardware", analysis)

    def test_research_pipeline_writes_summary_history_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_episodes=12,
                evaluation_episodes=4,
                benchmark_training_episodes=6,
                benchmark_evaluation_episodes=3,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="pipeline_test",
                seed=5,
            )
            summary = ResearchPipeline(config).run()

            self.assertIn("train", summary)
            self.assertIn("evaluation", summary)
            self.assertIn("benchmarks", summary)
            self.assertTrue(summary["artifacts"]["training_history"].endswith("_training_history.json"))
            self.assertTrue(summary["artifacts"]["report"].endswith("_report.md"))

    def test_moe_pipeline_writes_moe_benchmarks_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_episodes=10,
                evaluation_episodes=4,
                benchmark_training_episodes=4,
                benchmark_evaluation_episodes=2,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="moe_pipeline_suite_test",
                moe_enabled=True,
                moe_num_experts=12,
                moe_top_k=2,
                seed=19,
            )
            summary = ResearchPipeline(config).run()

            self.assertIn("dense_vs_moe", summary["benchmarks"])
            self.assertIn("moe_packed_vs_single_variant", summary["benchmarks"])
            self.assertIn("moe_static_vs_rl", summary["benchmarks"])
            self.assertIn("moe_experts", summary["analysis"])
            self.assertIn("moe_cache", summary["analysis"])
            self.assertTrue(summary["artifacts"]["report"].endswith("_report.md"))


if __name__ == "__main__":
    unittest.main()
