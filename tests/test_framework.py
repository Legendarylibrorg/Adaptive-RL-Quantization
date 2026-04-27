from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.gpu_profiles import apply_gpu_profile, infer_gpu_profile
from adaptive_quant.hardware import (
    DetectedHardware,
    host_aware_hardware_profiles,
)
from adaptive_quant.online_learning import OnlineLearningLoop, build_request_stream
from adaptive_quant.online_pipeline import run_online_pipeline_entrypoint
from adaptive_quant.policy import UniversalQuantizationPolicy
from adaptive_quant.quantization import finalize_decision
from adaptive_quant.recommendation import recommend_quantization
from adaptive_quant.research_pipeline import ResearchPipeline
from adaptive_quant.trainer import Trainer, build_trainer
from adaptive_quant.trainer_utils import summarize_episode_results, training_row
from adaptive_quant.types import (
    HardwareType,
    OnlineRequest,
    QuantizationDecision,
    QuantMode,
)
from analysis.analyzers import (
    analyze_hardware,
    analyze_inputs,
    analyze_online,
    analyze_quant,
)
from config_4090_universal import CONFIG_4090_UNIVERSAL


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

    def test_benchmark_releases_each_trainer_before_building_the_next(self) -> None:
        config = FrameworkConfig(training_episodes=2, evaluation_episodes=1, stability_probe_count=1, run_name="bench_release_test")
        suite = BenchmarkSuite(config)
        variants = {
            "baseline": config.clone(run_name="bench_release_baseline"),
            "adaptive": config.clone(run_name="bench_release_adaptive"),
            "learned": config.clone(run_name="bench_release_learned"),
        }
        live_trainers = 0
        max_live_trainers = 0
        closed: list[str] = []

        class FakeTrainer:
            def __init__(self, name: str) -> None:
                nonlocal live_trainers, max_live_trainers
                self.name = name
                live_trainers += 1
                max_live_trainers = max(max_live_trainers, live_trainers)

            def train(self) -> dict[str, object]:
                return {"run_name": self.name}

            def evaluate(self, hardware: HardwareType | None = None) -> dict[str, object]:
                return {
                    "mean_reward": 1.0,
                    "hardware": hardware.value if hardware is not None else "all",
                }

            def close(self) -> None:
                nonlocal live_trainers
                live_trainers -= 1
                closed.append(self.name)

        def fake_build_trainer(trainer_config: FrameworkConfig, log_path: str | None = None) -> FakeTrainer:
            del log_path
            return FakeTrainer(trainer_config.run_name)

        with mock.patch("adaptive_quant.benchmark.build_trainer", side_effect=fake_build_trainer):
            train, evaluation = suite._run_variants(variants)

        self.assertEqual(max_live_trainers, 1)
        self.assertEqual(live_trainers, 0)
        self.assertEqual(
            closed,
            [variant.run_name for variant in variants.values()],
        )
        self.assertEqual(train["baseline"]["run_name"], "bench_release_baseline")
        self.assertEqual(evaluation["learned"]["hardware"], "all")

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

    def test_torch_policy_algorithm_must_be_known(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="algo_validation_test", torch_policy_algorithm="invalid_algo")

    def test_env_sampling_sequential_is_deterministic_in_prompt_and_hardware(self) -> None:
        cfg = FrameworkConfig(
            env_sampling_mode="sequential",
            multi_hardware=True,
            run_name="seq_sampling_test",
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
        )
        env = AdaptiveQuantizationEnv(cfg, log_path=f"{tempfile.gettempdir()}/seq_sampling.jsonl")
        modes = cfg.ordered_hardware()
        for ep in range(3):
            state = env.reset(episode_index=ep, phase="train")
            self.assertEqual(state.hardware_profile.hardware_type, modes[ep % len(modes)])
        p0 = env.reset(episode_index=0, phase="train").prompt.prompt_id
        p0_again = env.reset(episode_index=0, phase="train").prompt.prompt_id
        self.assertEqual(p0, p0_again)

    def test_env_sampling_forced_requires_defaults(self) -> None:
        cfg = FrameworkConfig(
            env_sampling_mode="forced",
            run_name="forced_sampling_test",
            training_episodes=1,
            stability_probe_count=1,
        )
        env = AdaptiveQuantizationEnv(cfg, log_path=f"{tempfile.gettempdir()}/forced_bad.jsonl")
        with self.assertRaises(ValueError):
            env.reset()

    def test_reproducible_research_preset_matches_documented_toggles(self) -> None:
        cfg = FrameworkConfig.reproducible_research(seed=777, run_name="preset_test", training_episodes=4)
        self.assertEqual(cfg.seed, 777)
        self.assertEqual(cfg.prompt_split_seed, 777)
        self.assertEqual(cfg.env_sampling_mode, "sequential")
        self.assertEqual(cfg.rl_train_policy_mode, "deterministic")
        self.assertEqual(cfg.stability_probe_sampling, "deterministic")
        self.assertTrue(cfg.torch_deterministic)
        self.assertFalse(cfg.torch_compile)
        self.assertTrue(cfg.rl_train_deterministic())
        self.assertEqual(cfg.training_episodes, 4)

    def test_research_env_config_validators(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_env_mode", env_sampling_mode="unknown")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_policy_mode", rl_train_policy_mode="maybe")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_probe_mode", stability_probe_sampling="coinflip")

    def test_reward_perplexity_reference_hinge_penalizes_over_ref(self) -> None:
        base = FrameworkConfig(stability_probe_count=1, run_name="ppl_ref_env", training_episodes=1, evaluation_episodes=1)
        env = AdaptiveQuantizationEnv(base, log_path=f"{tempfile.gettempdir()}/ppl_ref.jsonl")
        metrics = {
            "latency_ms": 10.0,
            "throughput_tps": 100.0,
            "perplexity": 14.0,
            "memory_mb": 512.0,
        }
        r_base = env._compute_reward(metrics, 0.0)
        w = replace(base.reward_weights, zeta_perplexity_over_ref=2.5)
        cfg = base.clone(
            reward_weights=w,
            reward_perplexity_reference=10.0,
        )
        env2 = AdaptiveQuantizationEnv(cfg, log_path=f"{tempfile.gettempdir()}/ppl_ref2.jsonl")
        r_guard = env2._compute_reward(metrics, 0.0)
        self.assertLess(r_guard, r_base)
        self.assertAlmostEqual(r_base - r_guard, 2.5 * (14.0 - 10.0))

    def test_token_efficiency_reward_penalizes_slow_routes(self) -> None:
        base = FrameworkConfig(stability_probe_count=1, run_name="route_eff_reward", training_episodes=1, evaluation_episodes=1)
        w = replace(base.reward_weights, eta_token_latency=0.5)
        cfg = base.clone(reward_weights=w)
        env = AdaptiveQuantizationEnv(cfg, log_path=f"{tempfile.gettempdir()}/route_eff_reward.jsonl")
        fast_metrics = {
            "latency_ms": 100.0,
            "throughput_tps": 120.0,
            "perplexity": 8.0,
            "memory_mb": 900.0,
            "latency_ms_per_token": 1.0,
        }
        slow_metrics = {
            **fast_metrics,
            "latency_ms_per_token": 5.0,
        }

        self.assertLess(env._compute_reward(slow_metrics, 0.0), env._compute_reward(fast_metrics, 0.0))

    def test_backend_reports_per_token_latency_metrics(self) -> None:
        config = FrameworkConfig(training_episodes=2, evaluation_episodes=1, stability_probe_count=1, run_name="per_tok_metrics")
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/per_tok_metrics.jsonl")
        state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
        decision = QuantizationDecision(mode=QuantMode.LEARNED, precision_level=0.25)
        result = env.evaluate_current(decision)

        self.assertEqual(result.metrics.tokens_processed, float(state.input_features.prompt_length))
        self.assertGreater(result.metrics.latency_ms_per_token, 0.0)
        self.assertIn("latency_ms_per_token", training_row(1.0, result))
        summary = summarize_episode_results([result])
        self.assertIn("mean_tokens_processed", summary)
        self.assertGreater(summary["mean_latency_ms_per_token"], 0.0)

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
        self.assertEqual(infer_gpu_profile("NVIDIA GeForce RTX 3090", 24.0), "rtx3090")
        self.assertEqual(infer_gpu_profile("NVIDIA A100-SXM4-80GB", 80.0), "a100_80gb")
        self.assertEqual(infer_gpu_profile("Generic 12GB GPU", 12.0), "rtx4070")

        config = FrameworkConfig(training_backend="pytorch", torch_gpu_profile="auto", run_name="gpu_profile_test")
        tuned, metadata = apply_gpu_profile(config, device_name="NVIDIA GeForce RTX 4080", total_memory_gb=16.0)
        self.assertEqual(tuned.torch_gpu_profile, "rtx4080")
        self.assertEqual(metadata["selected_gpu_profile"], "rtx4080")
        self.assertGreaterEqual(tuned.torch_batch_episodes, tuned.torch_minibatch_size)

    def test_host_aware_profiles_follow_detected_machine(self) -> None:
        detected = DetectedHardware(
            system="linux",
            machine="x86_64",
            cpu_count=32,
            total_memory_gb=64.0,
            accelerator_type=HardwareType.GPU,
            accelerator_name="NVIDIA GeForce RTX 4090",
            accelerator_memory_gb=24.0,
            accelerator_profile="rtx4090",
            cuda_available=True,
        )
        profiles = host_aware_hardware_profiles(detected)
        self.assertGreater(profiles[HardwareType.CPU].memory_budget_mb, 7_500.0)
        self.assertGreater(profiles[HardwareType.GPU].compute_factor, 1.85)
        self.assertGreater(profiles[HardwareType.GPU].preferred_bits, profiles[HardwareType.CPU].preferred_bits)

    def test_host_aware_3090_uses_ampere_template_not_fallback(self) -> None:
        detected = DetectedHardware(
            system="linux",
            machine="x86_64",
            cpu_count=16,
            total_memory_gb=32.0,
            accelerator_type=HardwareType.GPU,
            accelerator_name="NVIDIA GeForce RTX 3090",
            accelerator_memory_gb=24.0,
            accelerator_profile="rtx3090",
            cuda_available=True,
        )
        profiles = host_aware_hardware_profiles(detected)
        self.assertEqual(profiles[HardwareType.GPU].name, "rtx3090")
        self.assertGreater(profiles[HardwareType.GPU].compute_factor, 1.85)

    def test_build_request_stream_preserves_library_prompt_ids(self) -> None:
        config = FrameworkConfig(
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            run_name="online_request_ids_test",
            seed=11,
        )
        requests = build_request_stream(config, request_count=12)
        env = AdaptiveQuantizationEnv(config, log_path=f"{tempfile.gettempdir()}/online_request_ids.jsonl")

        seen_ids = set()
        for request in requests:
            prompt = env.prompt_library.by_id(request.prompt_id)
            self.assertEqual(request.prompt_text, prompt.text)
            self.assertEqual(request.prompt_domain, prompt.domain)
            seen_ids.add(request.prompt_id)

        self.assertTrue(seen_ids)

    def test_build_request_stream_honors_zero_request_count(self) -> None:
        config = FrameworkConfig(
            training_episodes=1,
            evaluation_episodes=1,
            stability_probe_count=1,
            run_name="online_request_zero_test",
            online_requests=12,
            seed=17,
        )
        self.assertEqual(build_request_stream(config, request_count=0), [])

    def test_evaluate_honors_zero_episode_override(self) -> None:
        config = FrameworkConfig(
            training_episodes=2,
            evaluation_episodes=3,
            stability_probe_count=1,
            run_name="eval_zero_override_test",
        )
        trainer = build_trainer(config, log_path=f"{tempfile.gettempdir()}/eval_zero_override.jsonl")
        try:
            summary = trainer.evaluate(episodes=0)
        finally:
            trainer.close()

        self.assertEqual(summary["mean_reward"], 0.0)
        self.assertEqual(summary["mean_latency_ms"], 0.0)

    def test_recommend_quantization_returns_best_fixed_candidate(self) -> None:
        config = FrameworkConfig(
            training_episodes=8,
            evaluation_episodes=4,
            recommendation_eval_episodes=4,
            recommendation_candidate_limit=4,
            stability_probe_count=1,
            run_name="recommend_test",
        )
        trainer = build_trainer(config, log_path=f"{tempfile.gettempdir()}/recommend_test.jsonl")
        try:
            trainer.train()
            recommendation = recommend_quantization(trainer, config)
        finally:
            trainer.close()

        self.assertIn(recommendation["target_hardware"], {"gpu", "cpu", "low_resource"})
        self.assertGreater(recommendation["candidate_count"], 0)
        self.assertIn("adaptive_policy", recommendation)
        self.assertIsNotNone(recommendation["recommended_quant"])
        assert recommendation["recommended_quant"] is not None
        self.assertIn("evaluation", recommendation["recommended_quant"])

    def test_recommend_quantization_does_not_require_writable_log_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            read_only_logs = Path(temp_dir) / "read_only_logs"
            read_only_logs.mkdir()
            os.chmod(read_only_logs, 0o555)
            config = FrameworkConfig(
                training_episodes=8,
                evaluation_episodes=4,
                recommendation_eval_episodes=4,
                recommendation_candidate_limit=4,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=str(read_only_logs),
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="recommend_read_only",
            )
            trainer = build_trainer(config, log_path=f"{temp_dir}/logs/train.jsonl")
            try:
                trainer.train()
                recommendation = recommend_quantization(trainer, config)
            finally:
                trainer.close()
                os.chmod(read_only_logs, 0o755)

            self.assertGreater(recommendation["candidate_count"], 0)

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
            self.assertIn("recommendation", summary)
            self.assertIn("benchmarks", summary)
            self.assertTrue(summary["artifacts"]["training_history"].endswith("_training_history.json"))
            self.assertTrue(summary["artifacts"]["recommendation"].endswith("_recommendation.json"))
            self.assertTrue(summary["artifacts"]["report"].endswith("_report.md"))

    def test_research_pipeline_preserves_trainer_build_error(self) -> None:
        config = FrameworkConfig(training_episodes=1, evaluation_episodes=1, run_name="build_error_test")
        with mock.patch("adaptive_quant.trainer.build_trainer", side_effect=RuntimeError("build failed")):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                ResearchPipeline(config).run()

    def test_online_pipeline_writes_summary_history_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FrameworkConfig(
                training_episodes=10,
                evaluation_episodes=4,
                stability_probe_count=1,
                outputs_dir=temp_dir,
                log_dir=f"{temp_dir}/logs",
                benchmark_dir=f"{temp_dir}/benchmarks",
                analysis_dir=f"{temp_dir}/analysis",
                checkpoint_dir=f"{temp_dir}/checkpoints",
                report_dir=f"{temp_dir}/reports",
                run_name="online_pipeline_test",
                online_learning=True,
                online_requests=8,
                online_exploration_rate=1.0,
                online_canary_ratio=1.0,
                online_replay_capacity=32,
                online_min_replay_size=2,
                online_update_interval=2,
                online_batch_size=4,
                online_drift_window=4,
                online_drift_reward_delta=999.0,
                seed=23,
            )
            summary = run_online_pipeline_entrypoint(config, request_count=6, footer_mode="none")

            self.assertIn("bootstrap_train", summary)
            self.assertIn("online", summary)
            self.assertIn("evaluation", summary)
            self.assertIn("analysis", summary)
            self.assertTrue(summary["artifacts"]["training_history"].endswith("_training_history.json"))
            self.assertTrue(summary["artifacts"]["online_detail"].endswith("_online_summary.json"))
            self.assertTrue(summary["artifacts"]["online_telemetry"].endswith("_online_telemetry.jsonl"))
            self.assertTrue(summary["artifacts"]["online_replay"].endswith("_online_replay.jsonl"))
            self.assertTrue(summary["artifacts"]["report"].endswith("_report.md"))
            self.assertTrue(Path(summary["artifacts"]["report"]).is_file())

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
