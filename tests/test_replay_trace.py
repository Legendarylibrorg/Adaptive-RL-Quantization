from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import JsonlLogger, load_jsonl
from adaptive_quant.policy import UniversalQuantizationPolicy
from adaptive_quant.replay_trace import (
    build_manifest_steps,
    chain_step_hash,
    config_fingerprint,
    finalize_replay_artifacts,
    load_replay_manifest,
    replay_from_manifest_file,
    step_fingerprint,
    verify_jsonl_against_manifest,
)
from adaptive_quant.trainer_utils import feedback_vector, zero_previous_action


class ReplayTraceTests(unittest.TestCase):
    def _repro_config(self, run_name: str) -> FrameworkConfig:
        return FrameworkConfig.reproducible_research(
            run_name=run_name,
            training_episodes=3,
            evaluation_episodes=2,
            stability_probe_count=1,
            write_research_report=False,
        )

    def test_reproducible_preset_enables_hash_replay_stack(self) -> None:
        cfg = self._repro_config("preset_flags")
        self.assertTrue(cfg.jsonl_integrity_chain)
        self.assertTrue(cfg.replay_manifest_enabled)
        self.assertTrue(cfg.replay_verify_after_run)
        self.assertFalse(cfg.detect_host_hardware)

    def test_manifest_round_trip_jsonl_verify_and_simulator_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/trace.jsonl"
            cfg = self._repro_config("round_trip")
            env = AdaptiveQuantizationEnv(cfg, log_path=log_path)
            policy = UniversalQuantizationPolicy(cfg)
            previous = zero_previous_action()
            try:
                for episode in range(3):
                    state = env.reset(
                        previous_action=previous,
                        phase="train",
                        episode_index=episode,
                    )
                    decision, trace = policy.act(state, deterministic=True)
                    result = env.evaluate_current(decision, episode_index=episode)
                    policy.update(trace, result.metrics.reward)
                    previous = feedback_vector(
                        result.decision,
                        max_bits=max(cfg.discrete_bit_widths),
                        scale_upper=cfg.scale_bounds[1],
                        clip_upper=cfg.clip_bounds[1],
                    )
            finally:
                env.logger.close()

            report = finalize_replay_artifacts(cfg, log_path)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertTrue(report["jsonl_verify"]["verified"])
            self.assertTrue(report["replay_verify"]["verified"])
            manifest_path = report["manifest_path"]
            self.assertTrue(Path(manifest_path).is_file())

            manifest = load_replay_manifest(manifest_path)
            self.assertEqual(manifest["config_sha256"], config_fingerprint(cfg))
            records = load_jsonl(log_path, require_integrity_chain=True)
            self.assertEqual(len(records), 3)
            self.assertTrue(records[0].get("_integrity_hash"))

            second_pass = replay_from_manifest_file(cfg, manifest_path, verify_jsonl=log_path)
            self.assertTrue(second_pass["replay"]["verified"])
            self.assertTrue(second_pass["jsonl_verify"]["verified"])

    def test_tampered_jsonl_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "tamper.jsonl"
            cfg = self._repro_config("tamper")
            logger = JsonlLogger(str(log_path), integrity_chain=True)
            logger.log({"episode": 0, "phase": "train", "prompt_id": "p0", "metrics": {"reward": 1.0}})
            logger.close()
            manifest_path = finalize_replay_artifacts(cfg, log_path)
            assert manifest_path is not None
            lines = log_path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace('"reward": 1.0', '"reward": 9.0')
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_jsonl_against_manifest(log_path, manifest_path["manifest_path"], config=cfg)

    def test_chain_head_matches_step_hashes(self) -> None:
        record = {
            "episode": 1,
            "phase": "eval",
            "prompt_id": "alpha",
            "hardware_mode": "gpu",
            "decision": {"mode": "discrete", "base_bit_width": 4},
            "metrics": {"reward": 0.5, "latency_ms": 1.0, "throughput_tps": 2.0, "perplexity": 3.0, "memory_mb": 4.0, "stability_penalty": 0.0},
        }
        steps = build_manifest_steps([record])
        self.assertEqual(len(steps), 1)
        step_hash = step_fingerprint(record)
        self.assertEqual(steps[0]["step_sha256"], step_hash)
        self.assertEqual(steps[0]["chain_sha256"], chain_step_hash("", step_hash))


if __name__ == "__main__":
    unittest.main()
