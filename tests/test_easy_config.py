from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.cli.calibrate_llama_cpp import _build_calibration_config
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.easy_config import (
    config_from_dict,
    load_config,
    named_preset,
    quick_config,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


class EasyConfigTests(unittest.TestCase):
    def test_e2e_smoke_json_loads(self) -> None:
        path = _REPO_ROOT / "config.e2e_smoke.json"
        self.assertTrue(path.is_file(), "repo should ship config.e2e_smoke.json for quick E2E runs")
        cfg = load_config(path)
        self.assertEqual(cfg.run_name, "e2e_smoke")
        self.assertEqual(cfg.training_episodes, 32)
        self.assertEqual(cfg.seed, 13)
        self.assertEqual(cfg.env_sampling_mode, "sequential")
        self.assertEqual(cfg.rl_train_policy_mode, "deterministic")
        self.assertEqual(cfg.stability_probe_sampling, "deterministic")

    def test_named_preset_reproducible(self) -> None:
        cfg = named_preset("reproducible")
        self.assertEqual(cfg.env_sampling_mode, "sequential")
        self.assertTrue(cfg.torch_deterministic)
        self.assertTrue(cfg.jsonl_integrity_chain)
        self.assertTrue(cfg.replay_manifest_enabled)
        self.assertTrue(cfg.replay_verify_after_run)

    def test_named_preset_unknown(self) -> None:
        with self.assertRaises(ValueError):
            named_preset("not_a_real_preset")

    def test_quick_config_partial(self) -> None:
        cfg = quick_config(run_name="quick_test", training_episodes=99)
        self.assertEqual(cfg.run_name, "quick_test")
        self.assertEqual(cfg.training_episodes, 99)
        self.assertEqual(cfg.evaluation_episodes, 400)

    def test_config_from_dict_coerces_lists(self) -> None:
        cfg = config_from_dict(
            {
                "run_name": "coerce_test",
                "hardware_modes": ["gpu", "cpu"],
                "discrete_bit_widths": [4, 8],
                "scale_bounds": [0.5, 1.5],
                "router_hf_allowed_models": ["org/model-a", "org/model-b"],
            }
        )
        self.assertEqual(cfg.hardware_modes, ("gpu", "cpu"))
        self.assertEqual(cfg.discrete_bit_widths, (4, 8))
        self.assertEqual(cfg.scale_bounds, (0.5, 1.5))
        self.assertEqual(cfg.router_hf_allowed_models, ("org/model-a", "org/model-b"))

    def test_config_from_dict_accepts_nested_sections(self) -> None:
        cfg = config_from_dict(
            {
                "run_name": "nested_sections",
                "moe": {"num_experts": 8, "top_k": 1},
                "torch": {"preflight": False},
            }
        )
        self.assertEqual(cfg.moe_num_experts, 8)
        self.assertEqual(cfg.moe_top_k, 1)
        self.assertFalse(cfg.torch_preflight)

    def test_config_from_dict_merges_reward_weights(self) -> None:
        cfg = config_from_dict(
            {"run_name": "rw_test", "reward_weights": {"gamma_perplexity": 0.5}},
        )
        self.assertEqual(cfg.reward_weights.gamma_perplexity, 0.5)
        self.assertEqual(cfg.reward_weights.alpha_latency, 0.020)

    def test_config_strict_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict({"run_name": "x", "not_a_field": 1}, strict=True)

    def test_load_json_with_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(
                json.dumps(
                    {
                        "preset": "minimal",
                        "run_name": "layered",
                        "training_episodes": 123,
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertEqual(cfg.run_name, "layered")
            self.assertEqual(cfg.training_episodes, 123)
            self.assertEqual(cfg.stability_probe_count, 1)

    def test_framework_config_from_file_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"run_name": "alias_test"}), encoding="utf-8")
            cfg = FrameworkConfig.from_file(path)
            self.assertEqual(cfg.run_name, "alias_test")

    def test_load_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(
                'run_name = "toml_run"\ntraining_episodes = 42\n',
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertEqual(cfg.run_name, "toml_run")
            self.assertEqual(cfg.training_episodes, 42)

    def test_load_config_is_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps({"run_name": "strict_default", "training_episode": 42}), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("Unknown FrameworkConfig keys", str(ctx.exception))

    def test_load_config_rejects_oversized_local_file(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("adaptive_quant.logging_utils.MAX_LOCAL_READ_BYTES", 16),
        ):
            path = Path(tmp) / "oversized.json"
            path.write_text(json.dumps({"run_name": "oversized_config"}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("Config file exceeds local read limit", str(ctx.exception))

    def test_calibration_config_applies_cli_seed(self) -> None:
        base = FrameworkConfig(run_name="calibration_seed_test", seed=13)
        cfg = _build_calibration_config(base, 1234)
        self.assertEqual(cfg.seed, 1234)
        self.assertEqual(cfg.backend, "llama_cpp")
        self.assertFalse(cfg.prompt_split_enabled)


if __name__ == "__main__":
    unittest.main()
