from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
            }
        )
        self.assertEqual(cfg.hardware_modes, ("gpu", "cpu"))
        self.assertEqual(cfg.discrete_bit_widths, (4, 8))
        self.assertEqual(cfg.scale_bounds, (0.5, 1.5))

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


if __name__ == "__main__":
    unittest.main()
