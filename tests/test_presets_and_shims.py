from __future__ import annotations

import importlib
import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.presets import (
    CONFIG,
    CONFIG_3090,
    CONFIG_4090,
    CONFIG_4090_UNIVERSAL,
    CONFIG_GPU,
    CONFIG_MOE,
    CONFIG_ONLINE,
    make_rtx_torch_preset,
)


class PresetModuleTests(unittest.TestCase):
    def test_baseline_preset_is_framework_config(self) -> None:
        self.assertIsInstance(CONFIG, FrameworkConfig)
        self.assertEqual(CONFIG.training_backend, "python")

    def test_gpu_and_moe_presets_distinct(self) -> None:
        self.assertEqual(CONFIG_GPU.training_backend, "pytorch")
        self.assertTrue(CONFIG_MOE.moe_enabled)

    def test_rtx_presets_label_hosts(self) -> None:
        self.assertEqual(CONFIG_3090.training_host_label, "rtx3090")
        self.assertEqual(CONFIG_4090.training_host_label, "rtx4090")
        self.assertTrue(CONFIG_4090_UNIVERSAL.multi_hardware)

    def test_online_preset_enables_online_fields(self) -> None:
        self.assertGreater(CONFIG_ONLINE.online_reward_guard, 0.0)

    def test_make_rtx_torch_preset_builds_pytorch_config(self) -> None:
        cfg = make_rtx_torch_preset(
            training_host_label="rtx4090",
            benchmark_training_episodes=64,
            benchmark_evaluation_episodes=16,
            run_name="custom_torch",
            torch_gpu_profile="rtx4090",
            torch_hidden_dim=256,
            torch_mlp_depth=2,
            torch_batch_episodes=32,
            torch_minibatch_size=16,
            torch_update_epochs=2,
            torch_entropy_coef=0.01,
            torch_preflight_batch_size=8,
            torch_preflight_min_free_memory_gb=4.0,
        )
        self.assertEqual(cfg.run_name, "custom_torch")
        self.assertEqual(cfg.training_backend, "pytorch")
        self.assertEqual(cfg.torch_gpu_profile, "rtx4090")


class ConfigModuleTests(unittest.TestCase):
    _EXPORTS = (
        "CONFIG",
        "CONFIG_3090",
        "CONFIG_4090",
        "CONFIG_4090_UNIVERSAL",
        "CONFIG_GPU",
        "CONFIG_MOE",
        "CONFIG_ONLINE",
    )

    def test_config_module_exports_framework_configs(self) -> None:
        mod = importlib.import_module("config")
        for export_name in self._EXPORTS:
            with self.subTest(export=export_name):
                config = getattr(mod, export_name)
                self.assertIsInstance(config, FrameworkConfig)


if __name__ == "__main__":
    unittest.main()
