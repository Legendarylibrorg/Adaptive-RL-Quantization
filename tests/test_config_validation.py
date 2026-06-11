"""Config validation tests for quant_mode and hardware_modes."""

from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import validate_hardware_modes, validate_quant_mode


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_quant_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_quant_mode("not_a_mode")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_quant", quant_mode="invalid", detect_host_hardware=False)

    def test_empty_hardware_modes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_hardware_modes(())
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="bad_hw", hardware_modes=(), detect_host_hardware=False)

    def test_invalid_hardware_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_hardware_modes(("tpu",))
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="bad_hw",
                hardware_modes=("gpu", "tpu"),
                detect_host_hardware=False,
            )

    def test_duplicate_hardware_modes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_hardware_modes(("gpu", "gpu"))


if __name__ == "__main__":
    unittest.main()
