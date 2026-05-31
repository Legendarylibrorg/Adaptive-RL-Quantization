from __future__ import annotations

import unittest
from unittest import mock

from adaptive_quant.cli import calibrate_llama_cpp
from adaptive_quant.configuration import FrameworkConfig


class CalibrateCliTests(unittest.TestCase):
    def test_main_exits_when_llama_cpp_paths_are_missing(self) -> None:
        config = FrameworkConfig(run_name="calibrate_cli_test")
        with (
            mock.patch.object(calibrate_llama_cpp, "load_config_or_fallback", return_value=config),
            mock.patch.object(
                calibrate_llama_cpp,
                "require_llama_cpp_paths",
                side_effect=FileNotFoundError("missing llama.cpp"),
            ),
        ):
            with self.assertRaises(SystemExit) as ctx:
                calibrate_llama_cpp.main(["--prompts", "0"])

        self.assertIn("missing llama.cpp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
