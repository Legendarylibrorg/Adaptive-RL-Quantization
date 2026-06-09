"""Tests for end-of-run CLI footers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.run_footer import print_pipeline_footer


class RunFooterTests(unittest.TestCase):
    def test_pipeline_footer_includes_deploy_decision(self) -> None:
        cfg = FrameworkConfig(run_name="footer_test", detect_host_hardware=False)
        summary = {
            "recommendation": {
                "target_hardware": "gpu",
                "decision": {
                    "deploy": "adaptive_policy",
                    "use_adaptive_policy": True,
                    "rationale": "test",
                },
            }
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_pipeline_footer(cfg, summary)
        output = buffer.getvalue()
        self.assertIn("deploy", output)
        self.assertIn("adaptive_policy", output)
        self.assertIn("evidence_level", output)
        self.assertIn("simulator", output)


if __name__ == "__main__":
    unittest.main()
