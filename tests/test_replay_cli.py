from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from adaptive_quant.cli import replay
from adaptive_quant.configuration import FrameworkConfig


class ReplayCliTests(unittest.TestCase):
    def test_requires_config_unless_building_manifest(self) -> None:
        with mock.patch("sys.argv", ["adaptive-rl-quant-replay"]):
            with self.assertRaises(SystemExit) as ctx:
                replay.main()
        self.assertIn("--config is required", str(ctx.exception))

    def test_verify_jsonl_only_exits_nonzero_on_failed_verification(self) -> None:
        config = FrameworkConfig.reproducible_research(run_name="replay_cli_test")
        report = {"verified": False, "mismatches": [{"kind": "hash_mismatch"}]}
        stdout = io.StringIO()
        with (
            mock.patch(
                "sys.argv",
                [
                    "adaptive-rl-quant-replay",
                    "--config",
                    "config.json",
                    "--manifest",
                    "manifest.json",
                    "--jsonl",
                    "episodes.jsonl",
                    "--verify-jsonl-only",
                ],
            ),
            mock.patch.object(replay, "load_config_or_fallback", return_value=config),
            mock.patch.object(replay, "enforce_security_bypass_policy"),
            mock.patch.object(replay, "verify_jsonl_against_manifest", return_value=report),
            contextlib.redirect_stdout(stdout),
        ):
            with self.assertRaises(SystemExit) as ctx:
                replay.main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_build_manifest_prints_report(self) -> None:
        config = FrameworkConfig.reproducible_research(run_name="replay_cli_test")
        report = {"manifest_path": "manifest.json", "step_count": 1}
        stdout = io.StringIO()
        with (
            mock.patch(
                "sys.argv",
                [
                    "adaptive-rl-quant-replay",
                    "--build-manifest",
                    "--jsonl",
                    "episodes.jsonl",
                ],
            ),
            mock.patch.object(replay, "load_config_or_fallback", return_value=config),
            mock.patch.object(replay, "enforce_security_bypass_policy"),
            mock.patch.object(replay, "finalize_replay_artifacts", return_value=report),
            contextlib.redirect_stdout(stdout),
        ):
            replay.main()

        self.assertEqual(json.loads(stdout.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
