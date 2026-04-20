from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import MAX_JSONL_LINE_BYTES, load_jsonl


class RunNameValidationTests(unittest.TestCase):
    def test_run_name_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="../pwn")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="..")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="a/../../b")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="a\\b")

    def test_run_name_rejects_weird_chars(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="has space")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="💥")
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="")

    def test_run_name_accepts_slug(self) -> None:
        cfg = FrameworkConfig(run_name="run-01.ok_test")
        self.assertEqual(cfg.run_name, "run-01.ok_test")

    def test_log_dir_rejects_parent_components(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="ok", log_dir="outputs/../secrets")

    def test_resume_checkpoint_rejects_double_dot(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(run_name="ok", resume_from_checkpoint="/tmp/../etc/passwd")


class JsonlReadLimitsTests(unittest.TestCase):
    def test_load_jsonl_rejects_huge_line(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            tmp.write('{"ok":true}\n')
            tmp.write("x" * (MAX_JSONL_LINE_BYTES + 16) + "\n")
            path = tmp.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_jsonl(path)
            self.assertIn("byte limit", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

