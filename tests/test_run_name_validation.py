from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig


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


if __name__ == "__main__":
    unittest.main()

