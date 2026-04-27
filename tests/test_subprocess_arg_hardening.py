from __future__ import annotations

import unittest


class SubprocessArgHardeningTests(unittest.TestCase):
    def test_common_run_rejects_non_list(self) -> None:
        from scripts import _common

        with self.assertRaises(TypeError):
            _common.run("echo hi")  # type: ignore[arg-type]

    def test_common_run_rejects_empty_list(self) -> None:
        from scripts import _common

        with self.assertRaises(TypeError):
            _common.run([])  # type: ignore[arg-type]

    def test_common_run_rejects_non_string_items(self) -> None:
        from scripts import _common

        with self.assertRaises(TypeError):
            _common.run(["echo", 1])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()

