from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.logging_utils import write_json, write_text_file


class LoggingUtilsTests(unittest.TestCase):
    def test_write_json_creates_parent_dirs_and_serializes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "artifact.json"
            write_json(path, {"answer": 42, "items": [1, 2, 3]})

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["answer"], 42)
            self.assertEqual(payload["items"], [1, 2, 3])

    def test_write_json_replace_failure_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "artifact.json"
            path.write_text('{"before": true}', encoding="utf-8")
            before_entries = sorted(p.name for p in root.iterdir())

            with mock.patch("adaptive_quant.logging_utils.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    write_json(path, {"after": True})

            after_entries = sorted(p.name for p in root.iterdir())
            self.assertEqual(before_entries, after_entries)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"before": true}')

    def test_write_text_file_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports" / "summary.md"
            write_text_file(path, "# Summary\n\nAll good.\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "# Summary\n\nAll good.\n")


if __name__ == "__main__":
    unittest.main()
