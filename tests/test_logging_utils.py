from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.logging_utils import (
    JsonlLogger,
    MAX_JSONL_LINE_BYTES,
    load_jsonl,
    read_json,
    write_json,
    write_text_file,
)


class LoggingUtilsTests(unittest.TestCase):
    def test_jsonl_logger_is_lazy_and_appends_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "events.jsonl"
            logger = JsonlLogger(str(path))
            self.assertFalse(path.exists())

            logger.log({"event": "first", "value": 1})
            logger.close()
            logger.log({"event": "second", "value": 2})

            self.assertEqual(
                load_jsonl(str(path)),
                [
                    {"event": "first", "value": 1},
                    {"event": "second", "value": 2},
                ],
            )

    def test_write_json_creates_parent_dirs_and_serializes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "artifact.json"
            write_json(path, {"answer": 42, "items": [1, 2, 3]})

            payload = read_json(path, label="write_json output (test)")
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

    def test_read_json_rejects_excessive_nesting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.json"
            # Depth 65: root dict + 64 nested dicts under key "k" -> innermost at depth 64 triggers limit.
            inner: dict[str, object] = {"x": 1}
            for _ in range(63):
                inner = {"k": inner}
            payload = {"k": inner}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_json(path, label="nested bomb")
            self.assertIn("nesting depth", str(ctx.exception))

    def test_load_jsonl_rejects_wide_container_graph(self) -> None:
        import adaptive_quant.logging_utils as logging_utils

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.jsonl"
            # Many sibling dicts in one array: each dict is a container node.
            inner = [{"i": j} for j in range(12)]
            line_obj = {"rows": inner}
            path.write_text(json.dumps(line_obj) + "\n", encoding="utf-8")
            orig = logging_utils.MAX_JSON_CONTAINER_NODES
            try:
                logging_utils.MAX_JSON_CONTAINER_NODES = 8
                with self.assertRaises(ValueError) as ctx:
                    load_jsonl(str(path))
                self.assertIn("container count", str(ctx.exception))
            finally:
                logging_utils.MAX_JSON_CONTAINER_NODES = orig

    def test_jsonl_logger_rejects_poisoned_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "events.jsonl"
            logger = JsonlLogger(str(path))
            inner: dict[str, object] = {"x": 1}
            for _ in range(65):
                inner = {"k": inner}
            with self.assertRaises(ValueError):
                logger.log({"event": "bad", "nested": inner})

    def test_jsonl_logger_rejects_line_serialization_over_limit(self) -> None:
        # Structural limits allow the dict, but UTF-8 line length must stay bounded.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "big.jsonl"
            logger = JsonlLogger(str(path))
            # JSON wrapper is shorter than the string cap; nudge past the line budget.
            big = "y" * (MAX_JSONL_LINE_BYTES - 8)
            with self.assertRaises(ValueError) as ctx:
                logger.log({"blob": big})
            self.assertIn("serializes", str(ctx.exception))

    def test_enforce_safe_parsed_json_rejects_non_finite_float(self) -> None:
        from adaptive_quant.logging_utils import enforce_safe_parsed_json

        with self.assertRaises(ValueError) as ctx:
            enforce_safe_parsed_json({"x": float("nan")}, label="nan probe")
        self.assertIn("non-finite float", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
