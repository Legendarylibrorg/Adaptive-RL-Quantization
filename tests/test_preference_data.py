"""Tests for preference dataset loading."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from adaptive_quant.llm_alignment.preference_data import load_preference_dataset


class PreferenceDataTests(unittest.TestCase):
    def test_load_json_list(self) -> None:
        with self.subTest("json list"):
            path = Path(self._tmpdir) / "prefs.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "prompt": "p",
                            "chosen": "c",
                            "rejected": "r",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_preference_dataset(path)
            self.assertEqual(rows[0]["chosen"], "c")

    def test_load_jsonl(self) -> None:
        path = Path(self._tmpdir) / "prefs.jsonl"
        path.write_text(
            '{"prompt":"p","chosen":"c","rejected":"r"}\n',
            encoding="utf-8",
        )
        rows = load_preference_dataset(path)
        self.assertEqual(len(rows), 1)

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
