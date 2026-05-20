from __future__ import annotations

import unittest

from adaptive_quant.backend import parse_llama_cpp_metrics
from adaptive_quant.backends.llama_cpp import extract_numeric


class LlamaCppParsingTests(unittest.TestCase):
    def test_extract_numeric_regex_finds_last_match(self) -> None:
        text = "10.0 tok/s\nsomething 12.5 tok/s\n"
        self.assertEqual(extract_numeric(text, "tok/s", default=0.0), 12.5)

    def test_extract_numeric_supports_marker_before_number(self) -> None:
        text = "tok/s 10.0\nsomething tok/s 12.5\n"
        self.assertEqual(extract_numeric(text, "tok/s", default=0.0), 12.5)

    def test_parse_metrics_from_stdout_and_stderr_style_text(self) -> None:
        text = """
        llama_print_timings:        load time =    12.34 ms
        llama_print_timings:      sample time =     5.00 ms /   10 runs   ( 0.50 ms per token, 2000.00 tok/s)
        llama_print_timings: prompt eval time =   123.00 ms /   50 tokens ( 2.46 ms per token,  406.50 tok/s)
        """
        parsed = parse_llama_cpp_metrics(text.lower())
        self.assertAlmostEqual(parsed["throughput_tps"], 406.50, places=2)
        self.assertAlmostEqual(parsed["latency_ms_per_token"], 2.46, places=2)

    def test_parse_metrics_ignores_missing_memory(self) -> None:
        parsed = parse_llama_cpp_metrics("1.0 ms per token, 100.0 tok/s")
        self.assertIn("throughput_tps", parsed)
        self.assertIn("latency_ms_per_token", parsed)
        self.assertNotIn("memory_mb", parsed)

    def test_parse_metrics_extracts_memory_only_with_memory_labels(self) -> None:
        text = """
        something about the model 4096 mb tokenizer cache (not memory usage)
        kv cache: 2048 mb
        prompt eval time = 123.00 ms / 50 tokens ( 2.46 ms per token,  406.50 tok/s)
        """
        parsed = parse_llama_cpp_metrics(text.lower())
        self.assertAlmostEqual(parsed["memory_mb"], 2048.0, places=3)


if __name__ == "__main__":
    unittest.main()
