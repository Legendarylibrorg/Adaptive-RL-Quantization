"""Tests for optional GGUF export pipeline step."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.gguf_export import (
    derive_quantize_binary,
    export_gguf,
    maybe_export_gguf,
    resolve_gguf_quant_type,
)


class GgufExportTests(unittest.TestCase):
    def test_maybe_export_skipped_when_disabled(self) -> None:
        cfg = FrameworkConfig(run_name="no_export", detect_host_hardware=False)
        result = maybe_export_gguf(cfg, None)
        self.assertFalse(result["enabled"])
        self.assertTrue(result["skipped"])

    def test_resolve_quant_type_from_recommendation_base_bits(self) -> None:
        cfg = FrameworkConfig(
            run_name="quant_map",
            llama_cpp_gguf_export_quant_type="Q4_K_M",
            detect_host_hardware=False,
        )
        recommendation = {
            "decision": {"deploy": "discrete|base=8|bits=-"},
            "recommended_quant": {
                "signature": "discrete|base=8|bits=8,8,8|scale=1.00|clip=1.00|precision=0.00|moe=-"
            },
        }
        self.assertEqual(resolve_gguf_quant_type(cfg, recommendation), "Q8_0")

    def test_derive_quantize_binary_from_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "llama-cli"
            quantize = root / "llama-quantize"
            cli.write_text("", encoding="utf-8")
            quantize.write_text("", encoding="utf-8")
            cfg = FrameworkConfig(
                run_name="derive_bin",
                llama_cpp_binary=str(cli),
                detect_host_hardware=False,
            )
            self.assertEqual(
                Path(derive_quantize_binary(cfg)).resolve(),
                quantize.resolve(),
            )

    def test_export_gguf_invokes_quantize_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.gguf"
            source.write_text("source", encoding="utf-8")
            quantize = root / "llama-quantize"
            quantize.write_text("", encoding="utf-8")
            cfg = FrameworkConfig(
                run_name="export_run",
                outputs_dir=str(root / "outputs"),
                log_dir=str(root / "outputs" / "logs"),
                benchmark_dir=str(root / "outputs" / "benchmarks"),
                analysis_dir=str(root / "outputs" / "analysis"),
                checkpoint_dir=str(root / "outputs" / "checkpoints"),
                report_dir=str(root / "outputs" / "reports"),
                gguf_export_dir=str(root / "outputs" / "gguf"),
                llama_cpp_gguf_export_enabled=True,
                llama_cpp_gguf_export_source=str(source),
                llama_cpp_gguf_quantize_binary=str(quantize),
                llama_cpp_gguf_export_quant_type="Q4_K_M",
                detect_host_hardware=False,
            )
            with mock.patch("adaptive_quant.pipeline.gguf_export.subprocess.run") as run_mock:
                run_mock.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                result = export_gguf(cfg, None)
            self.assertEqual(result["quant_type"], "Q4_K_M")
            self.assertTrue(Path(str(result["output_path"])).name.endswith("_Q4_K_M.gguf"))
            argv = run_mock.call_args[0][0]
            self.assertEqual(argv[0], str(quantize))
            self.assertEqual(argv[1], str(source))

    def test_gguf_export_enabled_requires_source(self) -> None:
        with self.assertRaises(ValueError):
            FrameworkConfig(
                run_name="bad_export",
                llama_cpp_gguf_export_enabled=True,
                detect_host_hardware=False,
            )


if __name__ == "__main__":
    unittest.main()
