from __future__ import annotations

import unittest
from unittest import mock

from adaptive_quant.nvidia_secure_boundary import (
    _ACK_HOST_VENV_ENV,
    _ACK_SECURE_VM_ENV,
    _SKIP_BOUNDARY_ENV,
    approved_nvidia_boundary,
    enforce_nvidia_secure_boundary,
    is_linux_nvidia_host,
    nvidia_boundary_report,
)


class NvidiaSecureBoundaryTests(unittest.TestCase):
    def test_non_nvidia_host_is_noop(self) -> None:
        with mock.patch(
            "adaptive_quant.nvidia_secure_boundary.is_linux_nvidia_host", return_value=False
        ):
            report = enforce_nvidia_secure_boundary(context="test")
        self.assertFalse(report["linux_nvidia_host"])

    def test_ci_skips_enforcement_on_nvidia_host(self) -> None:
        with (
            mock.patch(
                "adaptive_quant.nvidia_secure_boundary.is_linux_nvidia_host", return_value=True
            ),
            mock.patch("adaptive_quant.nvidia_secure_boundary.in_ci", return_value=True),
        ):
            report = enforce_nvidia_secure_boundary(context="test")
        self.assertTrue(report["in_ci"])

    def test_requires_ack_on_bare_nvidia_linux_host(self) -> None:
        with (
            mock.patch(
                "adaptive_quant.nvidia_secure_boundary.is_linux_nvidia_host", return_value=True
            ),
            mock.patch("adaptive_quant.nvidia_secure_boundary.in_ci", return_value=False),
            mock.patch("adaptive_quant.nvidia_secure_boundary.in_container", return_value=False),
            mock.patch("adaptive_quant.nvidia_secure_boundary.detect_wsl2", return_value=False),
        ):
            with self.assertRaises(SystemExit) as ctx:
                enforce_nvidia_secure_boundary(context="startup")
        self.assertIn("NVIDIA secure boundary required", str(ctx.exception))

    def test_host_venv_ack_allows_startup(self) -> None:
        with (
            mock.patch(
                "adaptive_quant.nvidia_secure_boundary.is_linux_nvidia_host", return_value=True
            ),
            mock.patch("adaptive_quant.nvidia_secure_boundary.in_ci", return_value=False),
            mock.patch.dict("os.environ", {_ACK_HOST_VENV_ENV: "1"}, clear=False),
        ):
            report = enforce_nvidia_secure_boundary(context="startup")
        self.assertEqual(report["approved_tier"], "host_venv")

    def test_secure_vm_ack_is_preferred_tier(self) -> None:
        with mock.patch.dict("os.environ", {_ACK_SECURE_VM_ENV: "1"}, clear=False):
            approved = approved_nvidia_boundary()
        self.assertIsNotNone(approved)
        assert approved is not None
        self.assertEqual(approved[0], "disposable_vm")

    def test_skip_boundary_aborts_when_configured(self) -> None:
        with (
            mock.patch(
                "adaptive_quant.nvidia_secure_boundary.is_linux_nvidia_host", return_value=True
            ),
            mock.patch("adaptive_quant.nvidia_secure_boundary.in_ci", return_value=False),
            mock.patch.dict(
                "os.environ",
                {_SKIP_BOUNDARY_ENV: "1", "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS": "1"},
                clear=False,
            ),
        ):
            with self.assertRaises(SystemExit) as ctx:
                enforce_nvidia_secure_boundary(context="startup")
        self.assertIn("skipped", str(ctx.exception).lower())

    def test_report_contains_linux_nvidia_flag(self) -> None:
        with (
            mock.patch("adaptive_quant.hardware.nvidia_smi_visible", return_value=False),
            mock.patch("platform.system", return_value="Linux"),
        ):
            self.assertFalse(is_linux_nvidia_host())
            report = nvidia_boundary_report()
            self.assertIn("linux_nvidia_host", report)


if __name__ == "__main__":
    unittest.main()
