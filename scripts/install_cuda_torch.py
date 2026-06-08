#!/usr/bin/env python3
"""Install a CUDA-enabled PyTorch wheel for GPU training on Linux + NVIDIA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adaptive_quant.hardware import nvidia_smi_visible  # noqa: E402
from adaptive_quant.torch_install import (  # noqa: E402
    DEFAULT_CUDA_INDEX,
    TORCH_CUDA_INDEX_CU126,
    TORCH_CUDA_INDEX_CU130,
    cuda_torch_pip_argv,
    cuda_torch_pip_command,
    torch_cuda_ready_report,
    validate_cuda_after_install,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install a CUDA-enabled PyTorch wheel for adaptive-rl-quant GPU entrypoints. "
            "PyTorch 2.12+ uses cu130 (default) or cu126 (legacy); cu128 wheels were removed."
        )
    )
    parser.add_argument(
        "--cuda",
        choices=("cu130", "cu126"),
        default="cu130",
        help="CUDA wheel index to use (default: cu130 for current NVIDIA drivers)",
    )
    parser.add_argument(
        "--skip-editable-install",
        action="store_true",
        help="Only install torch; do not run pip install -e . afterward",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the active torch/CUDA report and exit without installing",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Pass --force-reinstall to pip when upgrading torch",
    )
    parser.add_argument(
        "--skip-nvidia-smi-check",
        action="store_true",
        help="Skip the pre-install nvidia-smi visibility check",
    )
    return parser


def _index_for(cuda: str) -> str:
    return TORCH_CUDA_INDEX_CU130 if cuda == "cu130" else TORCH_CUDA_INDEX_CU126


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.check_only:
        report = torch_cuda_ready_report()
        for key, value in report.items():
            print(f"{key}: {value}")
        if report.get("cuda_available") and report.get("cuda_arch_supported") is False:
            return 1
        return 0 if report.get("cuda_available") else 1

    if not args.skip_nvidia_smi_check and not nvidia_smi_visible():
        print(
            "Warning: nvidia-smi did not list a GPU. Install will continue, but CUDA training "
            "requires a working NVIDIA driver on Linux.",
            file=sys.stderr,
        )

    index_url = _index_for(args.cuda)
    pip_cmd = cuda_torch_pip_argv(
        python=sys.executable,
        index_url=index_url,
        force_reinstall=args.force_reinstall,
    )
    commands = [pip_cmd]
    if not args.skip_editable_install:
        commands.append([sys.executable, "-m", "pip", "install", "-e", str(_REPO_ROOT)])

    print(f"Using CUDA index: {index_url}")
    print(f"Equivalent command: {cuda_torch_pip_command(index_url=index_url)}")
    for cmd in commands:
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=_REPO_ROOT)

    report = torch_cuda_ready_report()
    print("Post-install:", report)
    try:
        validate_cuda_after_install("cuda", report=report)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"If your driver is older, retry with --cuda cu126. Default index is {DEFAULT_CUDA_INDEX}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
