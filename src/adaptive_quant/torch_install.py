from __future__ import annotations

# PyTorch 2.12+ binary matrix (see https://pytorch.org/get-started/locally/):
# - cu130: stable default for current NVIDIA drivers
# - cu126: legacy fallback for older drivers / architectures
# - cu128: removed in PyTorch 2.12 — do not reference in install docs
TORCH_CUDA_INDEX_CU130 = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_INDEX_CU126 = "https://download.pytorch.org/whl/cu126"
DEFAULT_CUDA_INDEX = TORCH_CUDA_INDEX_CU130
INSTALL_CUDA_TORCH_SCRIPT = "python3 scripts/install_cuda_torch.py"


def cuda_torch_pip_command(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Return a pip command that installs a CUDA-enabled torch wheel."""
    return f"python3 -m pip install --upgrade torch --index-url {index_url}"


def cuda_torch_install_instructions(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Multi-line guidance for fixing a CPU-only or mismatched PyTorch install."""
    legacy = TORCH_CUDA_INDEX_CU126
    return (
        "Install a CUDA-enabled PyTorch wheel before GPU training:\n"
        f"  {INSTALL_CUDA_TORCH_SCRIPT}\n"
        "Or install manually:\n"
        f"  {cuda_torch_pip_command(index_url=index_url)}\n"
        f"  python3 -m pip install -e .\n"
        "If that wheel does not match your driver, try the legacy CUDA 12.6 build:\n"
        f"  {cuda_torch_pip_command(index_url=legacy)}\n"
        "Verify with:\n"
        "  python3 scripts/install_cuda_torch.py --check-only"
    )


def torch_cuda_ready_report() -> dict[str, object]:
    """Canonical JSON-friendly summary of the active torch/CUDA install."""
    from adaptive_quant.hardware import nvidia_smi_visible
    from adaptive_quant.torch_policy import torch_cuda_diagnostics

    report: dict[str, object] = dict(torch_cuda_diagnostics("cuda"))
    smi_visible = nvidia_smi_visible()
    report["nvidia_smi_visible"] = smi_visible

    if not report.get("torch_installed", False):
        report.setdefault("install_hint", INSTALL_CUDA_TORCH_SCRIPT)
        return report

    cuda_version = report.get("cuda_version")
    if not report.get("cuda_available"):
        if cuda_version is None:
            report["likely_cpu_only_wheel"] = True
        report.setdefault("install_hint", INSTALL_CUDA_TORCH_SCRIPT)
        if report.get("likely_cpu_only_wheel") and smi_visible:
            report["driver_gpu_detected"] = True
        return report

    arch_list = report.get("torch_cuda_arch_list")
    if arch_list:
        report["arch_list"] = arch_list
    return report


def validate_cuda_after_install(requested_device: str = "cuda") -> None:
    """Raise when CUDA is unavailable or the active wheel cannot run the visible GPU."""
    report = torch_cuda_ready_report()
    if not report.get("cuda_available"):
        hint = report.get("install_hint") or INSTALL_CUDA_TORCH_SCRIPT
        smi = " nvidia-smi sees a GPU but" if report.get("driver_gpu_detected") else ""
        raise RuntimeError(
            f"CUDA is not available after installing torch.{smi} "
            f"Confirm the driver with `nvidia-smi`, then retry:\n  {hint}"
        )
    from adaptive_quant.torch_policy import validate_cuda_runtime_compatibility

    validate_cuda_runtime_compatibility(requested_device)
