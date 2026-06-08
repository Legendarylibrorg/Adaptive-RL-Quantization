from __future__ import annotations

# PyTorch 2.12+ binary matrix (see https://pytorch.org/get-started/locally/):
# - cu130: stable default for current NVIDIA drivers
# - cu126: legacy fallback for older drivers / architectures
# - cu128: removed in PyTorch 2.12 — do not reference in install docs
TORCH_CUDA_INDEX_CU130 = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_INDEX_CU126 = "https://download.pytorch.org/whl/cu126"
DEFAULT_CUDA_INDEX = TORCH_CUDA_INDEX_CU130


def cuda_torch_pip_command(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Return a pip command that installs a CUDA-enabled torch wheel."""
    return f"python3 -m pip install --upgrade torch --index-url {index_url}"


def cuda_torch_install_instructions(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Multi-line guidance for fixing a CPU-only or mismatched PyTorch install."""
    legacy = TORCH_CUDA_INDEX_CU126
    return (
        "Install a CUDA-enabled PyTorch wheel before GPU training:\n"
        f"  {cuda_torch_pip_command(index_url=index_url)}\n"
        f"  python3 -m pip install -e .\n"
        "If that wheel does not match your driver, try the legacy CUDA 12.6 build:\n"
        f"  {cuda_torch_pip_command(index_url=legacy)}\n"
        "Verify with:\n"
        "  python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)\""
    )


def torch_cuda_ready_report() -> dict[str, object]:
    """Small JSON-friendly summary of the active torch/CUDA install."""
    try:
        import torch
    except Exception as exc:
        return {
            "torch_installed": False,
            "torch_import_error": repr(exc),
            "cuda_available": False,
            "install_hint": cuda_torch_pip_command(),
        }

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    report: dict[str, object] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        index = torch.cuda.current_device()
        report["device_name"] = str(torch.cuda.get_device_name(index))
        props = torch.cuda.get_device_properties(index)
        report["device_capability"] = f"{props.major}.{props.minor}"
        if hasattr(torch.cuda, "get_arch_list"):
            try:
                report["arch_list"] = list(torch.cuda.get_arch_list())
            except Exception:
                report["arch_list"] = []
    elif cuda_version is None:
        report["likely_cpu_only_wheel"] = True
        report["install_hint"] = cuda_torch_pip_command()
    return report
