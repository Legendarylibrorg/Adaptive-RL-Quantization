"""Hardware-aware unittest selection for post-install setup validation.

``./setup.sh`` and GPU entrypoints run this curated subset instead of the full
``unittest discover`` suite. The E2E smoke (``config.e2e_smoke.json``) remains
the integration check; these modules only verify install, config, CLI, and
optional torch/NVIDIA wiring for the current host.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

# Core: presets, config coercion, guardrails, feature math — no GPU required.
_BASE_MODULES: tuple[str, ...] = (
    "tests.test_presets_and_shims",
    "tests.test_easy_config",
    "tests.test_guardrails",
    "tests.test_features",
)

# CLI wiring and entrypoint scripts.
_CLI_MODULES: tuple[str, ...] = (
    "tests.test_cli_behavior",
)

# When PyTorch is importable (CPU or CUDA wheel).
_TORCH_MODULES: tuple[str, ...] = (
    "tests.test_torch_trainer",
)

# Linux hosts where ``nvidia-smi`` reports a GPU.
_NVIDIA_MODULES: tuple[str, ...] = (
    "tests.test_nvidia_secure_boundary",
    "tests.test_install_cuda_torch",
)


def torch_importable() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_setup_test_modules(
    *,
    include_torch: bool | None = None,
    include_nvidia: bool | None = None,
) -> list[str]:
    """Return unittest module names appropriate for this host."""
    modules = list(_BASE_MODULES) + list(_CLI_MODULES)

    if include_torch is None:
        include_torch = torch_importable()
    if include_nvidia is None:
        from adaptive_quant.nvidia_secure_boundary import is_linux_nvidia_host

        include_nvidia = is_linux_nvidia_host()

    if include_torch:
        modules.extend(_TORCH_MODULES)
    if include_nvidia:
        modules.extend(_NVIDIA_MODULES)

    return modules


def unittest_command(
    python_bin: str,
    modules: Sequence[str] | None = None,
    *,
    quiet: bool = True,
) -> list[str]:
    """Build a subprocess argv list for the setup test subset."""
    selected = list(modules) if modules is not None else resolve_setup_test_modules()
    cmd = [python_bin, "-m", "unittest", *selected]
    if quiet:
        cmd.append("-q")
    return cmd
