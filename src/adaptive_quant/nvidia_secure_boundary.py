"""Secure-boundary policy for Linux hosts with an NVIDIA GPU.

See ``docs/SECURE_RUN.md``. Startup and GPU entrypoints call :func:`enforce_nvidia_secure_boundary`
before installing CUDA PyTorch or running GPU training on a bare host venv.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_ACK_HOST_VENV_ENV = "ADAPTIVE_RL_NVIDIA_HOST_VENV_ACK"
_ACK_SECURE_VM_ENV = "ADAPTIVE_RL_NVIDIA_SECURE_VM"
_ACK_WSL_ENV = "ADAPTIVE_RL_NVIDIA_WSL_ACK"
_SKIP_BOUNDARY_ENV = "ADAPTIVE_RL_SKIP_NVIDIA_BOUNDARY"
_ABORT_ENV = "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS"

_BOUNDARY_DOC = "docs/SECURE_RUN.md"


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_linux_nvidia_host() -> bool:
    """Return True when this Linux host reports at least one GPU via ``nvidia-smi``."""
    if platform.system().lower() != "linux":
        return False
    from adaptive_quant.hardware import nvidia_smi_visible

    return nvidia_smi_visible()


def in_container() -> bool:
    return Path("/.dockerenv").is_file()


def in_ci() -> bool:
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return True
    return _env_enabled("CI")


def detect_wsl2() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version


def approved_nvidia_boundary() -> tuple[str, str] | None:
    """Return ``(tier_label, detail)`` when an approved isolation tier is active."""
    if in_container():
        return ("docker_container", "hardened container (see docker-compose.yml)")
    if _env_enabled(_ACK_SECURE_VM_ENV):
        return ("disposable_vm", _ACK_SECURE_VM_ENV)
    if detect_wsl2() and _env_enabled(_ACK_WSL_ENV):
        return ("wsl2", _ACK_WSL_ENV)
    if _env_enabled(_ACK_HOST_VENV_ENV):
        return ("host_venv", _ACK_HOST_VENV_ENV)
    return None


def nvidia_boundary_report() -> dict[str, object]:
    """JSON-friendly snapshot of the active NVIDIA boundary state."""
    approved = approved_nvidia_boundary()
    return {
        "linux_nvidia_host": is_linux_nvidia_host(),
        "wsl2": detect_wsl2(),
        "in_container": in_container(),
        "in_ci": in_ci(),
        "approved_tier": approved[0] if approved else None,
        "approved_via": approved[1] if approved else None,
        "skip_boundary": _env_enabled(_SKIP_BOUNDARY_ENV),
    }


def _boundary_failure_message(*, context: str) -> str:
    return (
        f"NVIDIA secure boundary required before {context}.\n"
        "An NVIDIA GPU was detected on Linux. Host venv CUDA training (Tier 4 in "
        f"{_BOUNDARY_DOC}) increases driver and native-binary attack surface.\n\n"
        "Approve exactly one isolation tier, then retry:\n"
        f"  export {_ACK_SECURE_VM_ENV}=1     # Tier 1-2 disposable VM / lab host\n"
        f"  export {_ACK_WSL_ENV}=1           # Tier 3 WSL2 (Windows host still in play)\n"
        f"  export {_ACK_HOST_VENV_ENV}=1     # Tier 4 host .venv (trusted artifacts only)\n\n"
        "Prefer VM + hardened Docker for untrusted models:\n"
        "  bash scripts/docker_secure_preflight.sh --gpu\n"
        f"See {_BOUNDARY_DOC} for the full threat model."
    )


def enforce_nvidia_secure_boundary(*, context: str = "startup") -> dict[str, object]:
    """
    Gate startup and GPU entrypoints on NVIDIA Linux hosts.

    CI and non-NVIDIA hosts are no-ops. When the boundary is skipped via
    ``ADAPTIVE_RL_SKIP_NVIDIA_BOUNDARY``, emit a warning and honor
    ``ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS``.
    """
    report = nvidia_boundary_report()
    if not report["linux_nvidia_host"] or report["in_ci"]:
        return report

    if _env_enabled(_SKIP_BOUNDARY_ENV):
        message = f"NVIDIA secure boundary skipped during {context} ({_SKIP_BOUNDARY_ENV}=1)."
        if _env_enabled(_ABORT_ENV):
            raise SystemExit(message)
        print(message, file=sys.stderr)
        report["boundary_enforced"] = False
        report["boundary_skipped"] = True
        return report

    approved = approved_nvidia_boundary()
    if approved is not None:
        tier, via = approved
        print(
            f"nvidia_secure_boundary: ok ({tier} via {via}) — context={context}",
            file=sys.stderr,
        )
        report["boundary_enforced"] = True
        report["approved_tier"] = tier
        report["approved_via"] = via
        return report

    raise SystemExit(_boundary_failure_message(context=context))


__all__ = [
    "approved_nvidia_boundary",
    "detect_wsl2",
    "enforce_nvidia_secure_boundary",
    "in_ci",
    "in_container",
    "is_linux_nvidia_host",
    "nvidia_boundary_report",
]
