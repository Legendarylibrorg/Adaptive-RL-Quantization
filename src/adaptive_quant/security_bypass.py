"""Detect and surface active security bypass environment variables at runtime."""

from __future__ import annotations

import os
import sys

_BYPASS_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("ADAPTIVE_RL_HF_ALLOW_UNLISTED", "Hugging Face repo allowlist disabled"),
    ("ADAPTIVE_RL_SKIP_CHECKPOINT_INTEGRITY", "Checkpoint integrity verification disabled"),
    ("ADAPTIVE_RL_SKIP_NVIDIA_BOUNDARY", "NVIDIA secure-boundary checks disabled"),
)

_ABORT_ENV = "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS"


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def active_security_bypasses() -> list[tuple[str, str]]:
    return [(name, desc) for name, desc in _BYPASS_ENV_VARS if _env_enabled(name)]


def enforce_security_bypass_policy(*, context: str = "pipeline") -> None:
    """Emit stderr warnings for bypass env vars; abort when ``ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS=1``."""
    active = active_security_bypasses()
    if not active:
        return
    lines = [f"SECURITY BYPASS active during {context}:"]
    for env_var, description in active:
        lines.append(f"  - {env_var}=1 ({description})")
    message = "\n".join(lines)
    if _env_enabled(_ABORT_ENV):
        raise SystemExit(
            message
            + "\nUnset bypass variables or do not set ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS=1."
        )
    print(message, file=sys.stderr)


__all__ = ["active_security_bypasses", "enforce_security_bypass_policy"]
