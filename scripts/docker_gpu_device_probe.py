#!/usr/bin/env python3
"""Verify NVIDIA device nodes are visible inside a GPU Compose container."""

from __future__ import annotations

import glob
import os
import sys


def _device_glob() -> str:
    # Test-only escape hatch so host GPU presence does not make probe tests nondeterministic.
    return os.environ.get("ADAPTIVE_RL_NVIDIA_DEVICE_GLOB", "/dev/nvidia*")


def _cuda_env_hint() -> str:
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    capabilities = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "").strip()
    parts: list[str] = []
    if visible:
        parts.append(f"NVIDIA_VISIBLE_DEVICES={visible!r}")
    if capabilities:
        parts.append(f"NVIDIA_DRIVER_CAPABILITIES={capabilities!r}")
    return "; ".join(parts)


def main() -> int:
    devices = sorted(glob.glob(_device_glob()))
    require = os.environ.get("ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if devices:
        preview = ", ".join(devices[:4])
        if len(devices) > 4:
            preview += ", ..."
        print(f"docker_gpu_device_probe: ok - {len(devices)} NVIDIA device node(s): {preview}")
        return 0

    message = (
        "no NVIDIA device nodes in container "
        "(GPU not passed through, NVIDIA Container Toolkit is missing, or compose `gpus` "
        "was not applied to `docker compose run`)"
    )
    hint = _cuda_env_hint()
    if hint:
        message = f"{message}; CUDA env: {hint}"
    if require:
        print(f"docker_gpu_device_probe: error - {message}", file=sys.stderr)
        return 1
    print(f"docker_gpu_device_probe: warning - {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
