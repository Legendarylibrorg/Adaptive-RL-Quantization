#!/usr/bin/env python3
"""Verify NVIDIA device nodes are visible inside a GPU Compose container."""

from __future__ import annotations

import glob
import os
import sys


def main() -> int:
    devices = sorted(glob.glob("/dev/nvidia*"))
    require = os.environ.get("ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if devices:
        preview = ", ".join(devices[:4])
        if len(devices) > 4:
            preview += ", ..."
        print(f"docker_gpu_device_probe: ok — {len(devices)} device node(s): {preview}")
        return 0

    message = (
        "no /dev/nvidia* in container "
        "(GPU not passed through, or compose `gpus` not applied to `docker compose run`)"
    )
    if require:
        print(f"docker_gpu_device_probe: {message}", file=sys.stderr)
        return 1
    print(f"docker_gpu_device_probe: warning — {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
