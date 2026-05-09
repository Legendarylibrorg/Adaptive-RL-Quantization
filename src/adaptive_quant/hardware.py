from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from adaptive_quant.gpu_profiles import infer_gpu_profile
from adaptive_quant.math_utils import clamp
from adaptive_quant.types import HardwareProfile, HardwareType


@dataclass(frozen=True)
class DetectedHardware:
    system: str
    machine: str
    cpu_count: int
    total_memory_gb: float | None
    accelerator_type: HardwareType
    accelerator_name: str | None = None
    accelerator_memory_gb: float | None = None
    accelerator_profile: str | None = None
    cuda_available: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "accelerator_type": self.accelerator_type.value,
            "accelerator_name": self.accelerator_name,
            "accelerator_memory_gb": self.accelerator_memory_gb,
            "accelerator_profile": self.accelerator_profile,
            "cuda_available": self.cuda_available,
        }


def default_hardware_profiles() -> dict[HardwareType, HardwareProfile]:
    return {
        HardwareType.GPU: HardwareProfile(
            hardware_type=HardwareType.GPU,
            name="cuda_gpu",
            compute_factor=1.85,
            throughput_bias=1.9,
            latency_bias=0.82,
            memory_budget_mb=24_000.0,
            preferred_bits=5.4,
            kernel_uniformity_preference=0.85,
            ngl=99,
        ),
        HardwareType.CPU: HardwareProfile(
            hardware_type=HardwareType.CPU,
            name="cpu_ngl0",
            compute_factor=0.95,
            throughput_bias=0.92,
            latency_bias=1.35,
            memory_budget_mb=7_500.0,
            preferred_bits=3.8,
            kernel_uniformity_preference=0.45,
            ngl=0,
        ),
        HardwareType.LOW_RESOURCE: HardwareProfile(
            hardware_type=HardwareType.LOW_RESOURCE,
            name="low_resource_sim",
            compute_factor=0.62,
            throughput_bias=0.58,
            latency_bias=1.72,
            memory_budget_mb=2_200.0,
            preferred_bits=2.9,
            kernel_uniformity_preference=0.25,
            ngl=0,
        ),
    }


_GPU_PROFILE_SIM_PRESETS: dict[str, dict[str, float]] = {
    "consumer_8gb": {
        "compute_factor": 1.35,
        "throughput_bias": 1.45,
        "latency_bias": 0.96,
        "preferred_bits": 4.7,
        "kernel_uniformity_preference": 0.74,
    },
    "rtx4070": {
        "compute_factor": 1.62,
        "throughput_bias": 1.72,
        "latency_bias": 0.88,
        "preferred_bits": 5.1,
        "kernel_uniformity_preference": 0.81,
    },
    "rtx4080": {
        "compute_factor": 1.82,
        "throughput_bias": 1.93,
        "latency_bias": 0.78,
        "preferred_bits": 5.4,
        "kernel_uniformity_preference": 0.86,
    },
    "rtx3090": {
        "compute_factor": 1.90,
        "throughput_bias": 2.01,
        "latency_bias": 0.74,
        "preferred_bits": 5.5,
        "kernel_uniformity_preference": 0.88,
    },
    "rtx4090": {
        "compute_factor": 2.00,
        "throughput_bias": 2.12,
        "latency_bias": 0.70,
        "preferred_bits": 5.8,
        "kernel_uniformity_preference": 0.91,
    },
    "l4": {
        "compute_factor": 1.78,
        "throughput_bias": 1.88,
        "latency_bias": 0.76,
        "preferred_bits": 5.3,
        "kernel_uniformity_preference": 0.88,
    },
    "pro_48gb": {
        "compute_factor": 2.18,
        "throughput_bias": 2.26,
        "latency_bias": 0.62,
        "preferred_bits": 6.0,
        "kernel_uniformity_preference": 0.94,
    },
    "a100_40gb": {
        "compute_factor": 2.34,
        "throughput_bias": 2.44,
        "latency_bias": 0.58,
        "preferred_bits": 6.2,
        "kernel_uniformity_preference": 0.95,
    },
    "a100_80gb": {
        "compute_factor": 2.52,
        "throughput_bias": 2.65,
        "latency_bias": 0.54,
        "preferred_bits": 6.4,
        "kernel_uniformity_preference": 0.97,
    },
    "h100": {
        "compute_factor": 2.82,
        "throughput_bias": 2.95,
        "latency_bias": 0.47,
        "preferred_bits": 6.6,
        "kernel_uniformity_preference": 0.99,
    },
}


@lru_cache(maxsize=1)
def detect_host_hardware() -> DetectedHardware:
    gpu_name, gpu_memory_gb, cuda_available = _detect_accelerator()
    accelerator_type = HardwareType.GPU if gpu_name is not None else HardwareType.CPU
    accelerator_profile = infer_gpu_profile(gpu_name, gpu_memory_gb) if gpu_name is not None else None
    return DetectedHardware(
        system=platform.system().lower(),
        machine=platform.machine().lower(),
        cpu_count=os.cpu_count() or 1,
        total_memory_gb=_detect_total_memory_gb(),
        accelerator_type=accelerator_type,
        accelerator_name=gpu_name,
        accelerator_memory_gb=gpu_memory_gb,
        accelerator_profile=accelerator_profile,
        cuda_available=cuda_available,
    )


def host_aware_hardware_profiles(
    detected: DetectedHardware | None = None,
) -> dict[HardwareType, HardwareProfile]:
    profiles = default_hardware_profiles()
    if detected is None:
        return profiles

    cpu_count = max(1, int(detected.cpu_count))
    total_memory_gb = detected.total_memory_gb or 8.0
    cpu_scale = min(cpu_count, 64) / 64.0
    arm_bonus = 0.06 if detected.machine in {"arm64", "aarch64"} else 0.0

    cpu_compute = clamp(0.78 + 0.62 * cpu_scale + arm_bonus, 0.78, 1.46)
    cpu_throughput = clamp(0.82 + 0.72 * cpu_scale + arm_bonus, 0.82, 1.56)
    cpu_latency = clamp(1.46 - 0.58 * cpu_scale - arm_bonus * 0.5, 0.82, 1.46)
    cpu_memory_budget_mb = clamp(total_memory_gb * 1024.0 * 0.68, 2_048.0, 96_000.0)
    cpu_preferred_bits = clamp(3.4 + min(total_memory_gb, 128.0) / 128.0 * 1.35, 3.4, 4.85)

    profiles[HardwareType.CPU] = HardwareProfile(
        hardware_type=HardwareType.CPU,
        name=f"{detected.system}_{detected.machine}_cpu",
        compute_factor=cpu_compute,
        throughput_bias=cpu_throughput,
        latency_bias=cpu_latency,
        memory_budget_mb=cpu_memory_budget_mb,
        preferred_bits=cpu_preferred_bits,
        kernel_uniformity_preference=clamp(0.42 + cpu_scale * 0.16 + arm_bonus * 0.4, 0.42, 0.72),
        ngl=0,
    )

    profiles[HardwareType.LOW_RESOURCE] = HardwareProfile(
        hardware_type=HardwareType.LOW_RESOURCE,
        name=f"{detected.system}_{detected.machine}_low_resource",
        compute_factor=clamp(cpu_compute * 0.58, 0.42, 0.92),
        throughput_bias=clamp(cpu_throughput * 0.52, 0.42, 0.95),
        latency_bias=clamp(cpu_latency * 1.24, 1.05, 1.95),
        memory_budget_mb=clamp(total_memory_gb * 1024.0 * 0.18, 1_200.0, 6_000.0),
        preferred_bits=clamp(cpu_preferred_bits - 0.95, 2.6, 3.7),
        kernel_uniformity_preference=0.24,
        ngl=0,
    )

    if detected.accelerator_type == HardwareType.GPU and detected.accelerator_profile is not None:
        template = _GPU_PROFILE_SIM_PRESETS.get(detected.accelerator_profile, _GPU_PROFILE_SIM_PRESETS["consumer_8gb"])
        gpu_memory_gb = detected.accelerator_memory_gb or 8.0
        profiles[HardwareType.GPU] = HardwareProfile(
            hardware_type=HardwareType.GPU,
            name=detected.accelerator_profile,
            compute_factor=template["compute_factor"],
            throughput_bias=template["throughput_bias"],
            latency_bias=template["latency_bias"],
            memory_budget_mb=clamp(gpu_memory_gb * 1024.0, 6_000.0, 128_000.0),
            preferred_bits=template["preferred_bits"],
            kernel_uniformity_preference=template["kernel_uniformity_preference"],
            ngl=99,
        )

    return profiles


def resolve_target_hardware(
    available: list[HardwareType],
    detected: DetectedHardware | None = None,
) -> HardwareType:
    if detected is not None and detected.accelerator_type in available:
        return detected.accelerator_type
    if HardwareType.CPU in available:
        return HardwareType.CPU
    if HardwareType.LOW_RESOURCE in available:
        return HardwareType.LOW_RESOURCE
    if available:
        return available[0]
    return HardwareType.CPU


def _detect_accelerator() -> tuple[str | None, float | None, bool]:
    from_torch = _detect_accelerator_from_torch()
    if from_torch[0] is not None:
        return from_torch
    return _detect_accelerator_from_nvidia_smi()


def _detect_accelerator_from_torch() -> tuple[str | None, float | None, bool]:
    try:
        import torch
    except Exception:
        return None, None, False
    try:
        if not torch.cuda.is_available():
            return None, None, False
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        memory_gb = round(float(props.total_memory) / float(1024 ** 3), 2)
        return str(props.name), memory_gb, True
    except Exception:
        return None, None, False


def _detect_accelerator_from_nvidia_smi() -> tuple[str | None, float | None, bool]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, False
    if completed.returncode != 0:
        return None, None, False
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None, None, False
    name, _, memory_text = first_line.partition(",")
    try:
        memory_gb = round(float(memory_text.strip()) / 1024.0, 2) if memory_text.strip() else None
    except ValueError:
        memory_gb = None
    return name.strip() or None, memory_gb, True


def _detect_total_memory_gb() -> float | None:
    memory_bytes = _detect_total_memory_bytes()
    if memory_bytes is None or memory_bytes <= 0:
        return None
    return round(float(memory_bytes) / float(1024 ** 3), 2)


def _detect_total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        for pages_key, page_size_key in (
            ("SC_PHYS_PAGES", "SC_PAGE_SIZE"),
            ("SC_AVPHYS_PAGES", "SC_PAGE_SIZE"),
        ):
            try:
                pages = os.sysconf(pages_key)
                page_size = os.sysconf(page_size_key)
            except (OSError, ValueError):
                continue
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return pages * page_size
    if platform.system().lower() == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode == 0:
            try:
                return int(completed.stdout.strip())
            except ValueError:
                return None
    if platform.system().lower() == "windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    return None
