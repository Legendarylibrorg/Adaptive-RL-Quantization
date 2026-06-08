from __future__ import annotations

import re
from dataclasses import dataclass

from adaptive_quant.configuration import FrameworkConfig


@dataclass(frozen=True)
class GpuProfile:
    name: str
    description: str
    overrides: dict[str, object]


GPU_PROFILES: dict[str, GpuProfile] = {
    "rtx4090": GpuProfile(
        name="rtx4090",
        description="High-throughput 24 GB Ada Lovelace profile.",
        overrides={
            "torch_hidden_dim": 768,
            "torch_mlp_depth": 3,
            "torch_batch_episodes": 1536,
            "torch_minibatch_size": 768,
            "torch_update_epochs": 4,
            "torch_entropy_coef": 0.008,
            "torch_preflight_batch_size": 12288,
            "torch_preflight_min_free_memory_gb": 10.0,
            "benchmark_training_episodes": 768,
            "benchmark_evaluation_episodes": 96,
        },
    ),
    "rtx3090": GpuProfile(
        name="rtx3090",
        description="24 GB Ampere profile with slightly smaller batches than 4090.",
        overrides={
            "torch_hidden_dim": 640,
            "torch_mlp_depth": 3,
            "torch_batch_episodes": 768,
            "torch_minibatch_size": 384,
            "torch_update_epochs": 4,
            "torch_entropy_coef": 0.008,
            "torch_preflight_batch_size": 6144,
            "torch_preflight_min_free_memory_gb": 9.0,
            "benchmark_training_episodes": 640,
            "benchmark_evaluation_episodes": 80,
        },
    ),
    "rtx4080": GpuProfile(
        name="rtx4080",
        description="16 GB Ada Lovelace profile tuned for mid-high consumer cards.",
        overrides={
            "torch_hidden_dim": 640,
            "torch_mlp_depth": 3,
            "torch_batch_episodes": 640,
            "torch_minibatch_size": 320,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 4096,
            "torch_preflight_min_free_memory_gb": 7.0,
            "replay_buffer_capacity": 48_000,
        },
    ),
    "rtx4070": GpuProfile(
        name="rtx4070",
        description="12 GB consumer profile for RTX 4070/4070 Ti class hardware.",
        overrides={
            "torch_hidden_dim": 512,
            "torch_mlp_depth": 3,
            "torch_batch_episodes": 512,
            "torch_minibatch_size": 256,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 3072,
            "torch_preflight_min_free_memory_gb": 5.0,
            "benchmark_training_episodes": 768,
            "benchmark_evaluation_episodes": 96,
            "replay_buffer_capacity": 32_768,
        },
    ),
    "consumer_8gb": GpuProfile(
        name="consumer_8gb",
        description="8 GB fallback profile for smaller consumer GPUs.",
        overrides={
            "torch_hidden_dim": 384,
            "torch_mlp_depth": 2,
            "torch_batch_episodes": 256,
            "torch_minibatch_size": 128,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 2048,
            "torch_preflight_min_free_memory_gb": 3.5,
            "benchmark_training_episodes": 512,
            "benchmark_evaluation_episodes": 64,
            "replay_buffer_capacity": 20_000,
        },
    ),
    "l4": GpuProfile(
        name="l4",
        description="24 GB datacenter inference profile for NVIDIA L4.",
        overrides={
            "torch_hidden_dim": 640,
            "torch_mlp_depth": 3,
            "torch_batch_episodes": 768,
            "torch_minibatch_size": 384,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 6144,
            "torch_preflight_min_free_memory_gb": 9.0,
        },
    ),
    "pro_48gb": GpuProfile(
        name="pro_48gb",
        description="48 GB workstation/datacenter profile for L40, RTX 6000 Ada, or A6000-class cards.",
        overrides={
            "torch_hidden_dim": 896,
            "torch_mlp_depth": 4,
            "torch_batch_episodes": 1536,
            "torch_minibatch_size": 768,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 12288,
            "torch_preflight_min_free_memory_gb": 18.0,
        },
    ),
    "a100_40gb": GpuProfile(
        name="a100_40gb",
        description="40 GB datacenter profile for A100-class accelerators.",
        overrides={
            "torch_hidden_dim": 1024,
            "torch_mlp_depth": 4,
            "torch_batch_episodes": 1536,
            "torch_minibatch_size": 768,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 12288,
            "torch_preflight_min_free_memory_gb": 16.0,
        },
    ),
    "a100_80gb": GpuProfile(
        name="a100_80gb",
        description="80 GB datacenter profile for A100 80 GB accelerators.",
        overrides={
            "torch_hidden_dim": 1152,
            "torch_mlp_depth": 4,
            "torch_batch_episodes": 2048,
            "torch_minibatch_size": 1024,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 16384,
            "torch_preflight_min_free_memory_gb": 30.0,
        },
    ),
    "h100": GpuProfile(
        name="h100",
        description="High-end Hopper profile for H100-class accelerators.",
        overrides={
            "torch_hidden_dim": 1280,
            "torch_mlp_depth": 4,
            "torch_batch_episodes": 2048,
            "torch_minibatch_size": 1024,
            "torch_update_epochs": 4,
            "torch_preflight_batch_size": 16384,
            "torch_preflight_min_free_memory_gb": 30.0,
        },
    ),
}

# Simulator ``HardwareProfile`` tuning keyed by :func:`infer_gpu_profile` names (not training overrides).
SIMULATOR_PROFILE_TUNING: dict[str, dict[str, float]] = {
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


def available_gpu_profiles() -> list[str]:
    return sorted(GPU_PROFILES.keys())


def normalize_gpu_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def infer_gpu_profile(device_name: str | None, total_memory_gb: float | None) -> str:
    normalized = normalize_gpu_name(device_name or "")
    if "h100" in normalized:
        return "h100"
    if "a100" in normalized and ("80" in normalized or (total_memory_gb or 0.0) >= 70.0):
        return "a100_80gb"
    if "a100" in normalized:
        return "a100_40gb"
    if "l40" in normalized or "rtx 6000 ada" in normalized or "a6000" in normalized:
        return "pro_48gb"
    if re.search(r"\bl4\b", normalized):
        return "l4"
    if "4090" in normalized:
        return "rtx4090"
    if "3090" in normalized:
        return "rtx3090"
    if "4080" in normalized:
        return "rtx4080"
    if "4070" in normalized:
        return "rtx4070"

    memory = total_memory_gb or 0.0
    if memory >= 90.0:
        return "h100"
    if memory >= 70.0:
        return "a100_80gb"
    if memory >= 44.0:
        return "pro_48gb"
    if memory >= 35.0:
        return "a100_40gb"
    if memory >= 22.0:
        return "rtx4090"
    if memory >= 15.0:
        return "rtx4080"
    if memory >= 11.0:
        return "rtx4070"
    return "consumer_8gb"


def resolve_gpu_profile(
    requested_profile: str,
    device_name: str | None = None,
    total_memory_gb: float | None = None,
) -> str:
    if requested_profile and requested_profile != "auto":
        if requested_profile not in GPU_PROFILES:
            supported = ", ".join(available_gpu_profiles())
            raise ValueError(
                f"Unsupported GPU profile '{requested_profile}'. Available profiles: {supported}"
            )
        return requested_profile
    return infer_gpu_profile(device_name, total_memory_gb)


def apply_gpu_profile(
    config: FrameworkConfig,
    requested_profile: str | None = None,
    device_name: str | None = None,
    total_memory_gb: float | None = None,
) -> tuple[FrameworkConfig, dict[str, object]]:
    requested = requested_profile or config.torch_gpu_profile
    selected = resolve_gpu_profile(
        requested, device_name=device_name, total_memory_gb=total_memory_gb
    )
    profile = GPU_PROFILES[selected]
    updated = config.clone(torch_gpu_profile=selected, **profile.overrides)
    metadata = {
        "requested_gpu_profile": requested,
        "selected_gpu_profile": selected,
        "gpu_profile_description": profile.description,
        "detected_device_name": device_name,
        "detected_total_memory_gb": total_memory_gb,
        "applied_overrides": profile.overrides,
    }
    return updated, metadata
