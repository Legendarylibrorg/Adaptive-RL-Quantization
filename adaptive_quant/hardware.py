from __future__ import annotations

from adaptive_quant.types import HardwareProfile, HardwareType


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

