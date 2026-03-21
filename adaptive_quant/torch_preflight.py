from __future__ import annotations

import inspect
import time

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, TorchPolicyAdapter, torch


def run_torch_preflight(config: FrameworkConfig, policy: TorchPolicyAdapter) -> dict[str, object]:
    if torch is None:
        raise ImportError("PyTorch is not installed in this environment.") from TORCH_IMPORT_ERROR

    report = collect_torch_system_report(config, policy)
    report["throughput_benchmark"] = benchmark_policy_throughput(config, policy)
    return report


def collect_torch_system_report(config: FrameworkConfig, policy: TorchPolicyAdapter) -> dict[str, object]:
    if torch is None:
        raise ImportError("PyTorch is not installed in this environment.") from TORCH_IMPORT_ERROR

    device = policy.device
    warnings: list[str] = []
    recommendations: list[str] = []
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters

    report: dict[str, object] = {
        "requested_device": config.torch_device,
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "compile_enabled": bool(config.torch_compile and hasattr(torch, "compile")),
        "fused_optimizer_requested": config.torch_fused_optimizer,
        "fused_optimizer_available": fused_available,
        "dtype": config.torch_dtype,
        "tf32_enabled": bool(config.torch_tf32),
        "warnings": warnings,
        "recommendations": recommendations,
    }

    if device.type != "cuda" or not torch.cuda.is_available():
        warnings.append("CUDA is not available. The 4090 path will not run efficiently on this machine.")
        return report

    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = _mem_get_info(index)
    free_gb = round(free_bytes / (1024 ** 3), 2)
    total_gb = round(total_bytes / (1024 ** 3), 2)
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    report.update(
        {
            "device_index": index,
            "device_name": properties.name,
            "device_capability": f"{properties.major}.{properties.minor}",
            "device_total_memory_gb": total_gb,
            "device_free_memory_gb": free_gb,
            "bf16_supported": bf16_supported,
            "batch_episodes": config.torch_batch_episodes,
            "minibatch_size": config.torch_minibatch_size,
        }
    )

    if "bf16" in config.torch_dtype.lower() and not bf16_supported:
        warnings.append("Requested bf16, but this CUDA stack does not report bf16 support.")
    if free_gb < config.torch_preflight_min_free_memory_gb:
        warnings.append(
            f"Only {free_gb:.2f} GB free on the selected GPU. "
            f"Consider closing other processes before training."
        )
    if config.torch_batch_episodes < config.torch_minibatch_size:
        warnings.append("`torch_batch_episodes` is smaller than `torch_minibatch_size`; this wastes update capacity.")
    if not config.torch_compile:
        recommendations.append("Enable `torch_compile` for longer 4090 runs once the setup is stable.")
    if not config.cache_prompt_features:
        recommendations.append("Enable `cache_prompt_features` to reduce CPU-side rollout overhead.")
    if not config.torch_fused_optimizer and fused_available:
        recommendations.append("Enable `torch_fused_optimizer` for slightly better optimizer throughput on CUDA.")
    return report


def benchmark_policy_throughput(config: FrameworkConfig, policy: TorchPolicyAdapter) -> dict[str, object]:
    if torch is None:
        raise ImportError("PyTorch is not installed in this environment.") from TORCH_IMPORT_ERROR

    batch_size = config.torch_preflight_batch_size
    warmup_steps = config.torch_preflight_warmup_steps
    timed_steps = config.torch_preflight_steps
    states = torch.randn(batch_size, policy.model.state_dim, device=policy.device, dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            with policy.autocast_context():
                _ = policy.model(states)
    _synchronize(policy.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(timed_steps):
            with policy.autocast_context():
                _ = policy.model(states)
    _synchronize(policy.device)
    inference_seconds = max(time.perf_counter() - start, 1e-9)

    model = policy.model
    model.train()
    for _ in range(warmup_steps):
        model.zero_grad(set_to_none=True)
        with policy.autocast_context():
            outputs = model(states)
            loss = _synthetic_loss(outputs)
        loss.backward()
    _synchronize(policy.device)
    start = time.perf_counter()
    for _ in range(timed_steps):
        model.zero_grad(set_to_none=True)
        with policy.autocast_context():
            outputs = model(states)
            loss = _synthetic_loss(outputs)
        loss.backward()
    _synchronize(policy.device)
    train_seconds = max(time.perf_counter() - start, 1e-9)
    model.zero_grad(set_to_none=True)
    model.eval()

    return {
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "inference_ms_per_step": round((inference_seconds / timed_steps) * 1000.0, 4),
        "inference_samples_per_second": round((batch_size * timed_steps) / inference_seconds, 2),
        "backward_ms_per_step": round((train_seconds / timed_steps) * 1000.0, 4),
        "backward_samples_per_second": round((batch_size * timed_steps) / train_seconds, 2),
    }


def _synthetic_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = outputs["value"].float().pow(2).mean()
    loss = loss + 0.01 * outputs["discrete_logits"].float().pow(2).mean()
    loss = loss + 0.01 * outputs["group_logits"].float().pow(2).mean()
    loss = loss + 0.01 * outputs["layer_logits"].float().pow(2).mean()
    loss = loss + 0.01 * outputs["learned_mean"].float().pow(2).mean()
    if "mode_logits" in outputs:
        loss = loss + 0.01 * outputs["mode_logits"].float().pow(2).mean()
    return loss


def _mem_get_info(index: int) -> tuple[int, int]:
    try:
        return torch.cuda.mem_get_info(index)
    except TypeError:
        return torch.cuda.mem_get_info()


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
