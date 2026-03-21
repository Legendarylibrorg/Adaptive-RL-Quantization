from __future__ import annotations

from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.quant_function_behavior import analyze as analyze_quant
from config_gpu import CONFIG_GPU
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.gpu_profiles import apply_gpu_profile
from adaptive_quant.logging_utils import write_json
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, torch
from adaptive_quant.torch_preflight import run_torch_preflight
from adaptive_quant.trainer import build_trainer


def main() -> None:
    try:
        config, gpu_profile_report = _resolve_runtime_gpu_config(CONFIG_GPU)
        trainer = build_trainer(config)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    preflight_report = None
    if config.torch_preflight:
        preflight_report = run_torch_preflight(config, trainer.policy)
        preflight_report["gpu_profile"] = gpu_profile_report
        write_json(f"{config.benchmark_dir}/{config.run_name}_preflight.json", preflight_report)
        print("Preflight summary:", preflight_report)
    train_summary = trainer.train()
    eval_summary = trainer.evaluate()
    benchmark_summary = BenchmarkSuite(config).run()

    analysis_root = f"{config.analysis_dir}/{config.run_name}"
    hardware_summary = analyze_hardware(f"{config.log_dir}/{config.run_name}_multi_hw.jsonl", f"{analysis_root}/hardware")
    input_summary = analyze_inputs(f"{config.log_dir}/{config.run_name}_dynamic.jsonl", f"{analysis_root}/inputs")
    quant_summary = analyze_quant(f"{config.log_dir}/{config.run_name}_learned.jsonl", f"{analysis_root}/quant")

    write_json(
        f"{config.benchmark_dir}/{config.run_name}_summary.json",
        {
            "train": train_summary,
            "evaluation": eval_summary,
            "benchmarks": benchmark_summary,
            "preflight": preflight_report,
            "gpu_profile": gpu_profile_report,
            "analysis": {
                "hardware": hardware_summary,
                "input": input_summary,
                "quant_function": quant_summary,
            },
        },
    )
    print("GPU profile:", gpu_profile_report)
    print("Training summary:", train_summary)
    print("Evaluation summary:", eval_summary)
    print("Benchmark summary written to:", f"{config.benchmark_dir}/{config.run_name}_summary.json")


def _resolve_runtime_gpu_config(config):
    if torch is None:
        raise ImportError(
            "PyTorch is required for `training_backend=\"pytorch\"`. "
            "Install a CUDA-enabled PyTorch build before running `run_pytorch_gpu.py`."
        ) from TORCH_IMPORT_ERROR
    device_name = None
    total_memory_gb = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_name = properties.name
        total_memory_gb = round(properties.total_memory / (1024 ** 3), 2)
    return apply_gpu_profile(config, device_name=device_name, total_memory_gb=total_memory_gb)


if __name__ == "__main__":
    main()
