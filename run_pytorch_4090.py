from __future__ import annotations

from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.quant_function_behavior import analyze as analyze_quant
from config_4090 import CONFIG_4090
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.logging_utils import write_json
from adaptive_quant.torch_preflight import run_torch_preflight
from adaptive_quant.trainer import build_trainer


def main() -> None:
    try:
        trainer = build_trainer(CONFIG_4090)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    preflight_report = None
    if CONFIG_4090.torch_preflight:
        preflight_report = run_torch_preflight(CONFIG_4090, trainer.policy)
        write_json(f"{CONFIG_4090.benchmark_dir}/{CONFIG_4090.run_name}_preflight.json", preflight_report)
        print("Preflight summary:", preflight_report)
    train_summary = trainer.train()
    eval_summary = trainer.evaluate()
    benchmark_summary = BenchmarkSuite(CONFIG_4090).run()

    analysis_root = f"{CONFIG_4090.analysis_dir}/{CONFIG_4090.run_name}"
    hardware_summary = analyze_hardware(f"{CONFIG_4090.log_dir}/{CONFIG_4090.run_name}_multi_hw.jsonl", f"{analysis_root}/hardware")
    input_summary = analyze_inputs(f"{CONFIG_4090.log_dir}/{CONFIG_4090.run_name}_dynamic.jsonl", f"{analysis_root}/inputs")
    quant_summary = analyze_quant(f"{CONFIG_4090.log_dir}/{CONFIG_4090.run_name}_learned.jsonl", f"{analysis_root}/quant")

    write_json(
        f"{CONFIG_4090.benchmark_dir}/{CONFIG_4090.run_name}_summary.json",
        {
            "train": train_summary,
            "evaluation": eval_summary,
            "benchmarks": benchmark_summary,
            "preflight": preflight_report,
            "analysis": {
                "hardware": hardware_summary,
                "input": input_summary,
                "quant_function": quant_summary,
            },
        },
    )
    print("Training summary:", train_summary)
    print("Evaluation summary:", eval_summary)
    print("Benchmark summary written to:", f"{CONFIG_4090.benchmark_dir}/{CONFIG_4090.run_name}_summary.json")


if __name__ == "__main__":
    main()
