from __future__ import annotations

from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.quant_function_behavior import analyze as analyze_quant
from config import CONFIG
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.logging_utils import write_json
from adaptive_quant.trainer import build_trainer


def main() -> None:
    trainer = build_trainer(CONFIG)
    try:
        train_summary = trainer.train()
        eval_summary = trainer.evaluate()
    finally:
        trainer.close()
    benchmark_summary = BenchmarkSuite(CONFIG).run()

    analysis_root = f"{CONFIG.analysis_dir}/{CONFIG.run_name}"
    hardware_summary = analyze_hardware(f"{CONFIG.log_dir}/{CONFIG.run_name}_multi_hw.jsonl", f"{analysis_root}/hardware")
    input_summary = analyze_inputs(f"{CONFIG.log_dir}/{CONFIG.run_name}_dynamic.jsonl", f"{analysis_root}/inputs")
    quant_summary = analyze_quant(f"{CONFIG.log_dir}/{CONFIG.run_name}_learned.jsonl", f"{analysis_root}/quant")

    write_json(
        f"{CONFIG.benchmark_dir}/{CONFIG.run_name}_summary.json",
        {
            "train": train_summary,
            "evaluation": eval_summary,
            "benchmarks": benchmark_summary,
            "analysis": {
                "hardware": hardware_summary,
                "input": input_summary,
                "quant_function": quant_summary,
            },
        },
    )
    print("Training summary:", train_summary)
    print("Evaluation summary:", eval_summary)
    print("Benchmark summary written to:", f"{CONFIG.benchmark_dir}/{CONFIG.run_name}_summary.json")


if __name__ == "__main__":
    main()
