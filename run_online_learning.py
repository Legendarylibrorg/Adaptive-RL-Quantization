from __future__ import annotations

from analysis.online_learning import analyze as analyze_online
from config_online import CONFIG_ONLINE
from adaptive_quant.logging_utils import write_json
from adaptive_quant.online_learning import OnlineLearningLoop, build_request_stream
from adaptive_quant.trainer import build_trainer


def main() -> None:
    trainer = build_trainer(CONFIG_ONLINE)
    try:
        bootstrap_summary = trainer.train()
        loop = OnlineLearningLoop(CONFIG_ONLINE, trainer=trainer)
        online_summary = loop.run_stream(build_request_stream(CONFIG_ONLINE))
        eval_summary = trainer.evaluate()
    finally:
        if "loop" in locals():
            loop.close()
        trainer.close()

    analysis_root = f"{CONFIG_ONLINE.analysis_dir}/{CONFIG_ONLINE.run_name}"
    online_analysis = analyze_online(CONFIG_ONLINE.online_telemetry_path(), f"{analysis_root}/online")
    write_json(
        f"{CONFIG_ONLINE.benchmark_dir}/{CONFIG_ONLINE.run_name}_summary.json",
        {
            "bootstrap_train": bootstrap_summary,
            "online": online_summary,
            "evaluation": eval_summary,
            "analysis": {"online_learning": online_analysis},
        },
    )
    print("Bootstrap summary:", bootstrap_summary)
    print("Online summary:", online_summary)
    print("Evaluation summary:", eval_summary)
    print("Online summary written to:", f"{CONFIG_ONLINE.benchmark_dir}/{CONFIG_ONLINE.run_name}_summary.json")


if __name__ == "__main__":
    main()
