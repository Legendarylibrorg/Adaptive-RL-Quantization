from __future__ import annotations

import argparse

from analysis.analyzers import analyze_online
from config_online import CONFIG_ONLINE
from adaptive_quant.logging_utils import write_json
from adaptive_quant.online_learning import OnlineLearningLoop, build_request_stream
from adaptive_quant.runner_cli import add_config_file_argument, load_config_or_fallback
from adaptive_quant.trainer import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Online learning loop (experimental).")
    add_config_file_argument(parser)
    args = parser.parse_args()
    cfg = load_config_or_fallback(args.config, CONFIG_ONLINE)

    trainer = build_trainer(cfg)
    try:
        bootstrap_summary = trainer.train()
        loop = OnlineLearningLoop(cfg, trainer=trainer)
        online_summary = loop.run_stream(build_request_stream(cfg))
        eval_summary = trainer.evaluate()
    finally:
        if "loop" in locals():
            loop.close()
        trainer.close()

    analysis_root = f"{cfg.analysis_dir}/{cfg.run_name}"
    online_analysis = analyze_online(cfg.online_telemetry_path(), f"{analysis_root}/online")
    write_json(
        f"{cfg.benchmark_dir}/{cfg.run_name}_summary.json",
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
    print("Online summary written to:", f"{cfg.benchmark_dir}/{cfg.run_name}_summary.json")


if __name__ == "__main__":
    main()
