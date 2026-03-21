from __future__ import annotations

import json
import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, write_json, write_scatter_plot
from adaptive_quant.math_utils import mean


def analyze(history_path: str, output_dir: str) -> dict[str, object]:
    source = Path(history_path)
    if not source.exists():
        summary = {"history_path": history_path, "records": 0}
        write_json(str(ensure_directory(output_dir) / "training_dynamics_summary.json"), summary)
        return summary

    records = json.loads(source.read_text(encoding="utf-8"))
    output_root = ensure_directory(output_dir)
    rewards = [float(record.get("batch_reward", record.get("reward", 0.0))) for record in records]
    points = [(float(record.get("step", 0.0)), float(record.get("batch_reward", record.get("reward", 0.0)))) for record in records]
    summary = {
        "history_path": history_path,
        "records": len(records),
        "mean_reward": mean(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
    }
    write_json(str(output_root / "training_dynamics_summary.json"), summary)
    write_scatter_plot(
        str(output_root / "training_reward_curve.svg"),
        "Training Reward Curve",
        points,
        "Update Step",
        "Reward",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/training_dynamics.py <history_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote training dynamics analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)
