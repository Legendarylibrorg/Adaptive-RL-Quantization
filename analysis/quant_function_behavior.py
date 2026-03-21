from __future__ import annotations

import sys
from pathlib import Path

from adaptive_quant.analysis_utils import ensure_directory, load_jsonl, write_bar_chart, write_json
from adaptive_quant.math_utils import mean


def analyze(log_path: str, output_dir: str) -> dict[str, object]:
    records = load_jsonl(log_path)
    output_root = ensure_directory(output_dir)
    learned_records = [record for record in records if record.get("decision", {}).get("mode") == "learned"]

    scale_values = [float(record["decision"].get("scale_factor", 0.0)) for record in learned_records]
    clip_values = [float(record["decision"].get("clipping_range", 0.0)) for record in learned_records]
    precision_values = [float(record["decision"].get("precision_level", 0.0)) for record in learned_records]
    average_bits = []
    for record in learned_records:
        bits = record.get("decision", {}).get("effective_layer_bits", [])
        if bits:
            average_bits.append(mean([float(bit) for bit in bits]))

    summary = {
        "log_path": log_path,
        "learned_episode_count": len(learned_records),
        "scale_factor": {"mean": mean(scale_values), "min": min(scale_values) if scale_values else 0.0, "max": max(scale_values) if scale_values else 0.0},
        "clipping_range": {"mean": mean(clip_values), "min": min(clip_values) if clip_values else 0.0, "max": max(clip_values) if clip_values else 0.0},
        "precision_level": {"mean": mean(precision_values), "min": min(precision_values) if precision_values else 0.0, "max": max(precision_values) if precision_values else 0.0},
        "effective_bits_mean": mean(average_bits),
    }
    write_json(str(output_root / "quant_function_behavior_summary.json"), summary)
    write_bar_chart(
        str(output_root / "quant_function_parameters.svg"),
        "Learned Quantization Parameters",
        {
            "scale": summary["scale_factor"]["mean"],
            "clip": summary["clipping_range"]["mean"],
            "precision": summary["precision_level"]["mean"],
            "bits": summary["effective_bits_mean"],
        },
        "Average value",
    )
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 analysis/quant_function_behavior.py <log_path> <output_dir>")
    summary = analyze(sys.argv[1], sys.argv[2])
    print(f"Wrote quant function analysis to {Path(sys.argv[2]).resolve()}")
    print(summary)

