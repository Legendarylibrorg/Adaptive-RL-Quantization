from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from adaptive_quant.logging_utils import md_table, write_json, write_text_file
from adaptive_quant.math_utils import fmt_float
from adaptive_quant.paper_bundle import aggregate_values, create_multiseed_paper_bundle
from adaptive_quant.presets.baseline import CONFIG as CONFIG_DENSE
from adaptive_quant.presets.moe import CONFIG_MOE
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.research_pipeline import run_pipeline_entrypoint


@dataclass(frozen=True)
class AggregateStat:
    mean: float
    std: float
    n: int
    stderr: float = 0.0
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    effect_size_vs_zero: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "stderr": self.stderr,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "effect_size_vs_zero": self.effect_size_vs_zero,
        }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_numeric(
    obj: object,
    *,
    prefix: str = "",
    max_items: int = 10_000,
) -> dict[str, float]:
    out: dict[str, float] = {}

    def walk(node: object, path: str) -> None:
        if len(out) >= max_items:
            return
        if _is_number(node):
            out[path] = float(node)  # type: ignore[arg-type]
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    continue
                walk(v, f"{path}.{k}" if path else k)
            return
        if isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(obj, prefix)
    return out


def _default_key_filter(key: str) -> bool:
    key_lower = key.lower()
    if key_lower.startswith("config."):
        return False
    if key_lower.endswith("_delta"):
        return True
    if "gap" in key_lower:
        return True
    for needle in (
        "mean_reward",
        "mean_latency_ms",
        "mean_throughput_tps",
        "mean_memory_mb",
        "mean_perplexity",
        "mean_stability_penalty",
        "generalization_gap_improvement",
        "single_policy_gap",
        "multi_policy_gap",
    ):
        if needle in key_lower:
            return True
    return False


def _aggregate_numeric_maps(maps: list[dict[str, float]]) -> dict[str, AggregateStat]:
    keys: set[str] = set()
    for m in maps:
        keys.update(m.keys())

    aggregated: dict[str, AggregateStat] = {}
    for key in sorted(keys):
        values = [m[key] for m in maps if key in m and math.isfinite(m[key])]
        if not values:
            continue
        stats = aggregate_values(values)
        aggregated[key] = AggregateStat(
            mean=float(stats["mean"]),
            std=float(stats["std"]),
            n=int(stats["n"]),
            stderr=float(stats["stderr"]),
            ci95_low=float(stats["ci95_low"]),
            ci95_high=float(stats["ci95_high"]),
            effect_size_vs_zero=float(stats["effect_size_vs_zero"]),
        )
    return aggregated


def _write_multiseed_report(
    *,
    run_name: str,
    seeds: list[int],
    per_seed_summaries: list[dict[str, Any]],
    per_seed_paths: list[str],
    aggregated: dict[str, AggregateStat],
    output_path: str,
    output_json_path: str,
) -> None:
    headline_keys = [
        "benchmarks.single_vs_multi.generalization_gap_improvement",
        "benchmarks.static_vs_dynamic.quality_variance_delta",
        "benchmarks.discrete_vs_learned.reward_delta",
        "evaluation.mean_reward",
        "evaluation.mean_latency_ms",
        "evaluation.mean_memory_mb",
    ]

    headline_rows: list[list[object]] = []
    for k in headline_keys:
        stat = aggregated.get(k)
        if stat is None:
            continue
        headline_rows.append([k, fmt_float(stat.mean), fmt_float(stat.std), str(stat.n)])

    per_seed_rows: list[list[object]] = []
    for seed, summary_path in zip(seeds, per_seed_paths, strict=True):
        per_seed_rows.append([str(seed), f"`{summary_path}`"])

    lines: list[str] = []
    lines.extend(
        [
            f"# {run_name} (multi-seed)",
            "",
            "## Overview",
            f"- seeds: `{seeds}`",
            f"- per-seed summaries: `{len(per_seed_summaries)}`",
            f"- aggregate JSON: `{output_json_path}`",
            "",
        ]
    )

    if headline_rows:
        lines.extend(
            [
                "## Headline aggregates (mean ± std)",
                "\n".join(md_table(["metric", "mean", "std", "n"], headline_rows)),
                "",
            ]
        )

    filtered = {k: v for k, v in aggregated.items() if _default_key_filter(k)}
    broad_rows: list[list[object]] = [
        [
            k,
            fmt_float(v.mean),
            fmt_float(v.std),
            fmt_float(v.ci95_low),
            fmt_float(v.ci95_high),
            str(v.n),
        ]
        for k, v in filtered.items()
    ]
    lines.extend(
        [
            "## Aggregate metrics (filtered)",
            "\n".join(
                md_table(["metric", "mean", "std", "ci95_low", "ci95_high", "n"], broad_rows)
            ),
            "",
            "## Per-seed artifacts",
            "\n".join(md_table(["seed", "summary"], per_seed_rows)),
            "",
            "## Notes",
            "- These statistics summarize the metrics produced by the pipeline (simulator by default unless you switch backends in the preset config).",
            "- For deeper inspection, open a per-seed `outputs/reports/*_report.md` and the per-seed figures under `outputs/analysis/<run_name>/...`.",
            "",
        ]
    )
    write_text_file(output_path, "\n".join(lines) + "\n")


def _parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if "-" in raw and "," not in raw:
        left, right = raw.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _select_preset(name: str):
    if name == "dense":
        return CONFIG_DENSE
    if name == "moe":
        return CONFIG_MOE
    raise SystemExit(f"Unknown preset: {name!r} (expected 'dense' or 'moe')")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a preset across multiple seeds and aggregate results."
    )
    parser.add_argument(
        "--preset", choices=["dense", "moe"], default="dense", help="Which config preset to run."
    )
    parser.add_argument("--seeds", default="13,17,23,29,31", help='Seeds as "a,b,c" or "a-b".')
    parser.add_argument(
        "--run-name",
        default=None,
        help="Base run name for the multiseed aggregate (defaults to preset run_name).",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        type=int,
        help="Override training_episodes (useful for fast smoke tests).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress end-of-run CLI banners (e.g. unit tests)."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided.")

    base_config = _select_preset(args.preset)
    if args.episodes is not None:
        base_config = base_config.clone(
            training_episodes=args.episodes,
            evaluation_episodes=max(1, args.episodes // 4),
            continuous_training=False,
        )
    base_run_name = str(args.run_name or base_config.run_name)
    multiseed_run_name = f"{base_run_name}_multiseed"

    per_seed_summaries: list[dict[str, Any]] = []
    per_seed_paths: list[str] = []
    per_seed_numeric: list[dict[str, float]] = []

    for seed in seeds:
        seed_run_name = f"{base_run_name}_seed{seed}"
        config = base_config.clone(seed=seed, run_name=seed_run_name)
        summary = run_pipeline_entrypoint(config, footer_mode="none" if args.quiet else "minimal")
        per_seed_summaries.append(summary)
        per_seed_paths.append(config.summary_path())

        numeric = _flatten_numeric(summary)
        per_seed_numeric.append(numeric)

    aggregated = _aggregate_numeric_maps(per_seed_numeric)

    output_json_path = f"{base_config.benchmark_dir}/{multiseed_run_name}_summary.json"
    output_md_path = f"{base_config.report_dir}/{multiseed_run_name}_report.md"

    aggregate_payload: dict[str, Any] = {
        "run_name": multiseed_run_name,
        "preset": args.preset,
        "base_run_name": base_run_name,
        "config": asdict(base_config),
        "git_commit": git_commit_hash(),
        "seeds": seeds,
        "per_seed": [
            {"seed": seed, "run_name": f"{base_run_name}_seed{seed}", "summary_path": path}
            for seed, path in zip(seeds, per_seed_paths, strict=True)
        ],
        "artifacts": {
            "per_seed_summaries": per_seed_paths,
            "report": output_md_path,
        },
        "aggregates": {k: v.to_dict() for k, v in aggregated.items() if _default_key_filter(k)},
    }
    paper_bundle = create_multiseed_paper_bundle(
        config=base_config,
        run_name=multiseed_run_name,
        aggregate_payload=aggregate_payload,
        aggregate_stats=aggregate_payload["aggregates"],
        report_path=output_md_path,
    )
    aggregate_payload["artifacts"]["paper_bundle"] = paper_bundle
    write_json(output_json_path, aggregate_payload)
    _write_multiseed_report(
        run_name=multiseed_run_name,
        seeds=seeds,
        per_seed_summaries=per_seed_summaries,
        per_seed_paths=per_seed_paths,
        aggregated=aggregated,
        output_path=output_md_path,
        output_json_path=output_json_path,
    )

    if not args.quiet:
        from adaptive_quant.run_footer import print_multiseed_footer

        print_multiseed_footer(
            multiseed_run_name=multiseed_run_name,
            seeds=seeds,
            aggregate_json=output_json_path,
            report_md=output_md_path,
            per_seed_summary_paths=per_seed_paths,
        )


if __name__ == "__main__":
    main()
