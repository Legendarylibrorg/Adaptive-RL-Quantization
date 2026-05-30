from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from adaptive_quant.cli.common import add_config_file_argument, load_config_or_fallback
from adaptive_quant.cli.presets import apply_short_run_episodes, select_dense_moe_preset
from adaptive_quant.cli.startup_overrides import apply_startup_overrides, enforce_privileged_override_policy
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.experiment_aggregate import extract_metric
from adaptive_quant.logging_utils import md_table, write_json, write_text_file
from adaptive_quant.math_utils import fmt_float
from adaptive_quant.paper_bundle import create_multiseed_paper_bundle
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.presets.baseline import CONFIG as CONFIG_DENSE
from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from adaptive_quant.sweep import (
    DEFAULT_OBJECTIVE,
    SweepSpec,
    SweepTrialPlan,
    SweepTrialResult,
    build_trial_plans,
    load_sweep_file,
    parse_vary_argument,
    rank_trials,
)


def _resolve_base_config(args: argparse.Namespace) -> FrameworkConfig:
    if args.sweep_config:
        spec, file_base = load_sweep_file(args.sweep_config)
        args._sweep_spec = spec  # type: ignore[attr-defined]
        if file_base is not None:
            return file_base
    if args.config:
        return load_config_or_fallback(args.config, CONFIG_DENSE)
    if args.preset:
        return select_dense_moe_preset(args.preset)
    return CONFIG_DENSE


def _resolve_sweep_spec(args: argparse.Namespace) -> SweepSpec:
    if hasattr(args, "_sweep_spec"):
        return args._sweep_spec  # type: ignore[attr-defined]

    grid: dict[str, tuple[Any, ...]] = {}
    for raw in args.vary or ():
        try:
            key, values = parse_vary_argument(raw)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        grid[key] = values

    direction = args.direction.strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise SystemExit("direction must be 'maximize' or 'minimize'")

    return SweepSpec(
        objective=args.objective,
        direction=direction,  # type: ignore[arg-type]
        seed=args.seed,
        grid=grid or None,
        trials=None,
        base_config_path=None,
    )


def _apply_trial_overrides(base_config: FrameworkConfig, plan: SweepTrialPlan) -> FrameworkConfig:
    enforce_privileged_override_policy(plan.overrides)
    return apply_startup_overrides(base_config, plan.overrides)


def _write_sweep_report(
    *,
    run_name: str,
    objective: str,
    direction: str,
    ranked_results: list[SweepTrialResult],
    output_path: str,
    output_json_path: str,
) -> None:
    leaderboard_rows: list[list[object]] = []
    for rank, result in enumerate(ranked_results, start=1):
        objective_value = result.objective_value
        override_bits = ", ".join(
            f"{key}={value!r}" for key, value in sorted(result.plan.overrides.items())
        )
        leaderboard_rows.append(
            [
                str(rank),
                str(result.plan.trial_id),
                result.plan.run_name_suffix,
                fmt_float(objective_value) if objective_value is not None else "n/a",
                override_bits or "(defaults)",
                f"`{result.summary_path}`",
            ]
        )

    lines = [
        f"# {run_name} (hyperparameter sweep)",
        "",
        "## Overview",
        f"- objective: `{objective}` ({direction})",
        f"- trials: `{len(ranked_results)}`",
        f"- aggregate JSON: `{output_json_path}`",
        "",
        "## Leaderboard",
        "\n".join(
            md_table(
                ["rank", "trial_id", "suffix", "objective", "overrides", "summary"],
                leaderboard_rows,
            )
        ),
        "",
        "## Notes",
        "- Trials are ranked by the selected objective metric extracted from each pipeline summary.",
        "- Open per-trial `outputs/reports/*_report.md` files for full benchmark and analysis artifacts.",
        "",
    ]
    write_text_file(output_path, "\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter sweep and rank trials by an objective metric."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load sweep grid/trials/objective from a .json or .toml file.",
    )
    add_config_file_argument(parser, help_suffix=" Used when --sweep-config is omitted.")
    parser.add_argument(
        "--preset",
        choices=["dense", "moe"],
        default="dense",
        help="Base preset when no --config or sweep base_config is provided.",
    )
    parser.add_argument(
        "--vary",
        action="append",
        default=None,
        metavar="KEY=val1,val2",
        help=(
            "Grid-search one parameter. Repeat for cartesian products, e.g. "
            "--vary learning_rate=0.01,0.035 --vary reward_weights.beta_throughput=0.04,0.08"
        ),
    )
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help="Dotted metric path to rank trials (default: evaluation.mean_reward).",
    )
    parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether higher or lower objective values rank better.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Base run name for sweep artifacts (defaults to base config run_name).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed applied to every trial unless overridden in the sweep file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training_episodes for every trial (useful for smoke tests).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress end-of-run CLI banners (e.g. unit tests)."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_config = _resolve_base_config(args)
    spec = _resolve_sweep_spec(args)

    if args.episodes is not None:
        base_config = apply_short_run_episodes(base_config, args.episodes)

    seed = spec.seed if spec.seed is not None else args.seed
    if seed is not None:
        base_config = base_config.clone(seed=seed)

    try:
        plans = build_trial_plans(grid=spec.grid, explicit_trials=list(spec.trials or ()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_run_name = str(args.run_name or base_config.run_name)
    sweep_run_name = f"{base_run_name}_sweep"

    results: list[SweepTrialResult] = []
    for plan in plans:
        trial_run_name = f"{base_run_name}_trial{plan.trial_id:03d}_{plan.run_name_suffix}"
        trial_config = _apply_trial_overrides(base_config, plan).clone(run_name=trial_run_name)
        summary = run_pipeline_entrypoint(
            trial_config,
            footer_mode="none" if args.quiet else "minimal",
        )
        objective_value = extract_metric(summary, spec.objective)
        results.append(
            SweepTrialResult(
                plan=plan,
                summary=summary,
                summary_path=trial_config.summary_path(),
                objective_value=objective_value,
            )
        )

    ranked = rank_trials(results, objective=spec.objective, direction=spec.direction)

    output_json_path = f"{base_config.benchmark_dir}/{sweep_run_name}_summary.json"
    output_md_path = f"{base_config.report_dir}/{sweep_run_name}_report.md"

    aggregate_payload: dict[str, Any] = {
        "run_name": sweep_run_name,
        "base_run_name": base_run_name,
        "objective": spec.objective,
        "direction": spec.direction,
        "config": asdict(base_config),
        "git_commit": git_commit_hash(),
        "sweep": {
            "grid": spec.grid,
            "trials": spec.trials,
            "base_config_path": spec.base_config_path,
            "sweep_config_path": args.sweep_config,
        },
        "trials": [
            {
                "trial_id": result.plan.trial_id,
                "run_name_suffix": result.plan.run_name_suffix,
                "overrides": result.plan.overrides,
                "objective_value": result.objective_value,
                "summary_path": result.summary_path,
            }
            for result in results
        ],
        "leaderboard": [
            {
                "rank": rank,
                "trial_id": result.plan.trial_id,
                "run_name_suffix": result.plan.run_name_suffix,
                "objective_value": result.objective_value,
                "summary_path": result.summary_path,
            }
            for rank, result in enumerate(ranked, start=1)
        ],
        "artifacts": {
            "report": output_md_path,
            "per_trial_summaries": [result.summary_path for result in results],
        },
    }

    sweep_stats: dict[str, dict[str, float | int]] = {}
    for result in results:
        if result.objective_value is None:
            continue
        sweep_stats[f"trial_{result.plan.trial_id:03d}.{spec.objective}"] = {
            "mean": result.objective_value,
            "std": 0.0,
            "n": 1,
            "stderr": 0.0,
            "ci95_low": result.objective_value,
            "ci95_high": result.objective_value,
            "effect_size_vs_zero": 0.0,
        }
    paper_bundle = create_multiseed_paper_bundle(
        config=base_config,
        run_name=sweep_run_name,
        aggregate_payload=aggregate_payload,
        aggregate_stats=sweep_stats,
        report_path=output_md_path,
    )
    aggregate_payload["artifacts"]["paper_bundle"] = paper_bundle
    write_json(output_json_path, aggregate_payload)
    _write_sweep_report(
        run_name=sweep_run_name,
        objective=spec.objective,
        direction=spec.direction,
        ranked_results=ranked,
        output_path=output_md_path,
        output_json_path=output_json_path,
    )

    if not args.quiet:
        from adaptive_quant.run_footer import print_sweep_footer

        print_sweep_footer(
            sweep_run_name=sweep_run_name,
            objective=spec.objective,
            direction=spec.direction,
            trial_count=len(results),
            best_trial=ranked[0] if ranked else None,
            aggregate_json=output_json_path,
            report_md=output_md_path,
        )


if __name__ == "__main__":
    main()
