from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from adaptive_quant.cli.common import (
    add_config_file_argument,
    load_config_or_fallback,
)
from adaptive_quant.cli.presets import (
    apply_short_run_episodes,
    select_dense_moe_preset,
)
from adaptive_quant.cli.startup_overrides import (
    apply_startup_overrides,
    enforce_privileged_override_policy,
)
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.experiment_aggregate import extract_metric
from adaptive_quant.logging_utils import md_table, read_json, write_json, write_text_file
from adaptive_quant.math_utils import fmt_float
from adaptive_quant.paper_bundle import create_multiseed_paper_bundle
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.presets.baseline import CONFIG as CONFIG_DENSE
from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from adaptive_quant.sweep import (
    DEFAULT_OBJECTIVE,
    SweepDirection,
    SweepSeedResult,
    SweepSpec,
    SweepTrialPlan,
    SweepTrialResult,
    aggregate_objective_values,
    build_trial_plans,
    format_sweep_plan_preview,
    load_sweep_file,
    parse_seed_list,
    parse_vary_argument,
    rank_trials,
    trial_run_name,
)

_HEADLINE_METRICS = (
    "evaluation.mean_reward",
    "evaluation.mean_latency_ms",
    "evaluation.mean_throughput_tps",
    "evaluation.mean_memory_mb",
    "benchmarks.single_vs_multi.generalization_gap_improvement",
)


def _default_base_config(args: argparse.Namespace) -> FrameworkConfig:
    if args.config:
        return load_config_or_fallback(args.config, CONFIG_DENSE)
    if args.preset:
        return select_dense_moe_preset(args.preset)
    return CONFIG_DENSE


def _apply_outputs_dir(base_config: FrameworkConfig, outputs_dir: str) -> FrameworkConfig:
    output_root = Path(outputs_dir)
    return base_config.clone(
        outputs_dir=str(output_root),
        log_dir=str(output_root / "logs"),
        benchmark_dir=str(output_root / "benchmarks"),
        analysis_dir=str(output_root / "analysis"),
        checkpoint_dir=str(output_root / "checkpoints"),
        report_dir=str(output_root / "reports"),
    )


def _build_cli_sweep_spec(args: argparse.Namespace) -> SweepSpec:
    grid: dict[str, tuple[Any, ...]] = {}
    for raw in args.vary or ():
        try:
            key, values = parse_vary_argument(raw)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        grid[key] = values

    direction_raw = args.direction.strip().lower()
    if direction_raw == "maximize":
        direction: SweepDirection = "maximize"
    elif direction_raw == "minimize":
        direction = "minimize"
    else:
        raise SystemExit("direction must be 'maximize' or 'minimize'")

    seeds = tuple(parse_seed_list(args.seeds)) if args.seeds else None

    return SweepSpec(
        objective=args.objective,
        direction=direction,
        seed=args.seed,
        seeds=seeds,
        grid=grid or None,
        trials=None,
        base_config_path=None,
    )


def _resolve_sweep_inputs(args: argparse.Namespace) -> tuple[FrameworkConfig, SweepSpec]:
    if args.sweep_config:
        spec, file_base = load_sweep_file(args.sweep_config)
        base_config = file_base if file_base is not None else _default_base_config(args)
        return base_config, spec
    return _default_base_config(args), _build_cli_sweep_spec(args)


def _resolve_seed_list(args: argparse.Namespace, spec: SweepSpec) -> list[int] | None:
    if spec.seeds:
        return list(spec.seeds)
    if args.seeds:
        seeds = parse_seed_list(args.seeds)
        return seeds or None
    return None


def _apply_trial_overrides(base_config: FrameworkConfig, plan: SweepTrialPlan) -> FrameworkConfig:
    enforce_privileged_override_policy(plan.overrides)
    return cast(
        FrameworkConfig,
        apply_startup_overrides(base_config, plan.overrides),
    )


def _load_or_run_trial(
    trial_config: FrameworkConfig,
    *,
    resume: bool,
    quiet: bool,
) -> tuple[dict[str, Any], str, bool]:
    summary_path = trial_config.summary_path()
    if resume and Path(summary_path).is_file():
        return read_json(summary_path, label="Trial summary"), summary_path, True
    summary = run_pipeline_entrypoint(
        trial_config,
        footer_mode="none" if quiet else "minimal",
    )
    return summary, summary_path, False


def _execute_trial(
    *,
    base_config: FrameworkConfig,
    plan: SweepTrialPlan,
    base_run_name: str,
    objective: str,
    seeds: list[int] | None,
    resume: bool,
    quiet: bool,
) -> SweepTrialResult:
    trial_config = _apply_trial_overrides(base_config, plan)
    runs_skipped = 0

    if seeds:
        seed_results: list[SweepSeedResult] = []
        for seed in seeds:
            run_name = trial_run_name(base_run_name, plan, seed=seed)
            config = trial_config.clone(run_name=run_name, seed=seed)
            summary, summary_path, skipped = _load_or_run_trial(
                config,
                resume=resume,
                quiet=quiet,
            )
            if skipped:
                runs_skipped += 1
            seed_results.append(
                SweepSeedResult(
                    seed=seed,
                    summary=summary,
                    summary_path=summary_path,
                    objective_value=extract_metric(summary, objective),
                    skipped=skipped,
                )
            )

        objective_values = [result.objective_value for result in seed_results]
        objective_mean, objective_std, objective_n = aggregate_objective_values(objective_values)
        representative = max(
            seed_results,
            key=lambda result: (
                result.objective_value is None,
                -(result.objective_value or float("-inf")),
            ),
        )
        return SweepTrialResult(
            plan=plan,
            summary=representative.summary,
            summary_path=representative.summary_path,
            objective_value=objective_mean,
            objective_std=objective_std,
            objective_n=objective_n,
            seed_results=tuple(seed_results),
            runs_skipped=runs_skipped,
        )

    run_name = trial_run_name(base_run_name, plan)
    config = trial_config.clone(run_name=run_name)
    summary, summary_path, skipped = _load_or_run_trial(
        config,
        resume=resume,
        quiet=quiet,
    )
    return SweepTrialResult(
        plan=plan,
        summary=summary,
        summary_path=summary_path,
        objective_value=extract_metric(summary, objective),
        runs_skipped=1 if skipped else 0,
    )


def _objective_display(result: SweepTrialResult) -> str:
    if result.objective_value is None:
        return "n/a"
    if result.objective_n > 1 and result.objective_std is not None:
        return f"{fmt_float(result.objective_value)} ± {fmt_float(result.objective_std)} (n={result.objective_n})"
    return fmt_float(result.objective_value)


def _write_sweep_csv(
    *,
    ranked_results: list[SweepTrialResult],
    output_path: str,
    objective: str,
) -> None:
    rows: list[dict[str, str]] = []
    for rank, result in enumerate(ranked_results, start=1):
        rows.append(
            {
                "rank": str(rank),
                "trial_id": str(result.plan.trial_id),
                "suffix": result.plan.run_name_suffix,
                "objective": objective,
                "objective_value": ""
                if result.objective_value is None
                else str(result.objective_value),
                "objective_std": "" if result.objective_std is None else str(result.objective_std),
                "objective_n": str(result.objective_n),
                "summary_path": result.summary_path,
            }
        )
    fieldnames = [
        "rank",
        "trial_id",
        "suffix",
        "objective",
        "objective_value",
        "objective_std",
        "objective_n",
        "summary_path",
    ]
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_report(
    *,
    run_name: str,
    objective: str,
    direction: str,
    ranked_results: list[SweepTrialResult],
    output_path: str,
    output_json_path: str,
    output_csv_path: str,
    seeds: list[int] | None,
    runs_skipped: int,
) -> None:
    leaderboard_rows: list[list[object]] = []
    for rank, result in enumerate(ranked_results, start=1):
        override_bits = ", ".join(
            f"{key}={value!r}" for key, value in sorted(result.plan.overrides.items())
        )
        leaderboard_rows.append(
            [
                str(rank),
                str(result.plan.trial_id),
                result.plan.run_name_suffix,
                _objective_display(result),
                override_bits or "(defaults)",
                f"`{result.summary_path}`",
            ]
        )

    lines = [
        f"# {run_name} (hyperparameter sweep)",
        "",
        "## Overview",
        f"- objective: `{objective}` ({direction})",
        f"- trial settings: `{len(ranked_results)}`",
        f"- seeds: `{seeds if seeds else 'single run per setting'}`",
        f"- resumed/skipped pipeline runs: `{runs_skipped}`",
        f"- aggregate JSON: `{output_json_path}`",
        f"- leaderboard CSV: `{output_csv_path}`",
        "",
        "## Leaderboard",
        "\n".join(
            md_table(
                ["rank", "trial_id", "suffix", "objective", "overrides", "summary"],
                leaderboard_rows,
            )
        ),
        "",
    ]

    best = ranked_results[0] if ranked_results else None
    if best is not None:
        flat = extract_metric(best.summary, objective)
        headline_rows: list[list[object]] = []
        for metric in _HEADLINE_METRICS:
            value = extract_metric(best.summary, metric)
            if value is not None:
                headline_rows.append([metric, fmt_float(value)])
        if headline_rows:
            lines.extend(
                [
                    "## Best trial headline metrics",
                    f"- trial: `#{best.plan.trial_id}` ({best.plan.run_name_suffix})",
                    f"- ranking objective: `{objective}` = {_objective_display(best)}",
                    "\n".join(md_table(["metric", "value"], headline_rows)),
                    "",
                ]
            )
        elif flat is not None:
            lines.extend(
                [
                    "## Best trial",
                    f"- trial: `#{best.plan.trial_id}` ({best.plan.run_name_suffix})",
                    f"- `{objective}` = {_objective_display(best)}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Notes",
            "- Trials are ranked by the mean objective across seeds when `--seeds` / sweep `seeds` is set.",
            "- Use `--resume` to skip pipeline runs whose summary JSON already exists.",
            "- Open per-trial `outputs/reports/*_report.md` files for full benchmark and analysis artifacts.",
            "",
        ]
    )
    write_text_file(output_path, "\n".join(lines) + "\n")


def _trial_payload(result: SweepTrialResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial_id": result.plan.trial_id,
        "run_name_suffix": result.plan.run_name_suffix,
        "overrides": result.plan.overrides,
        "objective_value": result.objective_value,
        "objective_std": result.objective_std,
        "objective_n": result.objective_n,
        "summary_path": result.summary_path,
        "runs_skipped": result.runs_skipped,
    }
    if result.seed_results:
        payload["seed_runs"] = [
            {
                "seed": seed_result.seed,
                "objective_value": seed_result.objective_value,
                "summary_path": seed_result.summary_path,
                "skipped": seed_result.skipped,
            }
            for seed_result in result.seed_results
        ]
    return payload


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
        "--seeds",
        default=None,
        help='Repeat each trial setting across seeds ("a,b,c" or "a-b") and rank by mean objective.',
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training_episodes for every trial (useful for smoke tests).",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Override output root for all trial and aggregate artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the trial plan and exit without running the pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pipeline runs when the trial summary JSON already exists.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress end-of-run CLI banners (e.g. unit tests)."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_config, spec = _resolve_sweep_inputs(args)

    if args.outputs_dir:
        base_config = _apply_outputs_dir(base_config, args.outputs_dir)

    if args.episodes is not None:
        base_config = apply_short_run_episodes(base_config, args.episodes)

    seeds = _resolve_seed_list(args, spec)
    if seeds is None:
        seed = spec.seed if spec.seed is not None else args.seed
        if seed is not None:
            base_config = base_config.clone(seed=seed)

    try:
        plans = build_trial_plans(grid=spec.grid, explicit_trials=list(spec.trials or ()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_run_name = str(args.run_name or base_config.run_name)

    if args.dry_run:
        print(
            format_sweep_plan_preview(
                base_run_name=base_run_name,
                plans=plans,
                seeds=seeds,
                objective=spec.objective,
                direction=spec.direction,
            )
        )
        return

    sweep_run_name = f"{base_run_name}_sweep"

    results: list[SweepTrialResult] = []
    for plan in plans:
        results.append(
            _execute_trial(
                base_config=base_config,
                plan=plan,
                base_run_name=base_run_name,
                objective=spec.objective,
                seeds=seeds,
                resume=args.resume,
                quiet=args.quiet,
            )
        )

    ranked = rank_trials(results, objective=spec.objective, direction=spec.direction)
    runs_skipped = sum(result.runs_skipped for result in results)

    output_json_path = f"{base_config.benchmark_dir}/{sweep_run_name}_summary.json"
    output_md_path = f"{base_config.report_dir}/{sweep_run_name}_report.md"
    output_csv_path = f"{base_config.report_dir}/{sweep_run_name}_leaderboard.csv"

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
            "resume": bool(args.resume),
            **({"seeds": seeds} if seeds else {}),
        },
        "trials": [_trial_payload(result) for result in results],
        "leaderboard": [
            {
                "rank": rank,
                "trial_id": result.plan.trial_id,
                "run_name_suffix": result.plan.run_name_suffix,
                "objective_value": result.objective_value,
                "objective_std": result.objective_std,
                "objective_n": result.objective_n,
                "summary_path": result.summary_path,
            }
            for rank, result in enumerate(ranked, start=1)
        ],
        "artifacts": {
            "report": output_md_path,
            "leaderboard_csv": output_csv_path,
            "per_trial_summaries": [result.summary_path for result in results],
        },
    }

    sweep_stats: dict[str, dict[str, float | int]] = {}
    for result in results:
        if result.objective_value is None:
            continue
        sweep_stats[f"trial_{result.plan.trial_id:03d}.{spec.objective}"] = {
            "mean": result.objective_value,
            "std": float(result.objective_std or 0.0),
            "n": int(result.objective_n),
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
    _write_sweep_csv(
        ranked_results=ranked,
        output_path=output_csv_path,
        objective=spec.objective,
    )
    _write_sweep_report(
        run_name=sweep_run_name,
        objective=spec.objective,
        direction=spec.direction,
        ranked_results=ranked,
        output_path=output_md_path,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        seeds=seeds,
        runs_skipped=runs_skipped,
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
        if runs_skipped:
            print(f"[sweep] resumed {runs_skipped} existing trial summary file(s)")


if __name__ == "__main__":
    main()
