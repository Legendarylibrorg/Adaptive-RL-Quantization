"""Entrypoint for the **route learning** workflow.

Subcommands manage a JSON :class:`adaptive_quant.model_routes.RouteCatalog`, fetch entries
through ``huggingface-cli`` / ``hf``, and train / query a :class:`adaptive_quant.route_policy.RouteBandit`
that learns the best (model, quant) route per (hardware, task domain, complexity) bucket.

Quick reference (``adaptive-rl-quant-route <subcommand> --help`` for details):

* ``seed``      — write a default catalog with curated GGUF entries.
* ``list``      — print the catalog as a table or JSON.
* ``register``  — add or replace a route in the catalog.
* ``remove``    — drop a route from the catalog.
* ``download``  — fetch a route's GGUF via Hugging Face CLI (validated argv only).
* ``train``     — bandit train across simulated rewards from the existing framework backend.
* ``recommend`` — print the bandit's best route for a given (hardware, task) context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from adaptive_quant.cli.common import add_config_file_argument, load_config_or_fallback
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import hf_allowed_repos_from_env
from adaptive_quant.huggingface_cli import (
    build_download_command,
    find_huggingface_cli,
    require_huggingface_cli,
    run_download,
)
from adaptive_quant.logging_utils import md_table
from adaptive_quant.model_routes import (
    QUANT_BITS,
    ModelRoute,
    QuantSpec,
    RouteCatalog,
    default_route_catalog,
)
from adaptive_quant.route_pipeline import (
    evaluate_route_bandit,
    load_bandit_artifact,
    make_bandit,
    recommend_route,
    save_bandit_artifacts,
    train_route_bandit,
    validate_local_route_models,
)
from adaptive_quant.types import HardwareType

DEFAULT_CATALOG_PATH = "outputs/routes/catalog.json"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="adaptive-rl-quant-route",
        description=(
            "Manage a Hugging Face GGUF route catalog and learn the best route per task / "
            "hardware via a contextual UCB bandit on the existing simulator + reward stack."
        ),
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        metavar="PATH",
        help=f"Path to the route catalog JSON. Default: {DEFAULT_CATALOG_PATH}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seed_parser = sub.add_parser("seed", help="Write a default catalog with curated GGUF entries.")
    seed_parser.add_argument(
        "--force", action="store_true", help="Overwrite the catalog if it already exists."
    )

    list_parser = sub.add_parser("list", help="Print the catalog.")
    list_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Default: table.",
    )

    register_parser = sub.add_parser("register", help="Add or replace a route in the catalog.")
    register_parser.add_argument("--route-id", required=True, help="Stable route identifier.")
    register_parser.add_argument(
        "--repo", required=True, help="Hugging Face '<org>/<name>' repo id."
    )
    register_parser.add_argument(
        "--quant", required=True, help="Quant label, e.g. Q4_K_M, Q8_0, F16."
    )
    register_parser.add_argument(
        "--filename", default=None, help="Specific GGUF filename inside the repo."
    )
    register_parser.add_argument(
        "--revision", default=None, help="Optional Hub revision (branch / tag / sha)."
    )
    register_parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help="Override effective bits per weight (for novel quants not in the built-in table).",
    )
    register_parser.add_argument(
        "--parameters-b", type=float, default=None, help="Parameter count in billions."
    )
    register_parser.add_argument(
        "--size-mb", type=float, default=None, help="On-disk file size in MB."
    )
    register_parser.add_argument(
        "--local-path",
        default=None,
        help="Path to an already-downloaded local GGUF file for real llama.cpp route research.",
    )
    register_parser.add_argument(
        "--hardware-hint",
        action="append",
        default=None,
        help="Hardware affinity hint (gpu/cpu/low_resource/any). Pass repeatedly for multiple.",
    )
    register_parser.add_argument(
        "--domain-hint",
        action="append",
        default=None,
        help="Optional task domain hint (e.g. code, qa). Pass repeatedly for multiple.",
    )
    register_parser.add_argument("--notes", default="", help="Free-text note for the catalog row.")
    register_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing route with the same route_id.",
    )

    remove_parser = sub.add_parser("remove", help="Remove a route from the catalog.")
    remove_parser.add_argument("--route-id", required=True, help="Identifier of the route to drop.")

    download_parser = sub.add_parser("download", help="Fetch a route via huggingface-cli / hf.")
    download_parser.add_argument("--route-id", required=True, help="Catalog route to download.")
    download_parser.add_argument(
        "--local-dir",
        default=None,
        help="Destination directory (defaults to outputs/models/<route_id>).",
    )
    download_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Subprocess timeout in seconds. Default: 600.",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved argv without spawning huggingface-cli.",
    )

    train_parser = sub.add_parser("train", help="Train the route bandit and persist artifacts.")
    add_config_file_argument(train_parser, help_suffix="Otherwise uses config.py defaults.")
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=512,
        help="Number of bandit pulls per train invocation. Default: 512.",
    )
    train_parser.add_argument(
        "--ucb-c",
        type=float,
        default=1.5,
        help="UCB exploration coefficient. Smaller => greedier. Default: 1.5.",
    )
    train_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After training, sweep all (prompt, hardware) pairs greedily and store an evaluation block.",
    )
    train_parser.add_argument(
        "--evaluation-sweeps",
        type=int,
        default=1,
        help="How many sweeps the post-training evaluation performs (only used with --evaluate). Default: 1.",
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a previously written *_route_bandit.json to warm-start from.",
    )
    train_parser.add_argument(
        "--require-local-models",
        action="store_true",
        help="Require every catalog route to have an existing local GGUF path before training.",
    )

    recommend_parser = sub.add_parser("recommend", help="Print the best route for a context.")
    add_config_file_argument(recommend_parser)
    recommend_parser.add_argument(
        "--bandit",
        default=None,
        metavar="PATH",
        help="Path to a saved *_route_bandit.json. Defaults to the run's benchmark artifact.",
    )
    recommend_parser.add_argument(
        "--prompt-id",
        default=None,
        help="Prompt id from the bundled prompt library.",
    )
    recommend_parser.add_argument(
        "--prompt-text",
        default=None,
        help="Free-form prompt text to derive features from (overrides --prompt-id).",
    )
    recommend_parser.add_argument(
        "--domain",
        default="online",
        help="Prompt domain when using --prompt-text. Default: online.",
    )
    recommend_parser.add_argument(
        "--hardware",
        choices=tuple(hw.value for hw in HardwareType),
        default=HardwareType.GPU.value,
        help="Target hardware. Default: gpu.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    catalog_path = Path(args.catalog)

    if args.command == "seed":
        _cmd_seed(catalog_path, force=args.force)
    elif args.command == "list":
        _cmd_list(catalog_path, output_format=args.format)
    elif args.command == "register":
        _cmd_register(catalog_path, args)
    elif args.command == "remove":
        _cmd_remove(catalog_path, args)
    elif args.command == "download":
        _cmd_download(catalog_path, args)
    elif args.command == "train":
        _cmd_train(catalog_path, args)
    elif args.command == "recommend":
        _cmd_recommend(args)
    else:  # pragma: no cover - argparse enforces choices.
        raise SystemExit(f"Unknown subcommand: {args.command}")


def _cmd_seed(catalog_path: Path, *, force: bool) -> None:
    if catalog_path.exists() and not force:
        raise SystemExit(
            f"Catalog already exists at {catalog_path}. Pass --force to overwrite, or pick a new --catalog path."
        )
    catalog = default_route_catalog()
    catalog.save(str(catalog_path))
    print(f"Wrote default catalog ({len(catalog)} routes) to {catalog_path}")


def _cmd_list(catalog_path: Path, *, output_format: str) -> None:
    catalog = _load_catalog(catalog_path, allow_missing=True)
    if not catalog.routes:
        print(f"(empty catalog at {catalog_path}; run 'seed' or 'register' to populate it)")
        return
    if output_format == "json":
        json.dump(catalog.to_dict(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    rows: list[list[object]] = [
        [
            route.route_id,
            route.repo_id,
            route.quant_label,
            f"{route.effective_bits:.2f}",
            "-" if route.parameters_b is None else f"{route.parameters_b:g}B",
            "-" if route.size_mb is None else f"{route.size_mb:g}MB",
            ",".join(route.hardware_hints),
        ]
        for route in catalog.routes
    ]
    headers = ["route_id", "repo_id", "quant", "bpw", "params", "size", "hw_hints"]
    for line in md_table(headers, rows):
        print(line)


def _cmd_register(catalog_path: Path, args: argparse.Namespace) -> None:
    catalog = _load_catalog(catalog_path, allow_missing=True)
    quant_label = args.quant.strip().upper()
    if args.effective_bits is None and quant_label not in QUANT_BITS:
        raise SystemExit(
            f"Unknown quant label {quant_label!r}. Pass --effective-bits to register a novel quant. "
            f"Built-in labels: {sorted(QUANT_BITS)}"
        )

    route = ModelRoute(
        route_id=args.route_id,
        repo_id=args.repo,
        quant_label=quant_label,
        filename=args.filename,
        revision=args.revision,
        effective_bits=(
            float(args.effective_bits)
            if args.effective_bits is not None
            else QuantSpec.from_label(quant_label).effective_bits
        ),
        parameters_b=args.parameters_b,
        size_mb=args.size_mb,
        hardware_hints=tuple(args.hardware_hint) if args.hardware_hint else ("any",),
        domain_hints=tuple(args.domain_hint) if args.domain_hint else (),
        notes=args.notes,
        local_path=str(Path(args.local_path)) if args.local_path else None,
    )
    catalog.add(route, replace_existing=args.replace)
    catalog.save(str(catalog_path))
    # Keep stdout stable for scripting (JSON / table output). Emit human-status only when interactive
    # or explicitly requested (keeps unittest/CI quiet).
    if sys.stderr.isatty() or os.environ.get("ADAPTIVE_RL_QUANT_VERBOSE", "").strip() == "1":
        print(f"Registered route {route.route_id!r} → {catalog_path}", file=sys.stderr)


def _cmd_remove(catalog_path: Path, args: argparse.Namespace) -> None:
    catalog = _load_catalog(catalog_path)
    if not catalog.remove(args.route_id):
        raise SystemExit(f"No route with route_id={args.route_id!r} in {catalog_path}")
    catalog.save(str(catalog_path))
    print(f"Removed route {args.route_id!r} from {catalog_path}")


def _cmd_download(catalog_path: Path, args: argparse.Namespace) -> None:
    catalog = _load_catalog(catalog_path)
    route = catalog.by_id(args.route_id)
    cli = find_huggingface_cli() if args.dry_run else require_huggingface_cli()
    if cli is None:
        # dry-run with no CLI on PATH: synthesize a placeholder argv so the user can preview.
        from adaptive_quant.huggingface_cli import HuggingFaceCli

        cli = HuggingFaceCli(binary="hf", dialect="hf")

    local_dir = Path(args.local_dir) if args.local_dir else Path("outputs/models") / route.route_id
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    allowed_repos = tuple(hf_allowed_repos_from_env())
    if args.dry_run:
        argv = build_download_command(
            cli,
            repo_id=route.repo_id,
            filename=route.filename,
            revision=route.revision,
            local_dir=local_dir,
            allowed_repos=allowed_repos,
        )
        print("DRY RUN — would execute:")
        print("  " + " ".join(json.dumps(part) for part in argv))
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {route.repo_id} ({route.filename or 'all files'}) → {local_dir}")
    result = run_download(
        cli,
        repo_id=route.repo_id,
        filename=route.filename,
        revision=route.revision,
        local_dir=local_dir,
        allowed_repos=allowed_repos,
        timeout_s=float(args.timeout),
    )
    if result.timed_out:
        raise SystemExit(f"Download timed out after {args.timeout}s. argv={result.command}")
    if not result.ok:
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"huggingface-cli failed with exit code {result.returncode}. argv={result.command}"
        )

    resolved = result.local_path
    if resolved is None:
        # Fall back to the requested directory (download succeeded; CLI did not echo a path).
        resolved = local_dir
    catalog.update_local_path(route.route_id, str(resolved))
    catalog.save(str(catalog_path))
    print(f"Downloaded route {route.route_id!r} → {resolved}")


def _cmd_train(catalog_path: Path, args: argparse.Namespace) -> None:
    catalog = _load_catalog(catalog_path)
    config = _resolve_config(args.config)
    if config.backend == "llama_cpp" or args.require_local_models:
        try:
            validate_local_route_models(catalog)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    bandit = make_bandit(catalog, config, ucb_c=float(args.ucb_c))
    if args.resume is not None:
        _, resumed = load_bandit_artifact(args.resume)
        bandit.load_state_dict(resumed.state_dict())

    bandit, summary = train_route_bandit(
        config,
        catalog=catalog,
        iterations=int(args.iterations),
        bandit=bandit,
    )

    evaluation = None
    if args.evaluate:
        evaluation = evaluate_route_bandit(
            config,
            catalog=catalog,
            bandit=bandit,
            sweeps=int(args.evaluation_sweeps),
        )

    artifacts = save_bandit_artifacts(
        config=config,
        catalog=catalog,
        bandit=bandit,
        training_summary=summary,
        evaluation=evaluation,
    )
    print(
        f"Trained {summary.pulls} pulls, mean reward {summary.mean_reward:.3f} "
        f"(explore_rate={summary.explore_rate:.2f})"
    )
    if summary.final_recommendation is not None:
        rec = summary.final_recommendation
        print(
            "Final greedy recommendation @ "
            f"{rec['context_key']}: {rec['route_id']} "
            f"({rec['repo_id']} / {rec['quant_label']}, score={rec['score']:.3f})"
        )
    print(f"Bandit state: {artifacts['bandit']}")
    print(f"Summary:      {artifacts['summary']}")


def _cmd_recommend(args: argparse.Namespace) -> None:
    config = _resolve_config(args.config)
    bandit_path = (
        Path(args.bandit)
        if args.bandit is not None
        else Path(config.benchmark_dir) / f"{config.run_name}_route_bandit.json"
    )
    if not bandit_path.is_file():
        raise SystemExit(
            f"Bandit artifact not found: {bandit_path}. Train the bandit first via "
            "'adaptive-rl-quant-route train' or pass --bandit."
        )
    _saved_catalog, bandit = load_bandit_artifact(bandit_path)
    selection = recommend_route(
        config=config,
        bandit=bandit,
        prompt_id=args.prompt_id,
        prompt_text=args.prompt_text,
        domain=args.domain,
        hardware=HardwareType(args.hardware),
    )
    payload = {
        "context": selection.bucket_key,
        "route_id": selection.route.route_id,
        "repo_id": selection.route.repo_id,
        "quant_label": selection.route.quant_label,
        "effective_bits": selection.route.effective_bits,
        "filename": selection.route.filename,
        "score": selection.score,
        "feasible_for_hardware": selection.feasible,
        "reasoning": selection.reasoning,
        "local_path": selection.route.local_path,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _load_catalog(path: Path, *, allow_missing: bool = False) -> RouteCatalog:
    if not path.exists():
        if allow_missing:
            return RouteCatalog(routes=[])
        raise SystemExit(
            f"Catalog not found at {path}. Run 'adaptive-rl-quant-route seed' or pass --catalog."
        )
    return RouteCatalog.from_file(path)


def _resolve_config(config_path: str | None) -> FrameworkConfig:
    if config_path is not None:
        return load_config_or_fallback(config_path, FrameworkConfig())
    try:
        from adaptive_quant.presets.baseline import CONFIG
    except ImportError:
        return FrameworkConfig()
    if isinstance(CONFIG, FrameworkConfig):
        return CONFIG
    return FrameworkConfig()


if __name__ == "__main__":
    main()
