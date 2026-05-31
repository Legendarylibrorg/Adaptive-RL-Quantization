from __future__ import annotations

import csv
import hashlib
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adaptive_quant.analysis_utils import flatten_numeric
from adaptive_quant.checkpoint_integrity import sha256_canonical
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import load_jsonl, write_json, write_text_file
from adaptive_quant.math_utils import sample_std


def paper_bundle_dir(config: FrameworkConfig, *, run_name: str | None = None) -> Path:
    return Path(config.outputs_dir) / "paper_bundles" / (run_name or config.run_name)


def create_pipeline_paper_bundle(
    *,
    config: FrameworkConfig,
    summary: Mapping[str, Any],
    telemetry_path: str | None = None,
) -> dict[str, str]:
    bundle_dir = paper_bundle_dir(config)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metrics = flatten_numeric(summary)
    selected_metrics = _select_metrics(metrics, config=config)
    metric_rows = [{"metric": key, "value": value} for key, value in selected_metrics.items()]

    manifest = _manifest(config=config, run_name=config.run_name, summary=summary)
    manifest["metric_sources"] = _metric_sources(config)

    manifest_path = bundle_dir / "manifest.json"
    metrics_json_path = bundle_dir / "metrics_summary.json"
    metrics_csv_path = bundle_dir / "metrics_summary.csv"
    episodes_csv_path = bundle_dir / "episodes.csv"
    claims_json_path = bundle_dir / "claims_validation.json"
    claims_md_path = bundle_dir / "claims_validation.md"
    appendix_path = bundle_dir / "appendix.md"

    write_json(manifest_path, manifest)
    write_json(metrics_json_path, selected_metrics)
    _write_csv(metrics_csv_path, ["metric", "value"], metric_rows)
    episode_count = _write_episode_csv(
        episodes_csv_path,
        telemetry_path
        or _first_existing_path(
            [
                f"{config.log_dir}/{config.run_name}_route_telemetry.jsonl",
                f"{config.log_dir}/{config.run_name}.jsonl",
            ]
        ),
    )

    claims = _claims_validation(config=config, summary=summary, metrics=selected_metrics)
    write_json(claims_json_path, claims)
    write_text_file(claims_md_path, _claims_markdown(claims))
    write_text_file(
        appendix_path,
        _appendix_markdown(
            run_name=config.run_name,
            bundle_dir=bundle_dir,
            artifacts=summary.get("artifacts", {})
            if isinstance(summary.get("artifacts"), Mapping)
            else {},
            episode_count=episode_count,
            claims=claims,
        ),
    )

    return {
        "paper_bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "metrics_summary_json": str(metrics_json_path),
        "metrics_summary_csv": str(metrics_csv_path),
        "episodes_csv": str(episodes_csv_path),
        "appendix": str(appendix_path),
        "claims_validation_json": str(claims_json_path),
        "claims_validation_md": str(claims_md_path),
    }


def create_multiseed_paper_bundle(
    *,
    config: FrameworkConfig,
    run_name: str,
    aggregate_payload: Mapping[str, Any],
    aggregate_stats: Mapping[str, Mapping[str, float | int]],
    report_path: str,
) -> dict[str, str]:
    bundle_dir = paper_bundle_dir(config, run_name=run_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(config=config, run_name=run_name, summary=aggregate_payload)
    manifest["metric_sources"] = _metric_sources(config)
    manifest["seeds"] = list(aggregate_payload.get("seeds", []))

    rows = []
    for metric, stat in sorted(aggregate_stats.items()):
        rows.append(
            {
                "metric": metric,
                "mean": stat.get("mean", 0.0),
                "std": stat.get("std", 0.0),
                "n": stat.get("n", 0),
                "stderr": stat.get("stderr", 0.0),
                "ci95_low": stat.get("ci95_low", 0.0),
                "ci95_high": stat.get("ci95_high", 0.0),
                "effect_size_vs_zero": stat.get("effect_size_vs_zero", 0.0),
            }
        )

    manifest_path = bundle_dir / "manifest.json"
    stats_json_path = bundle_dir / "aggregate_stats.json"
    stats_csv_path = bundle_dir / "aggregate_stats.csv"
    appendix_path = bundle_dir / "appendix.md"
    claims_json_path = bundle_dir / "claims_validation.json"
    claims_md_path = bundle_dir / "claims_validation.md"

    write_json(manifest_path, manifest)
    write_json(stats_json_path, aggregate_stats)
    _write_csv(
        stats_csv_path,
        ["metric", "mean", "std", "n", "stderr", "ci95_low", "ci95_high", "effect_size_vs_zero"],
        rows,
    )
    claims = _claims_validation(
        config=config,
        summary=aggregate_payload,
        metrics={k: v.get("mean", 0.0) for k, v in aggregate_stats.items()},
    )
    write_json(claims_json_path, claims)
    write_text_file(claims_md_path, _claims_markdown(claims))
    write_text_file(
        appendix_path,
        _appendix_markdown(
            run_name=run_name,
            bundle_dir=bundle_dir,
            artifacts={"report": report_path},
            episode_count=0,
            claims=claims,
        ),
    )

    return {
        "paper_bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "aggregate_stats_json": str(stats_json_path),
        "aggregate_stats_csv": str(stats_csv_path),
        "appendix": str(appendix_path),
        "claims_validation_json": str(claims_json_path),
        "claims_validation_md": str(claims_md_path),
    }


def aggregate_values(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "mean": 0.0,
            "std": 0.0,
            "n": 0,
            "stderr": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "effect_size_vs_zero": 0.0,
        }
    mean = sum(finite) / len(finite)
    std = sample_std(finite)
    stderr = std / math.sqrt(len(finite)) if finite else 0.0
    ci = 1.96 * stderr
    effect = mean / std if std > 0.0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": len(finite),
        "stderr": stderr,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "effect_size_vs_zero": effect,
    }


def _select_metrics(metrics: Mapping[str, float], *, config: FrameworkConfig) -> dict[str, float]:
    """
    Pick a small set of high-signal metrics for `metrics_summary.*`.

    Research-grade default behavior:
    - When `backend="llama_cpp"`, reward/perplexity are *mixed* or simulator-derived, so we avoid
      selecting them as headline outputs by default.
    - Simulator runs can include reward/perplexity, since they are consistently defined there.
    """
    base_needles = (
        "mean_latency_ms",
        "mean_throughput_tps",
        "mean_memory_mb",
        "generalization_gap_improvement",
        "single_policy_gap",
        "multi_policy_gap",
    )
    simulator_only_needles = (
        "mean_reward",
        "mean_perplexity",
        "reward_delta",
        "quality_variance_delta",
    )
    needles = (
        base_needles if config.backend == "llama_cpp" else base_needles + simulator_only_needles
    )
    return {
        key: metrics[key]
        for key in sorted(metrics)
        if any(needle in key.lower() for needle in needles)
    }


def _manifest(
    *, config: FrameworkConfig, run_name: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    config_dict = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    llama_binary = config.llama_cpp_binary
    llama_model = config.llama_cpp_model
    external_quality_path = config.external_quality_path
    return {
        "run_name": run_name,
        "created_by": "adaptive-rl-quant",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": summary.get("git_commit"),
        "backend": config.backend,
        "training_backend": config.training_backend,
        "config_digest": _stable_digest(config_dict),
        "llama_cpp": {
            "binary": llama_binary,
            "binary_sha256": _sha256_file(llama_binary),
            "model": llama_model,
            "model_sha256": _sha256_file(llama_model),
            "generate_tokens": config.llama_cpp_generate_tokens,
            "context": config.llama_cpp_context,
            "threads": config.llama_cpp_threads,
        },
        "external_quality": {
            "path": external_quality_path,
            "sha256": _sha256_file(external_quality_path),
            "metric": config.external_quality_metric,
        },
    }


def _metric_sources(config: FrameworkConfig) -> dict[str, str]:
    has_external_quality = bool(config.external_quality_path)
    quality_source = (
        f"external:{config.external_quality_metric or 'perplexity'}"
        if has_external_quality
        else "simulator"
    )
    if config.backend == "llama_cpp":
        return {
            "latency_ms": "llama_cpp",
            "throughput_tps": "llama_cpp",
            "memory_mb": "llama_cpp_when_parseable_else_simulator",
            "perplexity": quality_source,
            "reward": (
                "mixed_llama_cpp_perf_plus_external_quality"
                if has_external_quality
                else "mixed_llama_cpp_perf_plus_simulator_quality"
            ),
        }
    return {
        "latency_ms": "simulator",
        "throughput_tps": "simulator",
        "memory_mb": "simulator",
        "perplexity": quality_source,
        "reward": "simulator_plus_external_quality" if has_external_quality else "simulator",
    }


def _claims_validation(
    *, config: FrameworkConfig, summary: Mapping[str, Any], metrics: Mapping[str, float]
) -> dict[str, Any]:
    evidence_level = "local_llama_cpp" if config.backend == "llama_cpp" else "simulator"
    warnings: list[str] = []
    has_external_quality = bool(config.external_quality_path)
    if config.backend == "llama_cpp":
        if has_external_quality:
            warnings.append(
                "Latency/throughput are locally measured and quality uses an external sidecar, but this is still single-machine evidence."
            )
            warnings.append(
                "Verify the external quality sidecar was generated from real datasets and fixed scoring code before citing quality claims."
            )
        else:
            warnings.append(
                "Latency/throughput are locally measured, but perplexity remains simulator-derived unless an external quality metric is supplied."
            )
        warnings.append(
            "Local results are single-machine evidence, not deployment-grade multi-device validation."
        )
    else:
        if has_external_quality:
            warnings.append(
                "Systems metrics are simulator-backed; only the configured quality metric uses an external sidecar."
            )
        else:
            warnings.append("All headline metrics are simulator-backed.")
    return {
        "evidence_level": evidence_level,
        "deployment_grade": False,
        "external_quality": has_external_quality,
        "external_quality_metric": config.external_quality_metric if has_external_quality else None,
        "metric_count": len(metrics),
        "has_benchmark_summary": isinstance(summary.get("benchmarks"), Mapping),
        "has_evaluation_summary": isinstance(summary.get("evaluation"), Mapping),
        "warnings": warnings,
    }


def _claims_markdown(claims: Mapping[str, Any]) -> str:
    lines = [
        "# Claims Validation",
        "",
        f"- evidence level: `{claims.get('evidence_level')}`",
        f"- deployment grade: `{claims.get('deployment_grade')}`",
        f"- external quality: `{claims.get('external_quality')}`",
        f"- external quality metric: `{claims.get('external_quality_metric')}`",
        f"- metric count: `{claims.get('metric_count')}`",
        "",
        "## Warnings",
    ]
    for warning in claims.get("warnings", []):
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _appendix_markdown(
    *,
    run_name: str,
    bundle_dir: Path,
    artifacts: Mapping[str, Any],
    episode_count: int,
    claims: Mapping[str, Any],
) -> str:
    lines = [
        f"# {run_name} Paper Bundle Appendix",
        "",
        "## Bundle Files",
        "- `manifest.json`",
        "- `metrics_summary.json` / `metrics_summary.csv` when available",
        "- `aggregate_stats.json` / `aggregate_stats.csv` for multi-seed runs",
        "- `episodes.csv` when telemetry exists",
        "- `claims_validation.json` / `claims_validation.md`",
        "",
        "## Linked Artifacts",
    ]
    if artifacts:
        for key, value in sorted(artifacts.items()):
            if value:
                lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No upstream artifacts were recorded.")
    lines.extend(
        [
            "",
            "## Reproducibility Checklist",
            f"- Bundle directory: `{bundle_dir}`",
            f"- Exported episode rows: `{episode_count}`",
            f"- Evidence level: `{claims.get('evidence_level')}`",
            "- Confirm local `llama.cpp` binary and GGUF hashes in `manifest.json` before citing local measurements.",
            "- Treat deployment-grade claims as unresolved until validated on the target device fleet and real prompt distribution.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_episode_csv(path: Path, telemetry_path: str | None) -> int:
    if telemetry_path is None:
        _write_csv(path, ["record_index"], [])
        return 0
    records = load_jsonl(telemetry_path)
    if not records:
        _write_csv(path, ["record_index"], [])
        return 0
    rows = [_flatten_record(record, index=index) for index, record in enumerate(records)]
    headers = sorted({key for row in rows for key in row})
    _write_csv(path, headers, rows)
    return len(rows)


def _flatten_record(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {"record_index": index}

    def walk(node: object, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if isinstance(key, str):
                    walk(value, f"{prefix}.{key}" if prefix else key)
            return
        if isinstance(node, (list, tuple)):
            row[prefix] = ",".join(str(value) for value in node[:16])
            return
        row[prefix] = node

    walk(record, "")
    return row


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _stable_digest(obj: object) -> str:
    return sha256_canonical(obj)


_MAX_DIGEST_FILE_BYTES = 256 << 20


def _sha256_file(path: object | None) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    if source.stat().st_size > _MAX_DIGEST_FILE_BYTES:
        return None
    digest = hashlib.sha256()
    remaining = _MAX_DIGEST_FILE_BYTES
    with source.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _first_existing_path(paths: Sequence[str]) -> str | None:
    for path in paths:
        if Path(path).is_file():
            return path
    return None
