"""Research architecture contract — evidence tiers, learning scope, and claim boundaries.

Every pipeline summary embeds a ``research`` block built here so runs state *what was
trained* (quantization policy), *how metrics were measured*, and *what claims are valid*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from adaptive_quant.configuration import FrameworkConfig

SCHEMA_VERSION = 1

EVIDENCE_SIMULATOR = "simulator"
EVIDENCE_LOCAL_LLAMA_CPP = "local_llama_cpp"
EVIDENCE_MULTISEED = "multiseed_aggregate"
EVIDENCE_SWEEP = "sweep_aggregate"

LEARNING_TARGET_POLICY = "quantization_policy"


def infer_evidence_level(config: FrameworkConfig) -> str:
    if config.backend == "llama_cpp":
        return EVIDENCE_LOCAL_LLAMA_CPP
    return EVIDENCE_SIMULATOR


def metric_sources_for_config(config: FrameworkConfig) -> dict[str, str]:
    """Per-metric provenance labels (shared by summaries, paper bundles, reports)."""
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


def _claim_boundary(config: FrameworkConfig, evidence_level: str) -> dict[str, object]:
    valid: list[str] = []
    invalid: list[str] = [
        "llm_weight_updates",
        "automatic_gguf_requantization",
        "multi_device_deployment_validation",
        "production_serving_sla",
    ]
    if evidence_level == EVIDENCE_SIMULATOR:
        valid.extend(
            [
                "policy_learning_dynamics",
                "benchmark_comparisons_within_simulator",
                "reward_engineering_ablations",
                "reproducible_rl_iteration",
            ]
        )
        invalid.append("real_hardware_latency_claims")
        invalid.append("real_inference_quality_claims_without_external_sidecar")
    elif evidence_level == EVIDENCE_LOCAL_LLAMA_CPP:
        valid.extend(
            [
                "single_machine_llama_cpp_latency_throughput",
                "route_selection_among_prebuilt_gguf",
                "policy_learning_with_local_measurements",
            ]
        )
        if config.external_quality_path:
            valid.append("external_quality_on_configured_sidecar")
        else:
            invalid.append("real_perplexity_claims_without_external_sidecar")
    return {"valid_claims": valid, "invalid_claims": invalid}


def _escalation_path(config: FrameworkConfig, evidence_level: str) -> list[str]:
    hints: list[str] = []
    if evidence_level == EVIDENCE_SIMULATOR:
        hints.append(
            "Escalate to local llama.cpp: set backend='llama_cpp', llama_cpp_binary, "
            "llama_cpp_model, and pre-built GGUF routes (see docs/LOCAL_RESEARCH.md)."
        )
    if not config.external_quality_path:
        hints.append(
            "Add external_quality_path for dataset-grounded quality instead of simulator perplexity."
        )
    hints.append(
        "Run adaptive-rl-quant-multiseed before comparative claims to report mean ± std across seeds."
    )
    if config.backend == "llama_cpp" and not config.router_enabled:
        hints.append(
            "Enable router_enabled with router_routes to compare multiple GGUF quant variants in one run."
        )
    hints.append(
        "Use outputs/paper_bundles/<run>/manifest.json and claims_validation.md when citing results."
    )
    return hints


def build_research_contract(
    config: FrameworkConfig,
    *,
    git_commit: str | None = None,
    pipeline: str = "offline_research",
    phases: Sequence[str] | None = None,
    evidence_level: str | None = None,
) -> dict[str, object]:
    """Machine-readable research scope block for ``*_summary.json`` and reports."""
    level = evidence_level or infer_evidence_level(config)
    sources = metric_sources_for_config(config)
    boundary = _claim_boundary(config, level)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "learning_target": {
            "object": LEARNING_TARGET_POLICY,
            "trained_artifact": "policy_checkpoint",
            "does_not_train": ["llm_weights", "gguf_quantization_export"],
            "summary": (
                "Trains an RL quantization/routing policy; GGUF files are pre-built "
                "measurement inputs, not outputs of in-run weight quantization."
            ),
        },
        "evidence": {
            "level": level,
            "deployment_grade": False,
            "metric_sources": sources,
            "claim_boundary": boundary,
        },
        "measurement": {
            "backend": config.backend,
            "training_backend": config.training_backend,
            "quant_mode": config.quant_mode,
            "moe_enabled": config.moe_enabled,
            "hardware_modes": list(config.hardware_modes),
            "router_enabled": config.router_enabled,
            "router_route_count": len(config.router_routes) if config.router_enabled else 0,
            "llama_cpp_configured": bool(
                config.backend == "llama_cpp" and config.llama_cpp_binary and config.llama_cpp_model
            ),
            "external_quality": bool(config.external_quality_path),
            "external_quality_metric": config.external_quality_metric
            if config.external_quality_path
            else None,
        },
        "reproducibility": {
            "git_commit": git_commit,
            "run_name": config.run_name,
            "seed": config.seed,
            "phases_completed": list(phases or ()),
        },
        "escalation_path": _escalation_path(config, level),
    }


def build_claims_validation(
    *,
    config: FrameworkConfig,
    summary: Mapping[str, Any],
    metrics: Mapping[str, float],
    evidence_level: str | None = None,
) -> dict[str, Any]:
    """Paper-bundle claims block (aligned with ``research`` contract)."""
    level = evidence_level or infer_evidence_level(config)
    warnings: list[str] = []
    has_external_quality = bool(config.external_quality_path)
    if level == EVIDENCE_LOCAL_LLAMA_CPP:
        if has_external_quality:
            warnings.append(
                "Latency/throughput are locally measured and quality uses an external sidecar, "
                "but this is still single-machine evidence."
            )
            warnings.append(
                "Verify the external quality sidecar was generated from real datasets and "
                "fixed scoring code before citing quality claims."
            )
        else:
            warnings.append(
                "Latency/throughput are locally measured, but perplexity remains "
                "simulator-derived unless an external quality metric is supplied."
            )
        warnings.append(
            "Local results are single-machine evidence, not deployment-grade multi-device validation."
        )
    else:
        if has_external_quality:
            warnings.append(
                "Systems metrics are simulator-backed; only the configured quality metric "
                "uses an external sidecar."
            )
        else:
            warnings.append("All headline metrics are simulator-backed.")
    boundary = _claim_boundary(config, level)
    return {
        "evidence_level": level,
        "learning_target": LEARNING_TARGET_POLICY,
        "deployment_grade": False,
        "external_quality": has_external_quality,
        "external_quality_metric": config.external_quality_metric if has_external_quality else None,
        "metric_count": len(metrics),
        "has_benchmark_summary": isinstance(summary.get("benchmarks"), Mapping),
        "has_evaluation_summary": isinstance(summary.get("evaluation"), Mapping),
        "valid_claims": boundary["valid_claims"],
        "invalid_claims": boundary["invalid_claims"],
        "warnings": warnings,
    }


def research_contract_report_lines(contract: Mapping[str, Any]) -> list[str]:
    """Markdown bullets for the Research scope section in reports."""
    learning = contract.get("learning_target")
    evidence = contract.get("evidence")
    measurement = contract.get("measurement")
    lines: list[str] = []
    if isinstance(learning, dict):
        lines.append(
            f"- **Learning target:** `{learning.get('object')}` "
            f"(checkpoint: `{learning.get('trained_artifact')}`)"
        )
        does_not = learning.get("does_not_train")
        if isinstance(does_not, list) and does_not:
            lines.append(f"- **Does not train:** {', '.join(f'`{x}`' for x in does_not)}")
    if isinstance(evidence, dict):
        lines.append(f"- **Evidence level:** `{evidence.get('level')}`")
        sources = evidence.get("metric_sources")
        if isinstance(sources, dict):
            perf = sources.get("latency_ms", "?")
            quality = sources.get("perplexity", "?")
            lines.append(f"- **Metric sources:** latency/throughput `{perf}`; quality `{quality}`")
        boundary = evidence.get("claim_boundary")
        if isinstance(boundary, dict):
            valid = boundary.get("valid_claims")
            if isinstance(valid, list) and valid:
                lines.append(f"- **Valid claims:** {valid[0]}")
    if isinstance(measurement, dict):
        lines.append(
            f"- **Measurement backend:** `{measurement.get('backend')}` "
            f"(training `{measurement.get('training_backend')}`)"
        )
        if measurement.get("router_enabled"):
            lines.append(
                f"- **Router:** `{measurement.get('router_route_count')}` pre-built GGUF route(s)"
            )
    return lines


__all__ = [
    "EVIDENCE_LOCAL_LLAMA_CPP",
    "EVIDENCE_MULTISEED",
    "EVIDENCE_SIMULATOR",
    "EVIDENCE_SWEEP",
    "LEARNING_TARGET_POLICY",
    "SCHEMA_VERSION",
    "build_claims_validation",
    "build_research_contract",
    "infer_evidence_level",
    "metric_sources_for_config",
    "research_contract_report_lines",
]
