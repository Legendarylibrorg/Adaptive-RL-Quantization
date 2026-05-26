"""Hash-chained replay manifests for deterministic experiment audit and re-verification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptive_quant.checkpoint_integrity import (
    INTEGRITY_FIELD,
    attach_dict_integrity,
    verify_dict_integrity,
)
from adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import load_jsonl, read_json, to_jsonable, write_json
from adaptive_quant.trainer_utils import feedback_vector, zero_previous_action
from adaptive_quant.types import QuantizationDecision, QuantMode

MANIFEST_VERSION = 1
MANIFEST_SCHEMA = "adaptive_quant.replay_manifest/v1"

# Keys that do not affect environment dynamics or step outcomes.
_CONFIG_FINGERPRINT_EXCLUDE = frozenset(
    {
        "resume_from_checkpoint",
        "write_research_report",
        "write_training_history",
        "replay_verify_after_run",
        "replay_manifest_enabled",
        "jsonl_integrity_chain",
        "jsonl_buffered",
        "jsonl_flush_every",
        "log_every_n_episodes",
        "run_name",
        "outputs_dir",
        "log_dir",
        "benchmark_dir",
        "analysis_dir",
        "checkpoint_dir",
        "report_dir",
        "training_host_label",
        "external_quality_path",
        "llama_cpp_binary",
        "llama_cpp_model",
    }
)

_INTEGRITY_META_KEYS = frozenset({"_integrity_hash", "_integrity_prev", INTEGRITY_FIELD})


def sha256_canonical(payload: Any) -> str:
    import json

    safe = to_jsonable(payload)
    body = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _require_simulator_replay(config: FrameworkConfig) -> None:
    if config.backend.strip().lower() != "simulator":
        raise ValueError(
            "Hash replay verification is only supported for backend='simulator'; "
            f"got {config.backend!r}."
        )


def _require_full_episode_logging(config: FrameworkConfig) -> None:
    if int(config.log_every_n_episodes) != 1:
        raise ValueError(
            "Replay manifests require log_every_n_episodes=1 "
            f"(got {config.log_every_n_episodes})."
        )


def _integrity_chain_required(
    config: FrameworkConfig | None, manifest: Mapping[str, Any] | None = None
) -> bool:
    if manifest is not None and str(manifest.get("jsonl_tail_integrity_sha256") or ""):
        return True
    if config is not None:
        return bool(config.jsonl_integrity_chain)
    return False


def assert_replay_verified(report: dict[str, Any] | None, config: FrameworkConfig) -> None:
    if not config.replay_verify_after_run or report is None:
        return
    for block_name in ("jsonl_verify", "replay_verify"):
        block = report.get(block_name)
        if not isinstance(block, dict):
            raise RuntimeError(
                f"Replay verification enabled but {block_name!r} is missing from replay report."
            )
        if not bool(block.get("verified")):
            mismatches = block.get("mismatches") or []
            raise RuntimeError(
                f"Replay verification failed ({block_name}): "
                f"{mismatches[:5]}"
            )


def strip_integrity_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _INTEGRITY_META_KEYS}


def config_fingerprint(config: FrameworkConfig) -> str:
    flat = config_to_flat_dict(config)
    body = {key: flat[key] for key in sorted(flat) if key not in _CONFIG_FINGERPRINT_EXCLUDE}
    return sha256_canonical(body)


def observation_fingerprint(record: Mapping[str, Any]) -> str:
    body = {
        "episode": record.get("episode"),
        "phase": record.get("phase"),
        "prompt_id": record.get("prompt_id"),
        "hardware_mode": record.get("hardware_mode"),
        "prompt_domain": record.get("prompt_domain"),
        "input_features": record.get("input_features"),
        "sensitivity": record.get("sensitivity"),
        "previous_action": record.get("previous_action"),
        "moe_context": record.get("moe_context"),
    }
    return sha256_canonical(body)


def outcome_fingerprint(record: Mapping[str, Any]) -> str:
    metrics = record.get("metrics") or {}
    body = {
        "decision": record.get("decision"),
        "metrics": {
            "latency_ms": metrics.get("latency_ms"),
            "throughput_tps": metrics.get("throughput_tps"),
            "perplexity": metrics.get("perplexity"),
            "memory_mb": metrics.get("memory_mb"),
            "stability_penalty": metrics.get("stability_penalty"),
            "reward": metrics.get("reward"),
            "tokens_processed": metrics.get("tokens_processed"),
            "latency_ms_per_token": metrics.get("latency_ms_per_token"),
            "swap_cost_ms": metrics.get("swap_cost_ms"),
            "cache_miss_count": metrics.get("cache_miss_count"),
            "variant_churn": metrics.get("variant_churn"),
        },
    }
    return sha256_canonical(body)


def step_fingerprint(record: Mapping[str, Any]) -> str:
    obs = observation_fingerprint(record)
    out = outcome_fingerprint(record)
    return sha256_canonical({"observation_sha256": obs, "outcome_sha256": out})


def chain_step_hash(previous_chain: str, step_hash: str) -> str:
    return hashlib.sha256(f"{previous_chain}:{step_hash}".encode()).hexdigest()


def decision_from_logged(payload: Mapping[str, Any]) -> QuantizationDecision:
    if not isinstance(payload, Mapping):
        raise TypeError("decision payload must be a mapping")
    mode_raw = payload.get("mode")
    if mode_raw is None:
        raise ValueError("decision payload missing mode")
    return QuantizationDecision(
        mode=QuantMode(str(mode_raw)),
        base_bit_width=payload.get("base_bit_width"),
        group_bit_widths=list(payload.get("group_bit_widths") or []),
        layer_bit_widths=list(payload.get("layer_bit_widths") or []),
        scale_factor=float(payload.get("scale_factor", 1.0)),
        clipping_range=float(payload.get("clipping_range", 1.0)),
        precision_level=float(payload.get("precision_level", 0.5)),
        effective_layer_bits=list(payload.get("effective_layer_bits") or []),
        moe_variant_indices=list(payload.get("moe_variant_indices") or []),
        moe_variant_names=list(payload.get("moe_variant_names") or []),
        fallback_applied=bool(payload.get("fallback_applied", False)),
        unstable=bool(payload.get("unstable", False)),
        metadata=dict(payload.get("metadata") or {}),
    )


def build_manifest_steps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    chain = ""
    for index, record in enumerate(records):
        clean = strip_integrity_meta(record)
        step_hash = step_fingerprint(clean)
        chain = chain_step_hash(chain, step_hash)
        steps.append(
            {
                "index": index,
                "episode": clean.get("episode"),
                "phase": clean.get("phase"),
                "observation_sha256": observation_fingerprint(clean),
                "outcome_sha256": outcome_fingerprint(clean),
                "step_sha256": step_hash,
                "chain_sha256": chain,
                "previous_action": clean.get("previous_action"),
                "decision": clean.get("decision"),
            }
        )
    return steps


def build_manifest_payload(
    config: FrameworkConfig,
    records: list[dict[str, Any]],
    *,
    git_commit: str | None = None,
    log_path: str | None = None,
) -> dict[str, Any]:
    steps = build_manifest_steps(records)
    chain_head = steps[-1]["chain_sha256"] if steps else ""
    jsonl_tail = ""
    if records:
        last = records[-1]
        jsonl_tail = str(last.get("_integrity_hash") or "")
    body: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "schema": MANIFEST_SCHEMA,
        "run_name": config.run_name,
        "config_sha256": config_fingerprint(config),
        "step_count": len(steps),
        "chain_head_sha256": chain_head,
        "jsonl_tail_integrity_sha256": jsonl_tail,
        "git_commit": git_commit,
        "log_path": log_path,
        "steps": steps,
    }
    return attach_dict_integrity(body)


def write_replay_manifest(
    config: FrameworkConfig,
    records: list[dict[str, Any]],
    path: str | Path | None = None,
    *,
    git_commit: str | None = None,
    log_path: str | None = None,
) -> str:
    target = Path(path or config.replay_manifest_path())
    payload = build_manifest_payload(
        config, records, git_commit=git_commit, log_path=log_path
    )
    write_json(target, payload)
    return str(target)


def load_replay_manifest(path: str | Path) -> dict[str, Any]:
    payload = read_json(path, label="Replay manifest")
    verify_dict_integrity(payload, label="Replay manifest")
    if int(payload.get("manifest_version", 0)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported replay manifest version in {path}: {payload.get('manifest_version')!r}"
        )
    schema = str(payload.get("schema") or "")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported replay manifest schema in {path}: {schema!r} "
            f"(expected {MANIFEST_SCHEMA!r})."
        )
    return payload


def verify_jsonl_against_manifest(
    jsonl_path: str | Path,
    manifest_path: str | Path,
    *,
    config: FrameworkConfig | None = None,
    require_integrity_chain: bool = True,
) -> dict[str, Any]:
    manifest = load_replay_manifest(manifest_path)
    chain_required = require_integrity_chain and (
        _integrity_chain_required(config, manifest)
        if config is not None
        else bool(manifest.get("jsonl_tail_integrity_sha256"))
    )
    records = load_jsonl(str(jsonl_path), require_integrity_chain=chain_required)
    expected_config = str(manifest["config_sha256"])
    if config is not None:
        actual_config = config_fingerprint(config)
        if actual_config != expected_config:
            raise ValueError(
                "Config fingerprint mismatch: manifest was built with a different experiment "
                f"contract (expected {expected_config!r}, got {actual_config!r})."
            )
    recomputed = build_manifest_steps(records)
    manifest_steps = manifest.get("steps") or []
    mismatches: list[dict[str, Any]] = []
    if len(recomputed) != len(manifest_steps):
        mismatches.append(
            {
                "kind": "step_count",
                "expected": len(manifest_steps),
                "actual": len(recomputed),
            }
        )
    for idx, (expected, actual) in enumerate(zip(manifest_steps, recomputed, strict=False)):
        for field in ("step_sha256", "chain_sha256", "observation_sha256", "outcome_sha256"):
            if str(expected.get(field)) != str(actual.get(field)):
                mismatches.append(
                    {
                        "kind": "hash_mismatch",
                        "index": idx,
                        "field": field,
                        "expected": expected.get(field),
                        "actual": actual.get(field),
                    }
                )
    tail_expected = str(manifest.get("jsonl_tail_integrity_sha256") or "")
    tail_actual = str(records[-1].get("_integrity_hash") or "") if records else ""
    if tail_expected and tail_actual != tail_expected:
        mismatches.append(
            {
                "kind": "jsonl_tail",
                "expected": tail_expected,
                "actual": tail_actual,
            }
        )
    chain_expected = str(manifest.get("chain_head_sha256") or "")
    chain_actual = recomputed[-1]["chain_sha256"] if recomputed else ""
    if chain_expected != chain_actual:
        mismatches.append(
            {
                "kind": "chain_head",
                "expected": chain_expected,
                "actual": chain_actual,
            }
        )
    return {
        "verified": not mismatches,
        "step_count": len(records),
        "mismatches": mismatches,
        "config_sha256": expected_config,
        "chain_head_sha256": chain_actual,
    }


def replay_manifest_steps(
    config: FrameworkConfig,
    manifest: Mapping[str, Any],
    *,
    log_path: str | None = None,
) -> dict[str, Any]:
    _require_simulator_replay(config)
    expected_config = str(manifest["config_sha256"])
    actual_config = config_fingerprint(config)
    if actual_config != expected_config:
        raise ValueError(
            "Cannot replay: config fingerprint does not match manifest "
            f"(expected {expected_config!r}, got {actual_config!r})."
        )
    env = AdaptiveQuantizationEnv(
        config,
        log_path=log_path,
        enable_logging=log_path is not None,
    )
    mismatches: list[dict[str, Any]] = []
    steps = list(manifest.get("steps") or [])
    previous_action = zero_previous_action()
    try:
        for step in steps:
            episode = step.get("episode")
            phase = str(step.get("phase") or "train")
            ep_index = int(episode) if episode is not None else None
            logged_previous = step.get("previous_action")
            if logged_previous is not None and list(logged_previous) != previous_action:
                mismatches.append(
                    {
                        "kind": "previous_action_chain",
                        "episode": episode,
                        "phase": phase,
                        "expected": previous_action,
                        "actual": logged_previous,
                    }
                )
            env.reset(previous_action=previous_action, phase=phase, episode_index=ep_index)
            decision_payload = step.get("decision")
            if not isinstance(decision_payload, Mapping):
                mismatches.append(
                    {
                        "kind": "missing_decision",
                        "episode": episode,
                        "phase": phase,
                    }
                )
                continue
            raw_decision = decision_from_logged(decision_payload)
            result = env.evaluate_current(raw_decision, episode_index=ep_index, log_episode=False)
            record = {
                "episode": episode,
                "phase": phase,
                "prompt_id": result.state.prompt.prompt_id,
                "hardware_mode": result.state.hardware_profile.hardware_type.value,
                "prompt_domain": result.state.prompt.domain,
                "input_features": to_jsonable(result.state.input_features),
                "sensitivity": to_jsonable(result.state.sensitivity),
                "previous_action": to_jsonable(result.state.previous_action),
                "moe_context": to_jsonable(result.state.moe_context),
                "decision": to_jsonable(result.decision),
                "metrics": to_jsonable(result.metrics),
            }
            expected_step = str(step.get("step_sha256"))
            actual_step = step_fingerprint(record)
            if expected_step != actual_step:
                mismatches.append(
                    {
                        "kind": "replay_step_hash",
                        "episode": episode,
                        "phase": phase,
                        "expected": expected_step,
                        "actual": actual_step,
                    }
                )
            previous_action = feedback_vector(
                result.decision,
                max_bits=max(config.discrete_bit_widths),
                scale_upper=config.scale_bounds[1],
                clip_upper=config.clip_bounds[1],
            )
    finally:
        env.logger.close()
    return {
        "replayed": len(steps),
        "verified": not mismatches,
        "mismatches": mismatches,
    }


def finalize_replay_artifacts(
    config: FrameworkConfig,
    log_path: str | Path,
    *,
    git_commit: str | None = None,
) -> dict[str, Any] | None:
    if not config.replay_manifest_enabled:
        return None
    _require_simulator_replay(config)
    _require_full_episode_logging(config)
    path = Path(log_path)
    if not path.is_file():
        return {"manifest_path": None, "reason": "log_missing"}
    records = load_jsonl(
        str(path),
        require_integrity_chain=bool(config.jsonl_integrity_chain),
    )
    if config.jsonl_integrity_chain and records and not records[-1].get("_integrity_hash"):
        raise ValueError(
            "jsonl_integrity_chain is enabled but the primary log has no _integrity_hash; "
            "flush the logger before building the replay manifest."
        )
    manifest_path = write_replay_manifest(
        config,
        records,
        git_commit=git_commit,
        log_path=str(path),
    )
    report: dict[str, Any] = {
        "manifest_path": manifest_path,
        "step_count": len(records),
        "config_sha256": config_fingerprint(config),
        "chain_head_sha256": build_manifest_steps(records)[-1]["chain_sha256"]
        if records
        else "",
    }
    if config.replay_verify_after_run:
        manifest = load_replay_manifest(manifest_path)
        report["jsonl_verify"] = verify_jsonl_against_manifest(
            path,
            manifest_path,
            config=config,
            require_integrity_chain=True,
        )
        report["replay_verify"] = replay_manifest_steps(config, manifest)
        assert_replay_verified(report, config)
    return report


def replay_from_manifest_file(
    config: FrameworkConfig,
    manifest_path: str | Path,
    *,
    verify_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    _require_simulator_replay(config)
    manifest = load_replay_manifest(manifest_path)
    result: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "replay": replay_manifest_steps(config, manifest),
    }
    if verify_jsonl is not None:
        result["jsonl_verify"] = verify_jsonl_against_manifest(
            verify_jsonl,
            manifest_path,
            config=config,
            require_integrity_chain=True,
        )
    return result
