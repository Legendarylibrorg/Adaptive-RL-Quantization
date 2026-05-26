"""Verify or replay a hash-chained experiment manifest against the simulator."""

from __future__ import annotations

import argparse
import json

from adaptive_quant.cli.common import load_config_or_fallback
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import validate_cli_path_argument
from adaptive_quant.replay_trace import (
    finalize_replay_artifacts,
    replay_from_manifest_file,
    verify_jsonl_against_manifest,
)
from adaptive_quant.security_bypass import enforce_security_bypass_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify hash-chained JSONL logs and optionally re-execute logged decisions "
            "against the simulator for audit/replay."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Experiment config (.json / .toml); must match the manifest config fingerprint.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        metavar="PATH",
        help="Replay manifest path (default: <log_dir>/<run_name>_replay_manifest.json).",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        metavar="PATH",
        help="Episode JSONL to verify (default: <log_dir>/<run_name>.jsonl).",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Build or rebuild the manifest from JSONL without re-executing steps.",
    )
    parser.add_argument(
        "--verify-jsonl-only",
        action="store_true",
        help="Only verify JSONL hashes against the manifest (no simulator replay).",
    )
    args = parser.parse_args()
    enforce_security_bypass_policy(context="replay cli")

    if args.config is None and not args.build_manifest:
        raise SystemExit("--config is required for replay verification (pass the original experiment JSON/TOML).")

    config = load_config_or_fallback(
        args.config,
        FrameworkConfig.reproducible_research(run_name="replay_cli"),
    )
    manifest_path = args.manifest or config.replay_manifest_path()
    jsonl_path = args.jsonl or config.primary_log_path()

    if args.build_manifest:
        validate_cli_path_argument("jsonl", jsonl_path)
        report = finalize_replay_artifacts(
            config,
            jsonl_path,
            git_commit=None,
        )
        print(json.dumps(report or {"reason": "replay_manifest_disabled"}, indent=2))
        return

    validate_cli_path_argument("manifest", manifest_path)
    validate_cli_path_argument("jsonl", jsonl_path)

    if args.verify_jsonl_only:
        report = verify_jsonl_against_manifest(
            jsonl_path,
            manifest_path,
            config=config,
            require_integrity_chain=bool(config.jsonl_integrity_chain),
        )
        print(json.dumps(report, indent=2))
        if not report.get("verified"):
            raise SystemExit(1)
        return

    report = replay_from_manifest_file(
        config,
        manifest_path,
        verify_jsonl=jsonl_path,
    )
    print(json.dumps(report, indent=2))
    replay_ok = bool((report.get("replay") or {}).get("verified"))
    jsonl_ok = bool((report.get("jsonl_verify") or {}).get("verified", True))
    if not replay_ok or not jsonl_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
