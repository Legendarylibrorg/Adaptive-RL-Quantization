"""CLI: online adaptation pipeline."""

from __future__ import annotations

import argparse

from adaptive_quant.cli.common import (
    add_config_file_argument,
    add_config_override_arguments,
    apply_config_overrides,
    load_config_or_fallback,
)
from adaptive_quant.online_pipeline import run_online_pipeline_entrypoint
from adaptive_quant.presets.online import CONFIG_ONLINE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online adaptation pipeline: offline warm-start, simulated serving, replay updates, and rollback."
    )
    add_config_file_argument(parser)
    add_config_override_arguments(parser)
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Override the number of online requests (defaults to config.online_requests).",
    )
    args = parser.parse_args()
    cfg = apply_config_overrides(load_config_or_fallback(args.config, CONFIG_ONLINE), args)
    run_online_pipeline_entrypoint(cfg, request_count=args.requests)


if __name__ == "__main__":
    main()
