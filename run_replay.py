#!/usr/bin/env python3
"""CLI wrapper: hash-based replay / verify (see adaptive_quant.cli.replay)."""

from __future__ import annotations

from _repo_entrypoint import main_for_script, run_script_main

main = main_for_script(__file__)

if __name__ == "__main__":
    run_script_main(__file__)
