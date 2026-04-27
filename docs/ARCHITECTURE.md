# Architecture Guide

This repository is organized around one goal: **reproducible adaptive-quantization research without hidden runtime branches**.

The code is intentionally split into a small number of layers so experiments stay auditable:

## Design principles

- **Config-first**: every meaningful run should be reconstructible from `FrameworkConfig` or a JSON/TOML file.
- **One artifact contract**: entrypoints write structured outputs under `outputs/` so analysis and reports do not depend on ad-hoc filenames.
- **Linux-first operations**: CUDA and low-level tooling are designed around Linux; Windows users should prefer **WSL2** for parity with the Linux workflow.
- **Simulator-first evidence**: the simulator path is the stable baseline for repeatable experiments; optional PyTorch and `llama.cpp` paths extend it without changing the config surface.
- **No hidden notebook logic**: training, benchmarking, analysis, and reporting live in versioned Python modules and CLIs.

## Repository layers

## 1. Entrypoints

- `run_research.py`: default simulator pipeline
- `run_pytorch.py`: same pipeline with the PyTorch trainer
- `run_moe_research.py`: MoE-focused preset and benchmarks
- `run_online_learning.py`: online adaptation pipeline
- `run_multiseed.py`: repeated runs and aggregation
- `run_calibrate_llama_cpp.py`: simulator calibration from local `llama.cpp` measurements
- `run_route_learning.py`: GGUF route catalog and contextual bandit workflow

These files should stay thin wrappers around package code.

Installed console commands map onto those wrappers:

- `adaptive-rl-quant`
- `adaptive-rl-quant-pytorch`
- `adaptive-rl-quant-moe`
- `adaptive-rl-quant-online`
- `adaptive-rl-quant-multiseed`
- `adaptive-rl-quant-calibrate`
- `adaptive-rl-quant-route`

## 2. Configuration layer

- `adaptive_quant/configuration.py`: canonical experiment contract
- `adaptive_quant/easy_config.py`: JSON/TOML loading and preset layering
- `config.py`, `config_*.py`: curated Python presets

This layer defines reproducibility knobs such as:

- seeds
- prompt sampling modes
- train/eval determinism
- benchmark budgets
- output paths

## 3. Core runtime

- `adaptive_quant/environment.py`: prompt + hardware state construction and reward evaluation
- `adaptive_quant/policy.py`: stdlib policy and checkpointable policy state
- `adaptive_quant/trainer.py`: stdlib trainer
- `adaptive_quant/torch_trainer.py`, `adaptive_quant/torch_policy.py`: PyTorch backend
- `adaptive_quant/backend.py`: simulator and `llama.cpp` measurement backends
- `adaptive_quant/benchmark.py`: fixed benchmark comparisons
- `adaptive_quant/recommendation.py`: deterministic recommendation pass
- `adaptive_quant/online_learning.py`, `adaptive_quant/online_pipeline.py`: online adaptation flow

The key architecture rule here is: **different backends share the same `FrameworkConfig` and artifact layout**.

## 4. Analysis and reporting

- `analysis/`: CLI wrappers for post-hoc analysis
- `analysis/analyzers.py`: shared analysis logic
- `adaptive_quant/research_pipeline.py`: full offline pipeline orchestration
- `adaptive_quant/run_footer.py`: consistent CLI summaries

Reports are intended to be derived from machine-readable outputs, not handwritten after the fact.

## 5. Tooling and ops

- `scripts/setup_from_clone.py`: bootstrap from a fresh clone
- `scripts/pre_commit_check.py`: syntax, hashes, tests, and scans
- `scripts/env_report.py`: environment diagnosis (`make doctor`)
- `Makefile`: Linux/WSL2-oriented convenience commands

## Research-grade workflow

For strong experimental hygiene, prefer this order:

1. `python3 scripts/setup_from_clone.py`
2. `adaptive-rl-quant --config config.e2e_smoke.json`
3. move to a dedicated JSON/TOML config or a copied Python preset
4. use `FrameworkConfig.reproducible_research(...)` or the `reproducible` preset when you need deterministic scheduling
5. validate with `adaptive-rl-quant-multiseed` before making comparative claims
6. keep generated summaries, reports, and config files together in version control or your experiment log

## Linux-first and WSL2

For simulator work:

- native Linux is the reference environment
- macOS remains supported
- Windows is supported, but **WSL2 is the recommended path** if you want the Linux layout, shell scripts, and GPU-oriented workflow

For GPU work:

- native Linux remains the primary target
- **WSL2 is the best-supported Windows route**
- keep the repo inside the Linux filesystem (for example `~/src/...`), not under `/mnt/c/...`, to avoid I/O penalties

## Architecture boundaries

This repository aims to produce **research-grade artifacts**, not a hosted inference service.

That means the repo is optimized for:

- reproducible runs
- benchmark comparisons
- calibration
- analysis
- reports

It is not optimized for:

- multi-user serving
- secret management for production deployments
- distributed training orchestration
- API hosting

Those boundaries are intentional and help keep the research code understandable instead of overloaded.
