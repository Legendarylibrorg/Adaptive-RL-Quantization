# Architecture Guide

This repository is organized around one goal: **reproducible adaptive-quantization research without hidden runtime branches**.

The code is intentionally split into a small number of layers so experiments stay auditable:

## Design principles

- **Config-first**: every meaningful run should be reconstructible from `FrameworkConfig` or a JSON/TOML file.
- **Hash-chained replay**: with `replay_manifest_enabled`, each JSONL step is chained (`_integrity_hash`) and summarized in `*_replay_manifest.json` (`config_sha256`, per-step `step_sha256`, `chain_head_sha256`) for audit and simulator re-verification (`adaptive-rl-quant-replay`).
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
- `run_sweep.py`: hyperparameter grid search and trial ranking
- `run_calibrate_llama_cpp.py`: simulator calibration from local `llama.cpp` measurements
- `run_route_learning.py`: GGUF route catalog and contextual bandit workflow
- `run_replay.py`: hash-chained JSONL replay / audit verification

Root `run_*.py` files prepend `src/` on `sys.path` and delegate to `adaptive_quant.cli`; installed commands call the same modules directly. From repo root, `./run` (or `make run`) starts the default simulator pipeline without activating a venv when `.venv` exists.

Path bootstrap for source checkouts:

- [`src/bootstrap.py`](../src/bootstrap.py): shared `ensure_repo_paths()` used by [`_repo_entrypoint.py`](../_repo_entrypoint.py) and [`src/analysis/__main__.py`](../src/analysis/__main__.py)
- [`tests/__init__.py`](../tests/__init__.py): lets `python3 -m unittest discover -s tests -t .` run without `pip install -e .`

Installed console commands map onto those wrappers:

- `adaptive-rl-quant`
- `adaptive-rl-quant-pytorch`
- `adaptive-rl-quant-moe`
- `adaptive-rl-quant-online`
- `adaptive-rl-quant-multiseed`
- `adaptive-rl-quant-sweep`
- `adaptive-rl-quant-calibrate`
- `adaptive-rl-quant-route`
- `adaptive-rl-quant-replay`
- `adaptive-rl-quant-analyze` / `python -m analysis`

## 2. Configuration layer

- `src/adaptive_quant/configuration/`: canonical experiment contract (`FrameworkConfig`, validation)
- `src/adaptive_quant/easy_config.py`: JSON/TOML loading and preset layering
- `src/adaptive_quant/presets/`: curated `FrameworkConfig` presets (`CONFIG`, `CONFIG_GPU`, `CONFIG_MOE`, …)
- `src/config.py`: single top-level export surface after `pip install -e .` (`from config import CONFIG_4090`, etc.)

Legacy per-preset `config_*.py` shim modules were removed; import named constants from `config` or `adaptive_quant.presets` directly.

This layer defines reproducibility knobs such as:

- seeds
- prompt sampling modes
- train/eval determinism
- benchmark budgets
- output paths

## 3. Core runtime

- `src/adaptive_quant/environment.py`: prompt + hardware state construction and reward evaluation
- `src/adaptive_quant/policy.py`: stdlib policy and checkpointable policy state
- `src/adaptive_quant/trainer.py`: stdlib trainer
- `src/adaptive_quant/torch_trainer.py`, `src/adaptive_quant/torch_policy.py`: PyTorch backend
- `src/adaptive_quant/backend.py` (facade) / `src/adaptive_quant/backends/`: simulator and `llama.cpp` measurement backends
- `src/adaptive_quant/benchmark.py`: fixed benchmark comparisons
- `src/adaptive_quant/recommendation.py`: deterministic recommendation pass
- `src/adaptive_quant/online_learning.py`, `src/adaptive_quant/online_pipeline.py`: online adaptation flow

The key architecture rule here is: **different backends share the same `FrameworkConfig` and artifact layout**.

## 4. Analysis and reporting

- `src/analysis/`: post-hoc analysis (`analyzers.py`, shared `log_records.py`, `python -m analysis` CLI)
- `src/adaptive_quant/research_pipeline.py`: full offline pipeline orchestration
- `src/adaptive_quant/experiment_aggregate.py`: shared numeric flattening/aggregation for multiseed and sweep
- `src/adaptive_quant/sweep.py`: hyperparameter grid expansion, trial naming, ranking
- `src/adaptive_quant/pipeline/`: VCS stamp, benchmark warnings, Markdown report helpers
- `src/adaptive_quant/research_pipeline.py`: training-history writers, analysis runner, full offline orchestration
- `src/adaptive_quant/run_footer.py`: consistent CLI summaries

### Routing modules (do not conflate)

Two separate “route” concepts share reward/hardware context but differ in arms and learning rule:

| Module | Purpose | Learner |
| --- | --- | --- |
| `routing.py` | In-run **backend / model_id** selection when `router_enabled` on `FrameworkConfig` | Stdlib policy-gradient + value baseline (hash or optional HF embeddings) |
| `model_routes.py` + `route_policy.py` + `route_pipeline.py` | Offline **GGUF catalog** workflow (`adaptive-rl-quant-route`): Hub repos, quant labels, local paths | Contextual **UCB1** bandit per (hardware, domain, complexity) bucket |

Use `routing.py` for experiments that pick among configured `router_routes` during training. Use the route catalog stack for comparing downloadable GGUF variants and persisting a bandit table next to `outputs/routes/catalog.json`.

### GPU profiles vs host hardware

| Module | Role |
| --- | --- |
| `gpu_profiles.py` | **Training** overrides (`torch_*` batch sizes, preflight budgets) keyed by `infer_gpu_profile`; also `SIMULATOR_PROFILE_TUNING` for simulator fidelity |
| `hardware.py` | **Runtime detection** (`detect_host_hardware`) and host-aware `HardwareProfile` construction for the env/reward path |

Profile names (`rtx4090`, `consumer_8gb`, …) are shared; training overrides and simulator tuning values are intentionally different tables.

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
6. tune hyperparameters with `adaptive-rl-quant-sweep` when comparing learning rates, reward weights, or torch batch settings ([SWEEP.md](SWEEP.md))
7. keep generated summaries, reports, and config files together in version control or your experiment log

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
