# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Hardware-aware **setup tests** (`src/adaptive_quant/setup_tests.py`, `scripts/run_setup_tests.py`): `./setup.sh` runs a host-specific unittest subset; `--full-tests` runs full `unittest discover`.
- NVIDIA **secure boundary** at startup (`src/adaptive_quant/nvidia_secure_boundary.py`, `scripts/nvidia_secure_startup.py`) for Linux hosts where `nvidia-smi` reports a GPU — see [docs/SECURE_RUN.md](docs/SECURE_RUN.md).
- [`docs/SWEEP.md`](docs/SWEEP.md): hyperparameter sweep guide (grid vs trials, objectives, artifacts, Makefile shortcuts).
- Root `./run` script and shared [`src/bootstrap.py`](src/bootstrap.py) for source-checkout path setup; [`tests/__init__.py`](tests/__init__.py) so `unittest discover -s tests -t .` works without an editable install.

### Changed

- Pipeline output: slim `analysis` in `*_summary.json`, narrative Markdown reports (tables + takeaways instead of JSON dumps), analysis log-path fallback, and actionable `decision` block on recommendations.
- `./setup.sh` no longer enforces the NVIDIA secure boundary on Linux GPU hosts (simulator bootstrap only); boundary remains on CUDA install, PyTorch CLI, and `run_4090_pipeline.sh`. Linux venv/Python version failures now print distro-specific hints.
- `scripts/run_4090_pipeline.sh`: runs setup tests on CPU by default (`RUN_TESTS=1`); `RUN_TESTS=full` runs the full suite; `RUN_TESTS=0` skips tests before the GPU preset.
- Slim `run_*.py` shims; removed redundant Unix shell wrappers around `setup_from_clone.py`, `pre_commit_check.py`, and `secret_scan.py` (use the Python scripts directly).
- CI, Makefile, and docs now use `python3 -m unittest discover -s tests -t .` consistently.

### Security

- Hash-pin CI dev tools (including `pip-audit`), and PyTorch CPU smoke via `pip-compile` lockfiles (`requirements/dev.txt`, `pytorch-cpu.txt`); `verify_lockfiles.py` enforces inline hashes; scheduled weekly `pip-audit` on main; bootstrap `setup_from_clone.py` uses `--require-hashes` for setuptools.
- Cap router/online prompt text; validate analysis CLI paths; bound paper-bundle digest reads; skip binary extensions in `secret_scan.py`; Docker `INSTALL_EXTRAS=torch` uses hash-pinned `pytorch-cpu.txt`.
- Drop redundant `safetensors` pin and `compat_tomllib` shim; stop re-exporting `subprocess` from `adaptive_quant.backend`.
- Reject Hugging Face route ids passed as `llama_cpp_model_path` overrides; clarify online router GGUF path via `RouteCandidate.llama_cpp_model_path()`.
- Harden Hugging Face model selection: require `router_hf_allowed_models` + pinned `router_hf_embedding_revision` for the HF router backend; validate `org/name` repo ids; GGUF filenames must end in `.gguf`; optional repo allowlists via `route_hf_allowed_repos` / `ADAPTIVE_RL_HF_ALLOWED_REPOS`.
- Restored `dependency-review-action` v5.0.0 (was briefly pinned to v4.9.0 on the PR branch).
- Reject `..` and control characters on llama.cpp runtime paths, route-catalog `local_path`, and `llama_cpp:` router routes.
- Cap episode / replay counters loaded from config at `MAX_EPISODE_COUNT` (1,000,000).
- Ignore `HF_CLI` overrides whose path contains `..`; skip `..` segments when parsing HF download stdout paths.
- Cap structural config integers (`num_layers`, torch dims, llama.cpp context, MoE topology, etc.) to block JSON/TOML memory DoS.
- Cap `recommendation_*`, `llama_cpp_generate_tokens`, `jsonl_flush_every`, and `llama_cpp_cache_max_entries` at config load.
- Optional `ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES` env var restricts resolved `llama_cpp_binary` to allowed directory roots.
- CI `pip-audit` job scans hash-pinned bootstrap requirements (`requirements/ci.txt`).

### Added

- Nested `FrameworkConfig` sections (`artifacts`, `moe`, `llama_cpp`, `torch`, `online`, `router`, `training`) with flat JSON/TOML key compatibility; `config_to_flat_dict()` for pipeline summaries.
- `adaptive-rl-quant-analyze` console entry (`analysis.__main__:main`).
- `hardware.detect_cuda_device()` for shared CUDA name/VRAM probing.
- Analysis warnings when JSONL logs are missing or empty after phase filter.
- Analysis unit tests (`tests/test_analysis_analyzers.py`) and optional real llama.cpp integration test (`ADAPTIVE_RL_RUN_LLAMA_CPP=1`).
- Unit tests for multiseed aggregation (`tests/test_multiseed_aggregation.py`), hyperparameter sweep (`tests/test_sweep.py`, `tests/test_experiment_aggregate.py`), online guardrails (`tests/test_guardrails.py`), and torch trainer helpers (`tests/test_torch_trainer.py`).
- `adaptive-rl-quant-sweep` console entry (`adaptive_quant.cli.sweep:main`) with grid search via `--vary` or [`config.sweep.example.json`](../config.sweep.example.json); shared aggregation helpers in `experiment_aggregate.py`.
- Direct unit coverage for reward math, features, backends (quality/simulator), pipeline helpers (`report_markdown`, `research_pipeline`), consolidated `config` exports, and CLI wiring (`tests/test_reward.py`, `test_features.py`, `test_backends_unit.py`, `test_pipeline_unit.py`, `test_presets_and_shims.py`, `test_cli_behavior.py`). `pre_commit_check.py` unittest step uses `discover -s tests -t .` (matches CI editable install).
- CI job `torch-cpu-smoke` (Ubuntu 3.12, hash-pinned `requirements/pytorch-cpu.txt`) runs torch trainer unit smoke without the full matrix.
- CI coverage gate (72% floor on `adaptive_quant`).
- Secure-run tooling: `docs/SECURE_RUN.md` tiers (VM → Docker → NVIDIA), `scripts/docker_secure_preflight.sh`, `scripts/docker_gpu_device_probe.py`, `config.docker.gpu_smoke.json`, Makefile `docker-gpu-verify` (local/VM use; not run in CI).

### Removed

- Orphaned `config.pytorch_smoke.json` (CI no longer runs a separate PyTorch CPU smoke job; use `config.example.pytorch.toml` locally).
- `adaptive_quant.runner_cli` re-export shim; use `adaptive_quant.cli.common` instead.

### Changed

- Split stdlib policy heads into `policy_heads.py`; expanded mypy scope (policy, environment, reward, analysis).
- Enforce **ruff** (lint + format) and **mypy** (configuration, logging, easy_config, backends, `route_pipeline`, CLI, `torch_trainer`) in `pre_commit_check.py` and CI (`pip install -e ".[dev]"`).
- **API:** `extract_numeric` is the public llama.cpp metric parser helper (`adaptive_quant.backends.llama_cpp`); removed `_extract_numeric` from `adaptive_quant.backend.__all__`. `git_commit_hash` is imported from `adaptive_quant.pipeline.vcs` (no longer listed in `research_pipeline.__all__`).
- `compat_tomllib` always uses stdlib `tomllib` (Python 3.11+ only; minimal fallback removed from the public path).
- CONFIG guide: prefer JSON/TOML for shared/CI runs; Python presets for local iteration.
- Bumped optional runtime pins (`torch`, `transformers`, `safetensors`) and dev `ruff` floor in `pyproject.toml`.
- Refreshed Docker base image digest (`python:3.12-slim-bookworm`) and pinned `dependency-review-action` with `fail-on-severity: high`.
- Grouped Dependabot updates for GitHub Actions, CI bootstrap, and optional Python extras.

## [0.1.0] - 2026-04-26

### Added

- Initial public release: simulator-first RL quantization research loop (`adaptive-rl-quant` and friends).
- Optional PyTorch/CUDA training path, MoE, online loop, multiseed aggregation, and llama.cpp calibration entrypoints.
- JSON/TOML `FrameworkConfig` loading, hash-verified CI bootstrap, and dependency review workflow.

[0.1.0]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/releases/tag/v0.1.0
[Unreleased]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/compare/v0.1.0...HEAD
