# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Hash-pin CI dev tools (including `pip-audit`), and PyTorch CPU smoke via `pip-compile` lockfiles (`requirements/dev.txt`, `pytorch-cpu.txt`); `verify_lockfiles.py` enforces inline hashes; scheduled weekly `pip-audit` on main.
- Cap router/online prompt text; validate analysis CLI paths; bound paper-bundle digest reads; skip binary extensions in `secret_scan.py`; Docker `INSTALL_EXTRAS=torch` uses hash-pinned `pytorch-cpu.txt`.
- Drop redundant `safetensors` pin and `compat_tomllib` shim; stop re-exporting `subprocess` from `adaptive_quant.backend`.

### Added

- Analysis unit tests (`tests/test_analysis_analyzers.py`) and optional real llama.cpp integration test (`ADAPTIVE_RL_RUN_LLAMA_CPP=1`).
- CI: PyTorch CPU smoke job (`config.pytorch_smoke.json`) and coverage gate (68% floor on `adaptive_quant`).

### Security

- Reject Hugging Face route ids passed as `llama_cpp_model_path` overrides; clarify online router GGUF path via `RouteCandidate.llama_cpp_model_path()`.
- Harden Hugging Face model selection: require `router_hf_allowed_models` + pinned `router_hf_embedding_revision` for the HF router backend; validate `org/name` repo ids; GGUF filenames must end in `.gguf`; optional repo allowlists via `route_hf_allowed_repos` / `ADAPTIVE_RL_HF_ALLOWED_REPOS`.

### Changed

- Enforce **ruff** (lint + format) and **mypy** (configuration, logging, easy_config) in `pre_commit_check.py` and CI (`pip install -e ".[dev]"`).
- `compat_tomllib` always uses stdlib `tomllib` (Python 3.11+ only; minimal fallback removed from the public path).
- CONFIG guide: prefer JSON/TOML for shared/CI runs; Python presets for local iteration.
- Bumped optional runtime pins (`torch`, `transformers`, `safetensors`) and dev `ruff` floor in `pyproject.toml`.
- Refreshed Docker base image digest (`python:3.12-slim-bookworm`) and pinned `dependency-review-action` with `fail-on-severity: high`.
- Grouped Dependabot updates for GitHub Actions, CI bootstrap, and optional Python extras.

### Security

- Restored `dependency-review-action` v5.0.0 (was briefly pinned to v4.9.0 on the PR branch).
- Reject `..` and control characters on llama.cpp runtime paths, route-catalog `local_path`, and `llama_cpp:` router routes.
- Cap episode / replay counters loaded from config at `MAX_EPISODE_COUNT` (1,000,000).
- Ignore `HF_CLI` overrides whose path contains `..`; skip `..` segments when parsing HF download stdout paths.
- Cap structural config integers (`num_layers`, torch dims, llama.cpp context, MoE topology, etc.) to block JSON/TOML memory DoS.
- Cap `recommendation_*`, `llama_cpp_generate_tokens`, `jsonl_flush_every`, and `llama_cpp_cache_max_entries` at config load.
- Optional `ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES` env var restricts resolved `llama_cpp_binary` to allowed directory roots.
- CI `pip-audit` job scans hash-pinned bootstrap requirements (`requirements/ci.txt`).

## [0.1.0] - 2026-04-26

### Added

- Initial public release: simulator-first RL quantization research loop (`adaptive-rl-quant` and friends).
- Optional PyTorch/CUDA training path, MoE, online loop, multiseed aggregation, and llama.cpp calibration entrypoints.
- JSON/TOML `FrameworkConfig` loading, hash-verified CI bootstrap, and dependency review workflow.

[0.1.0]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/releases/tag/v0.1.0
[Unreleased]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/compare/v0.1.0...HEAD
