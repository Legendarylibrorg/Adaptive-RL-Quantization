# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Bumped optional runtime pins (`torch`, `transformers`, `safetensors`) and dev `ruff` floor in `pyproject.toml`.
- Refreshed Docker base image digest (`python:3.12-slim-bookworm`) and pinned `dependency-review-action` with `fail-on-severity: high`.
- Grouped Dependabot updates for GitHub Actions, CI bootstrap, and optional Python extras.

### Security

- Restored `dependency-review-action` v5.0.0 (was briefly pinned to v4.9.0 on the PR branch).
- Reject `..` and control characters on llama.cpp runtime paths, route-catalog `local_path`, and `llama_cpp:` router routes.
- Cap episode / replay counters loaded from config at `MAX_EPISODE_COUNT` (1,000,000).
- Ignore `HF_CLI` overrides whose path contains `..`; skip `..` segments when parsing HF download stdout paths.
- Cap structural config integers (`num_layers`, torch dims, llama.cpp context, MoE topology, etc.) to block JSON/TOML memory DoS.
- Optional `ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES` env var restricts resolved `llama_cpp_binary` to allowed directory roots.
- CI `pip-audit` job scans hash-pinned bootstrap requirements (`requirements/ci.txt`).

## [0.1.0] - 2026-04-26

### Added

- Initial public release: simulator-first RL quantization research loop (`adaptive-rl-quant` and friends).
- Optional PyTorch/CUDA training path, MoE, online loop, multiseed aggregation, and llama.cpp calibration entrypoints.
- JSON/TOML `FrameworkConfig` loading, hash-verified CI bootstrap, and dependency review workflow.

[0.1.0]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/releases/tag/v0.1.0
[Unreleased]: https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/compare/v0.1.0...HEAD
