# Usage Guide

## Platform and working directory

- **Simulator path:** supported on **Linux, macOS, and Windows**. Work in the **repo root** (directory with `pyproject.toml` and `src/`). Python presets live under **`src/config*.py`** (or use `from config import CONFIG` after `pip install -e .`).
- **Windows:** use `py -3.11` or `python` instead of `python3`, and venv `Scripts\activate`.
- **GPU workflows:** remain **Linux-first**.
- **WSL2:** recommended on Windows when you want Linux shell, Makefile, and GPU-oriented workflow parity.

Install and system packages: [INSTALL.md](INSTALL.md). Command reference: [RUNNING.md](RUNNING.md).

## Basics

1. `python3 -m pip install -e .` (or `./setup.sh` for venv + tests + smoke)
2. `python3 -m unittest discover -s tests -t . -q` (works from a source checkout without install — see [CONTRIBUTING.md](../CONTRIBUTING.md))
3. `adaptive-rl-quant`, `./run`, or `adaptive-rl-quant --config my.json`

Cross-platform shortcuts: `python3 scripts/setup_from_clone.py` and `python3 scripts/pre_commit_check.py` on Unix-like hosts; on Windows use `py -3.11` or `python`.

**Dependencies:** Core package has **no required PyPI deps** ([`pyproject.toml`](../pyproject.toml)). Optional extras: `torch` (often CPU-only via PyPI), `hub` (route downloads), `router` (HF embedding router), `dev` (Ruff, coverage). GPU entrypoints need a CUDA `torch` wheel — run `python3 scripts/install_cuda_torch.py` on Linux + NVIDIA (or a manual `cu130`/`cu126` wheel), then `pip install -e .` and any extras (`[router]` for HF routing features).

## Configuration without editing Python

- Copy [`config.example.json`](../config.example.json) or add a `.toml` file.
- Run: `adaptive-rl-quant -c ./conf.toml`
- Details and `preset` keys: [CONFIG.md](CONFIG.md).

**API examples:**

```python
from adaptive_quant import load_config, quick_config, build_trainer
from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from config import CONFIG

run_pipeline_entrypoint(CONFIG)

cfg = load_config("my.json")
trainer = build_trainer(cfg)

cfg2 = quick_config(run_name="ablation", training_episodes=500)
```

Or `ResearchPipeline(cfg).run()`, `FrameworkConfig.from_file`, `FrameworkConfig.from_mapping`. Heavy symbols on `import adaptive_quant` (e.g. `Trainer`) load lazily — see [`__init__.py`](../src/adaptive_quant/__init__.py) (package source under `src/adaptive_quant/`).

## Outputs

Everything lands under `outputs/`: `logs/`, `benchmarks/` (summaries + optional `*_preflight.json` + `*_recommendation.json` + online detail JSON), `analysis/<run_name>/`, `checkpoints/`, `reports/`. Names follow `run_name` and path fields in config.

Multi-seed runs write `<run_name>_multiseed_summary.json` and `<run_name>_multiseed_report.md`. Hyperparameter sweeps write `<run_name>_sweep_summary.json` (leaderboard + per-trial metadata) and `<run_name>_sweep_report.md`, with one full pipeline summary per trial at `<base_run_name>_trialNNN_*_summary.json`. See **[SWEEP.md](SWEEP.md)** for sweep file format, objectives, and examples.

The recommendation artifact records detected host hardware, the target hardware class used for scoring, adaptive-policy performance on that target, and the best fixed quant candidate discovered from deterministic RL rollouts.

For `adaptive-rl-quant-online` (or `run_online_learning.py` from a source checkout), you also get `*_online_telemetry.jsonl`, `*_online_replay.jsonl`, `*_online_summary.json`, a standard `*_summary.json`, bootstrap `*_training_history.json`, and a Markdown report under `reports/`.

## Re-run analysis (no training)

From repo root (after `pip install -e .`, or from a source checkout via `run_*.py` / `python -m analysis`), pass **command**, **log or history path**, then **output dir**:

```bash
python -m analysis hardware_generalization path/to/*_multi_hw.jsonl out/
python -m analysis input_adaptation path/to/*_dynamic.jsonl out/
python -m analysis quant_function_behavior path/to/*_learned.jsonl out/
```

All analysis logic lives in [`analysis/analyzers.py`](../src/analysis/analyzers.py); invoke commands via `python -m analysis` (see command list with `--help`).
