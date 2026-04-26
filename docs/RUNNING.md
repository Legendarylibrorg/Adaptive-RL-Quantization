# Running Guide

**Platform:** The simulator entrypoints work on **Linux, macOS, and Windows**. Linux and WSL2 are the primary command paths in this guide. On Windows, substitute `py -3.11` or `python` where a `python3` helper command still appears. Run from the **repository root** (where `pyproject.toml` and `config.py` live).

1. Install: `python3 -m pip install -e .`, or run **`python3 scripts/setup_from_clone.py`** once (see [INSTALL.md](INSTALL.md)) — creates a venv, bootstraps `pip` when needed, then runs tests + RL smoke. Editable installs expose console commands such as `adaptive-rl-quant` and `adaptive-rl-quant-pytorch`.
2. Short reproducible E2E (no Python edits): [**`config.e2e_smoke.json`**](../config.e2e_smoke.json) — `adaptive-rl-quant --config config.e2e_smoke.json`
3. More examples: [CONFIG.md](CONFIG.md), [`config.example.json`](../config.example.json).

Artifacts and API: [USAGE.md](USAGE.md).

## Command conventions

- Prefer the installed console commands in user-facing workflows: `adaptive-rl-quant`, `adaptive-rl-quant-pytorch`, `adaptive-rl-quant-online`, and friends.
- Source-checkout equivalents (`python3 run_research.py`, `python3 run_pytorch.py`, and so on) remain available when you want to run directly from the repo tree.
- Analysis helpers under `analysis/` are still Python scripts and are intentionally invoked as `python3 analysis/...`.

## Choose an entrypoint

| Goal | Command |
| --- | --- |
| Main offline run (simulator, no PyTorch) | `adaptive-rl-quant` |
| Short reproducible E2E (tune episodes / seed in JSON) | `adaptive-rl-quant --config config.e2e_smoke.json` |
| … with your own file | `adaptive-rl-quant --config path.json` or `-c path.toml` |
| MoE preset | `adaptive-rl-quant-moe` |
| MoE + file | `adaptive-rl-quant-moe --config moe.json` |
| NVIDIA GPU (auto profile) | `adaptive-rl-quant-pytorch --preset gpu` |
| GPU + file (**replaces** `--preset`) | `adaptive-rl-quant-pytorch --config cuda_run.toml` |
| RTX 3090 preset | `adaptive-rl-quant-pytorch --preset 3090` or `make 3090` |
| RTX 4090 preset | `adaptive-rl-quant-pytorch --preset 4090` |
| Linux 4090 checks + run | `bash scripts/run_4090_pipeline.sh` |
| 4090 host, universal-policy naming | `adaptive-rl-quant-pytorch --preset 4090-universal` |
| Multi-seed (`dense` or `moe`) | `adaptive-rl-quant-multiseed --preset dense --seeds 13,17,23,29,31` |
| Calibrate simulator from llama.cpp | `adaptive-rl-quant-calibrate` |
| Calibrate + custom base config | `adaptive-rl-quant-calibrate --config my_base.json` |
| Online experiment | `adaptive-rl-quant-online` |
| Online + file | `adaptive-rl-quant-online --config online.toml` |

## Commands

```bash
adaptive-rl-quant
adaptive-rl-quant --config ./my_settings.json
adaptive-rl-quant-moe
adaptive-rl-quant-pytorch --preset gpu
adaptive-rl-quant-pytorch --preset 3090
adaptive-rl-quant-pytorch --config ./gpu_settings.toml
adaptive-rl-quant-online --config ./online.toml
adaptive-rl-quant-multiseed --preset moe --seeds 13,17,23
adaptive-rl-quant-calibrate --config ./paths_only.json
adaptive-rl-quant --help
adaptive-rl-quant-online --help
adaptive-rl-quant-pytorch --help
python3 -m unittest discover -s tests -v
```

Source checkouts can still call `python3 run_*.py` directly; the installed console commands are the public interface to prefer in user-facing docs.

Source-checkout equivalents:

```bash
python3 run_research.py --config ./my_settings.json
python3 run_moe_research.py
python3 run_pytorch.py --preset gpu
python3 run_pytorch.py --preset 3090
python3 run_online_learning.py --config ./online.toml
python3 run_multiseed.py --preset moe --seeds 13,17,23
python3 run_calibrate_llama_cpp.py --config ./paths_only.json
```

Before committing (whitespace, syntax, tests): `python3 scripts/pre_commit_check.py` on Unix-like hosts; on Windows use `py -3.11 scripts/pre_commit_check.py` or `python scripts/pre_commit_check.py`. On Linux/macOS, `bash scripts/pre_commit_check.sh` is a wrapper around the same Python implementation.

Multi-seed: seeds can be `a,b,c` or `0-9`. Reports under `outputs/reports/`.

Fixed horizons and episode counts live in each `config*.py`. For long PyTorch runs, enable `continuous_training` and related fields in [CONFIG.md](CONFIG.md).

## What every full run does

`adaptive-rl-quant` (via [`run_research.py`](../run_research.py)) and the GPU/MoE entrypoints call the shared **research pipeline**: train → evaluate → benchmarks → analysis (JSON + SVG) → optional Markdown report and checkpoints. Exact files depend on `run_name` and backend; see [USAGE.md](USAGE.md).

- **`adaptive-rl-quant-pytorch`** (via [`run_pytorch.py`](../run_pytorch.py)): CUDA preflight first (when enabled), then the same pipeline with a smaller benchmark budget than training.
- **`adaptive-rl-quant-moe`** (via [`run_moe_research.py`](../run_moe_research.py)): MoE benchmarks and extra MoE analysis.
- **`adaptive-rl-quant-online`** (via [`run_online_learning.py`](../run_online_learning.py)): offline warm-start, then simulated serving + replay + rollback with the same summary/report/checkpoint pattern as other entrypoints.

## Outputs

`outputs/logs/`, `outputs/benchmarks/` (including `*_preflight.json` on GPU), `outputs/analysis/<run_name>/`, `outputs/checkpoints/`, `outputs/reports/`.

## llama.cpp

Set `backend="llama_cpp"`, `llama_cpp_binary`, and `llama_cpp_model` in the preset you use ([`config.py`](../config.py), [`config_gpu.py`](../config_gpu.py), etc.), then run the same entrypoint.

## Post-hoc analysis

Regenerate plots from existing logs without retraining: [USAGE.md](USAGE.md) (scripts under `analysis/`).

## Notes

- Tests do not require PyTorch; GPU entrypoints do.
- Benchmarks use a smaller episode budget than main training on purpose.
