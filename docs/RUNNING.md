# Running Guide

**Platform:** The simulator entrypoints work on **Linux, macOS, and Windows**. Commands below use `python3`; on Windows substitute `py -3.11` or `python` as needed. Run from the **repository root** (where `pyproject.toml` and `config.py` live).

1. Install: `python3 -m pip install -e .`, or run **`python3 scripts/setup_from_clone.py`** once (see [INSTALL.md](INSTALL.md)) — creates a venv, bootstraps `pip` when needed, then runs tests + RL smoke.
2. Short reproducible E2E (no Python edits): [**`config.e2e_smoke.json`**](../config.e2e_smoke.json) — `python3 run_research.py --config config.e2e_smoke.json`
3. More examples: [CONFIG.md](CONFIG.md), [`config.example.json`](../config.example.json).

Artifacts and API: [USAGE.md](USAGE.md).

## Choose an entrypoint

| Goal | Command |
| --- | --- |
| Main offline run (simulator, no PyTorch) | `python3 run_research.py` |
| Short reproducible E2E (tune episodes / seed in JSON) | `python3 run_research.py --config config.e2e_smoke.json` |
| … with your own file | `python3 run_research.py --config path.json` or `-c path.toml` |
| MoE preset | `python3 run_moe_research.py` |
| MoE + file | `python3 run_moe_research.py --config moe.json` |
| NVIDIA GPU (auto profile) | `python3 run_pytorch.py --preset gpu` |
| GPU + file (**replaces** `--preset`) | `python3 run_pytorch.py --config cuda_run.toml` |
| RTX 4090 preset | `python3 run_pytorch.py --preset 4090` |
| Linux 4090 checks + run | `bash scripts/run_4090_pipeline.sh` |
| 4090 host, universal-policy naming | `python3 run_pytorch.py --preset 4090-universal` |
| Multi-seed (`dense` or `moe`) | `python3 run_multiseed.py --preset dense --seeds 13,17,23,29,31` |
| Calibrate simulator from llama.cpp | `python3 run_calibrate_llama_cpp.py` |
| Calibrate + custom base config | `python3 run_calibrate_llama_cpp.py --config my_base.json` |
| Online experiment | `python3 run_online_learning.py` |
| Online + file | `python3 run_online_learning.py --config online.toml` |

## Commands

```bash
python3 run_research.py
python3 run_research.py --config ./my_settings.json
python3 run_moe_research.py
python3 run_pytorch.py --preset gpu
python3 run_pytorch.py --config ./gpu_settings.toml
python3 run_multiseed.py --preset moe --seeds 13,17,23
python3 run_research.py --help
python3 run_pytorch.py --help
python3 -m unittest discover -s tests -v
```

Before committing (whitespace, syntax, tests): `python3 scripts/pre_commit_check.py` on Unix-like hosts; on Windows use `py -3.11 scripts/pre_commit_check.py` or `python scripts/pre_commit_check.py`. On Linux/macOS, `bash scripts/pre_commit_check.sh` is a wrapper around the same Python implementation.

Multi-seed: seeds can be `a,b,c` or `0-9`. Reports under `outputs/reports/`.

Fixed horizons and episode counts live in each `config*.py`. For long PyTorch runs, enable `continuous_training` and related fields in [CONFIG.md](CONFIG.md).

## What every full run does

[`run_research.py`](../run_research.py) (and GPU/MoE entrypoints) call the shared **research pipeline**: train → evaluate → benchmarks → analysis (JSON + SVG) → optional Markdown report and checkpoints. Exact files depend on `run_name` and backend; see [USAGE.md](USAGE.md).

- **`run_pytorch.py`**: CUDA preflight first (when enabled), then the same pipeline with a smaller benchmark budget than training.
- **`run_moe_research.py`**: MoE benchmarks and extra MoE analysis.
- **`run_online_learning.py`**: offline warm-start, then simulated serving + replay + rollback (optional extension).

## Outputs

`outputs/logs/`, `outputs/benchmarks/` (including `*_preflight.json` on GPU), `outputs/analysis/<run_name>/`, `outputs/checkpoints/`, `outputs/reports/`.

## llama.cpp

Set `backend="llama_cpp"`, `llama_cpp_binary`, and `llama_cpp_model` in the preset you use ([`config.py`](../config.py), [`config_gpu.py`](../config_gpu.py), etc.), then run the same entrypoint.

## Post-hoc analysis

Regenerate plots from existing logs without retraining: [USAGE.md](USAGE.md) (scripts under `analysis/`).

## Notes

- Tests do not require PyTorch; GPU entrypoints do.
- Benchmarks use a smaller episode budget than main training on purpose.
