# Running Guide

**Platform:** The simulator entrypoints work on **Linux, macOS, and Windows**. Linux and WSL2 are the primary command paths in this guide. On Windows, substitute `py -3.11` or `python` where a `python3` helper command still appears. Run from the **repository root** (where `pyproject.toml` lives; presets are defined in [`src/adaptive_quant/presets/`](../src/adaptive_quant/presets/) and re-exported from [`src/config.py`](../src/config.py)).

1. Install: `python3 -m pip install -e .`, or run **`./setup.sh`** / **`python3 scripts/setup_from_clone.py`** once (see [INSTALL.md](INSTALL.md)) — creates a venv, bootstraps `pip` when needed, then runs **hardware-aware setup tests** + E2E smoke (`config.e2e_smoke.json`). Use **`--full-tests`** for the full unittest suite; **`--quick`** skips tests and smoke. Editable installs expose console commands such as `adaptive-rl-quant` and `adaptive-rl-quant-pytorch`.
2. Default full run after setup: **`./run`** or `make run` (uses the venv CLI when present).
3. Short reproducible E2E (no Python edits): [**`config.e2e_smoke.json`**](../config.e2e_smoke.json) — `adaptive-rl-quant --config config.e2e_smoke.json`
4. More examples: [CONFIG.md](CONFIG.md), [`config.example.json`](../config.example.json).

Artifacts and API: [USAGE.md](USAGE.md).

## Command conventions

- Prefer the installed console commands in user-facing workflows: `adaptive-rl-quant`, `adaptive-rl-quant-pytorch`, `adaptive-rl-quant-online`, and friends.
- After setup, **`./run`** (or `make run`) starts the default simulator pipeline without activating the venv.
- Source-checkout equivalents (`python3 run_research.py`, `python3 run_pytorch.py`, and so on) work without `pip install -e .` — each shim prepends `src/` via [`_repo_entrypoint.py`](../_repo_entrypoint.py).
- Post-hoc analysis: **`python -m analysis <command> ...`** after `pip install -e .` or from a source checkout — see [USAGE.md](USAGE.md).

## Choose an entrypoint

| Goal | Command |
| --- | --- |
| Main offline run (simulator, no PyTorch) | `adaptive-rl-quant` or `./run` after setup |
| Short one-off episode override | `adaptive-rl-quant --training-episodes 500 --evaluation-episodes 100` |
| Short reproducible E2E (tune episodes / seed in JSON) | `adaptive-rl-quant --config config.e2e_smoke.json` |
| … with your own file | `adaptive-rl-quant --config path.json` or `-c path.toml` |
| MoE preset | `adaptive-rl-quant-moe` |
| MoE + file | `adaptive-rl-quant-moe --config moe.json` |
| NVIDIA GPU setup (Linux) | `python3 scripts/install_cuda_torch.py` or `make install-torch-cuda` |
| NVIDIA GPU (auto profile) | `adaptive-rl-quant-pytorch --preset gpu` |
| GPU + file (**replaces** `--preset`) | `adaptive-rl-quant-pytorch --config cuda_run.toml` |
| RTX 3090 preset | `adaptive-rl-quant-pytorch --preset 3090` or `make 3090` |
| RTX 4090 preset | `adaptive-rl-quant-pytorch --preset 4090` |
| Linux 4090 checks + setup tests + run | `bash scripts/run_4090_pipeline.sh` (`RUN_TESTS=0` to skip tests; `RUN_TESTS=full` for full unittest) |
| 4090 host, universal-policy naming | `adaptive-rl-quant-pytorch --preset 4090-universal` |
| Long routed RL post-training | `adaptive-rl-quant-pytorch --preset post-train` or `make post-train` |
| Multi-seed (`dense` or `moe`) | `adaptive-rl-quant-multiseed --preset dense --seeds 13,17,23,29,31` |
| Hyperparameter sweep | `adaptive-rl-quant-sweep --sweep-config config.sweep.example.json` |
| Hyperparameter sweep (CLI grid) | `adaptive-rl-quant-sweep --config config.e2e_smoke.json --vary learning_rate=0.02,0.035` |
| Calibrate simulator from llama.cpp | `adaptive-rl-quant-calibrate` |
| Calibrate + custom base config | `adaptive-rl-quant-calibrate --config my_base.json` |
| GGUF route catalog + contextual bandit | `adaptive-rl-quant-route --catalog outputs/routes/catalog.json seed` |
| Route-bandit training | `adaptive-rl-quant-route --catalog outputs/routes/catalog.json train --config local_llama.json --iterations 128 --evaluate` |
| Online experiment | `adaptive-rl-quant-online` |
| Online + file | `adaptive-rl-quant-online --config online.toml` |
| Hash-chain replay / audit verify | `adaptive-rl-quant-replay --config path.json` |
| DPO / preference alignment (optional `[alignment]`) | `adaptive-rl-quant-alignment --sft-model PATH --dataset PATH` |
| Post-hoc log / history analysis | `adaptive-rl-quant-analyze` or `python -m analysis` |

## Commands

```bash
adaptive-rl-quant
adaptive-rl-quant --config ./my_settings.json
adaptive-rl-quant --training-episodes 500 --evaluation-episodes 100 --run-name quick_ablation
adaptive-rl-quant --set training.learning_rate=0.02 --set reward_weights.beta_throughput=0.08
adaptive-rl-quant-moe
adaptive-rl-quant-pytorch --preset gpu
adaptive-rl-quant-pytorch --preset 3090
adaptive-rl-quant-pytorch --config ./gpu_settings.toml
adaptive-rl-quant-online --config ./online.toml
adaptive-rl-quant-multiseed --preset moe --seeds 13,17,23
adaptive-rl-quant-sweep --sweep-config config.sweep.example.json
adaptive-rl-quant-sweep --preset dense --vary learning_rate=0.02,0.035 --episodes 48
adaptive-rl-quant-calibrate --config ./paths_only.json
adaptive-rl-quant-route --catalog outputs/routes/catalog.json --help
adaptive-rl-quant --help
adaptive-rl-quant-online --help
adaptive-rl-quant-pytorch --help
python3 -m unittest discover -s tests -t . -v
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
python3 run_sweep.py --sweep-config config.sweep.example.json
python3 run_calibrate_llama_cpp.py --config ./paths_only.json
python3 run_route_learning.py --catalog outputs/routes/catalog.json --help
```

Before committing (whitespace, syntax, tests): `python3 scripts/pre_commit_check.py` on Unix-like hosts; on Windows use `py -3.11 scripts/pre_commit_check.py` or `python scripts/pre_commit_check.py`.

Multi-seed: seeds can be `a,b,c` or `0-9`. Reports under `outputs/reports/` as `<run_name>_multiseed_report.md`.

Hyperparameter sweep: pass a sweep file (`--sweep-config`) or repeat `--vary KEY=val1,val2` for a cartesian grid. Each trial runs the full research pipeline with unique `run_name` suffixes; results are ranked by `--objective` (default `evaluation.mean_reward`) and written to `outputs/benchmarks/<run_name>_sweep_summary.json` and `outputs/reports/<run_name>_sweep_report.md`. Quick smoke: `make sweep-smoke`. Full guide: **[SWEEP.md](SWEEP.md)**.

Fixed horizons and episode counts live in presets under `src/adaptive_quant/presets/` (or JSON/TOML `--config` files). For long PyTorch runs, use `--preset post-train` or enable `continuous_training` in [CONFIG.md](CONFIG.md).

Startup overrides are available on research-style entrypoints. Use named flags for common fields (`--training-episodes`, `--evaluation-episodes`, `--benchmark-training-episodes`, `--benchmark-evaluation-episodes`, `--run-name`, `--seed`) and repeat `--set KEY=VALUE` for tuning fields such as torch batch sizes or reward weights. `VALUE` is parsed with bounded JSON when possible. **Privileged keys** (backend, llama.cpp, router/HF allowlists, checkpoints) require `ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES=1`; prefer `--config` for those. Summaries record applied overrides under `security_audit.cli_startup_overrides`.

## What every full run does

`adaptive-rl-quant` (via [`run_research.py`](../run_research.py)) and the GPU/MoE entrypoints call the shared **research pipeline**: train → evaluate → benchmarks → analysis (JSON + SVG) → optional Markdown report and checkpoints. Exact files depend on `run_name` and backend; see [USAGE.md](USAGE.md).

- **`adaptive-rl-quant-pytorch`** (via [`run_pytorch.py`](../run_pytorch.py)): CUDA preflight first (when enabled), then the same pipeline with a smaller benchmark budget than training.
- **`adaptive-rl-quant-moe`** (via [`run_moe_research.py`](../run_moe_research.py)): MoE benchmarks and extra MoE analysis.
- **`adaptive-rl-quant-online`** (via [`run_online_learning.py`](../run_online_learning.py)): offline warm-start, then simulated serving + replay + rollback with the same summary/report/checkpoint pattern as other entrypoints.
- **`adaptive-rl-quant-multiseed`** (via [`run_multiseed.py`](../run_multiseed.py)): repeated full pipelines across seeds with mean/std aggregates.
- **`adaptive-rl-quant-sweep`** (via [`run_sweep.py`](../run_sweep.py)): cartesian hyperparameter grids or explicit trial lists, ranked by a pipeline objective metric.
- **`adaptive-rl-quant-route`** (via [`run_route_learning.py`](../run_route_learning.py)): manages GGUF route catalogs and trains a contextual bandit over route choices. See [ROUTES.md](ROUTES.md) and [LOCAL_RESEARCH.md](LOCAL_RESEARCH.md).

## Outputs

Under `outputs/` by default: `logs/`, `benchmarks/` (including `*_preflight.json` on GPU), `analysis/<run_name>/`, `checkpoints/`, `reports/`, `paper_bundles/<run_name>/`, optional `gguf/`, plus route workflow dirs `routes/` and `models/`. Relocate via config or multiseed/sweep `--outputs-dir` — see [CONFIG.md](CONFIG.md#output-paths).

## llama.cpp

Set `backend="llama_cpp"`, `llama_cpp_binary`, and `llama_cpp_model` in the preset you use ([`src/config.py`](../src/config.py) or a JSON/TOML `--config` file), then run the same entrypoint.

## Post-hoc analysis

Regenerate plots from existing logs without retraining: [USAGE.md](USAGE.md) (`python -m analysis`).

## Notes

- Tests do not require PyTorch; GPU entrypoints do.
- Benchmarks use a smaller episode budget than main training on purpose.
