# Running Guide

This guide explains the supported entrypoints and what each one does.

## Which mode should I use?

| Goal | Recommended command | Notes |
| --- | --- | --- |
| Reproduce the main offline research path | `python3 run_research.py` | Canonical baseline and best default starting point. |
| Reproduce the canonical MoE research path | `python3 run_moe_research.py` | Enables packed expert variants and MoE benchmark comparisons. |
| Calibrate simulator against llama.cpp | `python3 run_calibrate_llama_cpp.py` | Fits simulator multipliers from measured latency/throughput (requires llama.cpp binary + model). |
| Train on a 4090 and learn a universal policy | `python3 run_4090_universal.py` | Explicit 4090-host universal-policy preset. |
| Run the optimized Linux RTX 4090 path | `bash scripts/run_4090_pipeline.sh` | Linux NVIDIA host (recommended). |
| Run CUDA training on another NVIDIA GPU | `python3 run_pytorch_gpu.py` | Auto-detects a GPU profile. |
| Explore continual adaptation | `python3 run_online_learning.py` | Experimental extension. |

## Primary entrypoints

Canonical offline research run:

```bash
python3 run_research.py
```

Multi-seed dense run (recommended for more meaningful numbers):

```bash
python3 run_multiseed.py --preset dense --seeds 13,17,23,29,31
```

Notes:

- Seed syntax supports `a,b,c` and ranges like `0-9`.
- Multi-seed runs write an aggregate report to `outputs/reports/<run_name>_multiseed_report.md` and per-seed reports to `outputs/reports/<run_name>_seed<seed>_report.md`.

Canonical MoE research run:

```bash
python3 run_moe_research.py
```

Multi-seed MoE run:

```bash
python3 run_multiseed.py --preset moe --seeds 13,17,23
```

4090-host universal policy run:

```bash
python3 run_4090_universal.py
```

Fixed RTX 4090 PyTorch run:

```bash
python3 run_pytorch_4090.py
```

Experimental online adaptation run:

```bash
python3 run_online_learning.py
```

## Secondary helpers

Generic GPU PyTorch run:

```bash
python3 run_pytorch_gpu.py
```

Recommended 4090 validation + run:

```bash
bash scripts/run_4090_pipeline.sh
```

Tests:

```bash
python3 -m unittest discover -s tests -v
```

## What `run_research.py` does

`run_research.py` uses [`config.py`](../config.py) and:

1. trains the default policy
2. evaluates it
3. runs the benchmark suite
4. writes analysis outputs

This is the easiest path for:

- checking that the project works
- iterating on logic without CUDA
- validating the simulator pipeline
- reproducing the core research setup

## What `run_pytorch_gpu.py` does

`run_pytorch_gpu.py` uses [`config_gpu.py`](../config_gpu.py) and:

1. detects the current CUDA device
2. selects a tuned GPU profile
3. runs a CUDA preflight
4. writes a preflight JSON report
5. trains with the PyTorch actor-critic and PPO-style updates
6. evaluates the trained policy
7. runs benchmarks using a smaller benchmark budget
8. writes analysis outputs

This is the recommended path for most NVIDIA GPUs.

## What `run_moe_research.py` does

`run_moe_research.py` uses [`config_moe.py`](../config_moe.py) and:

1. enables the packed-expert-bank MoE path
2. trains the MoE-aware policy
3. runs MoE benchmark comparisons
4. writes MoE expert and cache analysis outputs
5. writes the same training history, checkpoint, and report artifacts as the standard research path

## What `run_4090_universal.py` does

`run_4090_universal.py` uses [`config_4090_universal.py`](../config_4090_universal.py) and:

1. trains on a 4090 host
2. keeps `multi_hardware=True`
3. targets `gpu`, `cpu`, and `low_resource` hardware modes
4. writes outputs under a dedicated universal-policy run name
5. uses the same CUDA preflight, training, benchmarks, and reports as the standard 4090 path

## What `run_pytorch_4090.py` does

`run_pytorch_4090.py` uses [`config_4090.py`](../config_4090.py) and:

1. runs a CUDA preflight
2. writes a preflight JSON report
3. trains with the PyTorch actor-critic and PPO-style updates
4. evaluates the trained policy
5. runs benchmarks using a smaller benchmark budget
6. writes analysis outputs

This is the path intended for a fixed RTX 4090 preset.

The shared research pipeline also writes:

- a training-history JSON file
- a final checkpoint
- a markdown experiment report

## What `run_online_learning.py` does

`run_online_learning.py` uses [`config_online.py`](../config_online.py) and:

1. bootstraps the policy with an offline simulator training phase
2. simulates a live multi-hardware request stream
3. logs online telemetry and replay records
4. applies replay-based policy updates
5. runs canary checks before serving exploratory decisions
6. rolls back to the best recent policy snapshot if live reward drifts too far
7. writes an online analysis summary

This is an optional systems extension for continual-improvement experiments. It is not the main path for the paper’s offline research claims.

## Where outputs go

Logs:

- `outputs/logs/*.jsonl`

Benchmarks and summaries:

- `outputs/benchmarks/*_benchmarks.json`
- `outputs/benchmarks/*_summary.json`
- `outputs/benchmarks/*_preflight.json`
- `outputs/benchmarks/*_online_summary.json`
- `outputs/checkpoints/*`
- `outputs/reports/*`

Analysis:

- `outputs/analysis/<run_name>/hardware`
- `outputs/analysis/<run_name>/inputs`
- `outputs/analysis/<run_name>/quant`
- `outputs/analysis/<run_name>/online`

## Typical workflows

Validate the repository quickly:

```bash
python3 -m unittest discover -s tests -v
python3 run_research.py
```

Run the full generic GPU path:

```bash
python3 run_pytorch_gpu.py
```

After the run, inspect:

- `outputs/benchmarks/adaptive_universal_policy_torch_gpu_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch_gpu_summary.json`

Recommended 4090 path on Linux:

```bash
bash scripts/run_4090_pipeline.sh
```

After the run, inspect:

- `outputs/benchmarks/adaptive_universal_policy_torch4090_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_summary.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_training_history.json`
- `outputs/checkpoints/adaptive_universal_policy_torch4090_final.pt`
- `outputs/reports/adaptive_universal_policy_torch4090_report.md`

Run the fixed 4090 path directly:

```bash
python3 run_pytorch_4090.py
```

After the run, inspect:

- `outputs/benchmarks/adaptive_universal_policy_torch4090_preflight.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_summary.json`
- `outputs/benchmarks/adaptive_universal_policy_torch4090_training_history.json`
- `outputs/checkpoints/adaptive_universal_policy_torch4090_final.pt`
- `outputs/reports/adaptive_universal_policy_torch4090_report.md`

Run the continual online path:

```bash
python3 run_online_learning.py
```

After the run, inspect:

- `outputs/benchmarks/adaptive_online_policy_online_summary.json`
- `outputs/benchmarks/adaptive_online_policy_summary.json`
- `outputs/analysis/adaptive_online_policy/online`

Run with llama.cpp backend:

Edit [`config.py`](../config.py), [`config_gpu.py`](../config_gpu.py), or [`config_4090.py`](../config_4090.py):

- set `backend="llama_cpp"`
- set `llama_cpp_binary`
- set `llama_cpp_model`

Then rerun the same entrypoint you want.

## Runtime notes

- The simulator path is deterministic enough for development and tests.
- The GPU path is optimized for throughput, but the environment rollout itself still happens in Python because the task is simulator- and decision-heavy.
- Benchmarks intentionally use a smaller training budget than the main GPU run so the comparison suite stays practical.
- If you care most about stable, meaningful research results, prefer `run_research.py` and the PyTorch GPU entrypoints over the experimental online loop.
- If you want the clearest repo tour, think in this order: `run_research.py`, then `run_pytorch_4090.py` or `scripts/run_4090_pipeline.sh`, then `run_online_learning.py` only if needed.
- If you want the MoE-focused version of the research story, insert `run_moe_research.py` right after `run_research.py`.
- If you want the clearest “4090 host, universal policy target” workflow, use `run_4090_universal.py`.
